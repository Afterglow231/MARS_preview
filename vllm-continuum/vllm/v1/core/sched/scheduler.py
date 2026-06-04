# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any, Optional, Union, Tuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.core.estimate_with_func import (InterceptionRecorder, ToolCallEstimator)

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.interception_recorder = InterceptionRecorder()
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        elif self.scheduler_config.policy == "continuum":
            self.policy = SchedulingPolicy.CONTINUUM
        elif self.scheduler_config.policy == "continuum_dy":
            self.policy = SchedulingPolicy.CONTINUUM_DY
        # Implement Autellix based on PLAS
        elif self.scheduler_config.policy == "autellix":
            self.policy = SchedulingPolicy.AUTELLIX
        elif self.scheduler_config.policy == "infercept":
            self.policy = SchedulingPolicy.INFERCEPT
        elif self.scheduler_config.policy == "mars":
            self.policy = SchedulingPolicy.MARS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # MARS policy state (only used when self.policy == SchedulingPolicy.MARS)
        self._mars_level_by_req_id: dict[str, int] = {}
        self._mars_last_scheduled_by_req_id: dict[str, float] = {}
        self._mars_level_quanta: tuple[int, ...] = (2048, 1024, 512, 256)
        self._mars_aging_interval_s: float = 30.0
        # Window admission control for MARS.
        self._mars_active_window_size: int = 30
        self._mars_overflow = create_request_queue(SchedulingPolicy.FCFS)

        self._autellix_service_thresholds: tuple[int, ...] = (2048, 8192, 32768)
        self._autellix_level_quanta: tuple[int, ...] = (2048, 1024, 512, 256)
        self._autellix_beta: float = 2.0
        self._autellix_program_service: dict[str, int] = defaultdict(int)
        self._autellix_program_wait: dict[str, int] = defaultdict(int)
        # Optional tail fallback: switch from Autellix to FCFS after N
        # finished jobs (counted by job_id). Disabled by default.
        self._autellix_fcfs_after_n_finished = max(
            0,
            int(self.scheduler_config.autellix_tail_fcfs_after_finished),
        )
        self._autellix_finished_job_ids: set[str] = set()
        self._autellix_fcfs_mode: bool = False

        # Initialize ToolCallEstimator with tokenizer config
        self.tool_call_estimator = ToolCallEstimator(
            model_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            tokenizer_revision=vllm_config.model_config.tokenizer_revision,
        )

        # TODO(Hanchen) This stored the list of pineed requests and the time they need to be removed
        self.pinned_requests: list[Tuple[Request, float]] = []

        # Recorded for continuum_dy
        self._continuum_evicted_at: dict[str, float] = {}
        self._continuum_queue_delay_samples: deque[float] = deque(maxlen=200)
        self._continuum_program_served: dict[str, int] = defaultdict(int)
        self._continuum_program_k_history: dict[str, list[int]] = {}
        self._continuum_eta_n: int = 0
        self._continuum_eta_sum_x: float = 0.0
        self._continuum_eta_sum_y: float = 0.0
        self._continuum_eta_sum_x2: float = 0.0
        self._continuum_eta_sum_y2: float = 0.0
        self._continuum_eta_sum_xy: float = 0.0
        # Track the first entry time for each job_id in running queue (for job_id level FCFS)
        self.running_job_id_first_entry_time: dict[str] = {}
        # Track prefill start time for throughput measurement
        self.request_prefill_start_time: dict[str, float] = {}
        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

    def pop_running_request_based_on_last_step(self, request: Request) -> tuple[Request, bool]:
        """Pop a request from running queue based on job_id level FCFS and last step."""
        if len(self.running) <= 1:
            # If no running request can be preempted, release the longest retained KV.
            latest_pin_end_request = None
            latest_pin_end_time = -float('inf')
            for req, end_time in self.pinned_requests:
                if end_time > latest_pin_end_time:
                    latest_pin_end_time = end_time
                    latest_pin_end_request = req
            if latest_pin_end_request is not None:
                self.pinned_requests.remove((latest_pin_end_request, latest_pin_end_time))
                return latest_pin_end_request, True

            raise IndexError("pop from empty running queue")
                
        # First, find the request that is not last step
        latest_request = None
        latest_entry_time = -float('inf')
        
        for req in self.running:
            job_entry_time = self.running_job_id_first_entry_time.get(req.job_id)
            if job_entry_time > latest_entry_time and not req.is_last_step:
                latest_entry_time = job_entry_time
                latest_request = req
        
        if latest_request is not None:
            self.running.remove(latest_request)
            return latest_request, False

        # Second, check the other requests
        for req in self.running:
            job_entry_time = self.running_job_id_first_entry_time.get(req.job_id)
            if job_entry_time > latest_entry_time:
                latest_entry_time = job_entry_time
                latest_request = req
        
        if latest_request is not None:
            self.running.remove(latest_request)
            return latest_request, False
    
    def pin_request(self, request: Request, length_of_pin: float) -> None:
        self.interception_recorder.request_pinned(request)
        self.pinned_requests.append((request, time.time() + length_of_pin))

    def unpin_request(self, request: Request, end_time: float) -> None:
        self.pinned_requests.remove((request, end_time))
        self.interception_recorder.request_unpinned(request)
        self.kv_cache_manager.free(request)

    # TODO (Hanchen) this needs to be called at the beginning of each step to clean up pinned request based on system time
    # The LRU is handled by kv cache mangager through a reference counter
    def unpin_requests_regular(self) -> None:
        waiting_job_ids = [req.job_id for req in self.waiting]

        for request, end_time in list(self.pinned_requests):
            if request.job_id not in waiting_job_ids and time.time() >= end_time:
                self.unpin_request(request, end_time)

    def is_pinned(self, request: Request) -> bool:
        for req, _ in self.pinned_requests:
            if req.job_id == request.job_id:
                return True
        return False

    # Average queue-delay of continuum. Recorded for continuum_dy
    def _continuum_queue_delay_s(self) -> float:
        if not self._continuum_queue_delay_samples:
            return 0.0
        return sum(self._continuum_queue_delay_samples) / len(
            self._continuum_queue_delay_samples)

    def _continuum_eta(self) -> float:
        n = self._continuum_eta_n
        if n < 2:
            return 1.0

        sum_x = self._continuum_eta_sum_x
        sum_y = self._continuum_eta_sum_y
        sum_x2 = self._continuum_eta_sum_x2
        sum_y2 = self._continuum_eta_sum_y2
        sum_xy = self._continuum_eta_sum_xy

        num = n * sum_xy - sum_x * sum_y
        den_x = n * sum_x2 - sum_x * sum_x
        den_y = n * sum_y2 - sum_y * sum_y
        if den_x <= 0.0 or den_y <= 0.0:
            return 1.0
        corr = num / math.sqrt(den_x * den_y)
        corr = max(-1.0, min(1.0, corr))
        return -corr

    def _infercept_context_tokens(self, request: Request) -> int:
        return int(max(0, min(request.num_tokens, self.max_model_len)))

    def _infercept_other_running_tokens(self, request: Request) -> int:
        other_tokens = 0
        for running_request in self.running:
            if running_request.request_id == request.request_id:
                continue
            other_tokens += self._infercept_context_tokens(running_request)
        return other_tokens

    @staticmethod
    def _infercept_preserve_cost_s_tokens(*, hold_s: float,
                                          context_tokens: int) -> float:
        """
        Cost of retaining KV cache in Infercept.
        Infercept's per-token KV memory term M cancels out when comparing preserve and discard for the same model, so we calculate the cost in token-seconds.
        """
        return max(0.0, float(hold_s)) * max(0, int(context_tokens))

    @staticmethod
    def _infercept_chunked_discard_cost_s_tokens(
            *, prefill_reload_s: float, context_tokens: int,
            other_running_tokens: int) -> float:
        """Chunked-discard waste in token-seconds."""
        total_tokens = max(0, int(context_tokens)) + max(
            0, int(other_running_tokens))
        return max(0.0, float(prefill_reload_s)) * total_tokens / 2.0

    def _infercept_should_pin(self, request: Request) -> bool:
        """Tool-call detection."""
        if self.policy != SchedulingPolicy.INFERCEPT:
            return False
        if request.job_id is None:
            return False
        if request.is_last_step is not False:
            return False
        if request.this_func_call is None:
            return False
        return bool(str(request.this_func_call).strip())

    def _infercept_pin_ttl_s(self, request: Request) -> float:
        """
        Compute the KV-retention TTL from waste comparison.
        Preserve cost grows as TTL * C_i. Discard/recompute cost is estimated as Tfwd(C_i) * (C_i + C_other) / 2.
        """
        c_i = self._infercept_context_tokens(request)
        if c_i <= 0:
            return 0.0

        c_other = self._infercept_other_running_tokens(request)
        t_fwd = float(
            self.tool_call_estimator.estimate_prefill_reload_s(
                c_i, env_prefix="INFERCEPT"))
        discard_cost = self._infercept_chunked_discard_cost_s_tokens(
            prefill_reload_s=t_fwd,
            context_tokens=c_i,
            other_running_tokens=c_other,
        )

        if discard_cost <= 0.0:
            return 0.0

        ttl_s = discard_cost / float(c_i)
        preserve_cost = self._infercept_preserve_cost_s_tokens(
            hold_s=ttl_s,
            context_tokens=c_i,
        )
        logger.debug(
            "InferCept cost model request=%s job=%s tool=%s "
            "context_tokens=%d other_tokens=%d prefill_reload_s=%.6f "
            "preserve_cost_s_tokens=%.6f discard_cost_s_tokens=%.6f "
            "pin_ttl_s=%.6f",
            request.request_id,
            request.job_id,
            request.this_func_call,
            c_i,
            c_other,
            t_fwd,
            preserve_cost,
            discard_cost,
            ttl_s,
        )
        return max(0.0, ttl_s)

    def _mars_remaining_level(self, request: Request, *,
                              num_computed_tokens: Optional[int] = None) -> int:
        # Return the MARS priority level based on remaining prefill work. Smaller levels have higher priority.

        if num_computed_tokens is None:
            num_computed_tokens = request.num_computed_tokens

        # Treat decode as highest priority as decode-phase work is latency-sensitive and should run first.
        if num_computed_tokens >= request.num_prompt_tokens:
            return 0

        remaining_prompt = request.num_prompt_tokens - num_computed_tokens
        if remaining_prompt <= 2048:
            return 0
        if remaining_prompt <= 8192:
            return 1
        if remaining_prompt <= 32768:
            return 2
        return len(self._mars_level_quanta) - 1

    def _mars_effective_level(self, request: Request, now: float, *,
                              num_computed_tokens: Optional[int] = None) -> int:
        # Return the current MARS scheduling level, which combines two signals:
        # 1. feedback_level: how much service this request has already consumed.
        # 2. remaining_level: how much prefill work is still left.
        # The larger of the two is used as the base level, so either heavy remaining prefill work 
        # or prior over-service can lower the request's priority.
        # Aging then boosts requests that have waited for a while, preventing
        # starvation. Smaller levels mean higher scheduling priority.

        req_id = request.request_id
        feedback_level = self._mars_level_by_req_id.get(req_id, 0)
        base_level = max(
            feedback_level,
            self._mars_remaining_level(request,
                                       num_computed_tokens=num_computed_tokens),
        )

        if self._mars_aging_interval_s <= 0:
            return base_level

        last_scheduled = self._mars_last_scheduled_by_req_id.get(req_id)
        if last_scheduled is None:
            # Ensure a newly arrived request gets its initial level.
            last_scheduled = now - self._mars_aging_interval_s * (
                len(self._mars_level_quanta) + 1)

        boost = int((now - last_scheduled) // self._mars_aging_interval_s)
        return max(0, base_level - boost)

    def _mars_quantum_for_level(self, level: int) -> int:
        # Return the per-step token quantum for a MARS priority level.
        level = max(0, min(level, len(self._mars_level_quanta) - 1))
        quantum = self._mars_level_quanta[level]
        return min(quantum, self.max_num_scheduled_tokens)

    def _mars_reorder_running(self, now: float) -> None:
        # Reorder running requests according to the current MARS priority.
        def sort_key(request: Request) -> tuple[int, float, float]:
            req_id = request.request_id
            effective_level = self._mars_effective_level(request, now)
            last_scheduled = self._mars_last_scheduled_by_req_id.get(req_id)
            if last_scheduled is None:
                last_scheduled = now - self._mars_aging_interval_s * (
                    len(self._mars_level_quanta) + 1)
            return (effective_level, last_scheduled, request.arrival_time)

        self.running.sort(key=sort_key)

    def _mars_on_scheduled(self, request: Request, *, num_scheduled_tokens: int,
                           quantum: int, now: float) -> None:
        # Update MARS feedback state after a request is scheduled.
        req_id = request.request_id
        self._mars_last_scheduled_by_req_id[req_id] = now

        # Promote decode-phase requests.
        if request.num_computed_tokens >= request.num_prompt_tokens:
            self._mars_level_by_req_id[req_id] = 0
            return

        current_level = self._mars_level_by_req_id.get(req_id, 0)
        max_level = len(self._mars_level_quanta) - 1

        remaining_after = (
            request.num_tokens_with_spec + request.num_output_placeholders -
            request.num_computed_tokens - num_scheduled_tokens)
        if num_scheduled_tokens >= quantum and remaining_after > 0:
            self._mars_level_by_req_id[req_id] = min(current_level + 1,
                                                     max_level)

    def _mars_admit_overflow(self) -> None:
        # Move queued overflow requests into the MARS active window.
        if self.policy != SchedulingPolicy.MARS:
            return
        while (self._mars_overflow
               and (len(self.running) + len(self.waiting) <
                    self._mars_active_window_size)):
            request = self._mars_overflow.pop_request()
            # Treat as newly admitted: reset feedback state.
            self._mars_level_by_req_id[request.request_id] = 0
            self._mars_last_scheduled_by_req_id.pop(request.request_id, None)
            self.waiting.add_request(request)

    def _autellix_level_for_service(self, service: int) -> int:
        for level, threshold in enumerate(self._autellix_service_thresholds):
            if service < threshold:
                return level
        return len(self._autellix_service_thresholds)

    def _autellix_quantum_for_level(self, level: int) -> int:
        level = max(0, min(level, len(self._autellix_level_quanta) - 1))
        quantum = self._autellix_level_quanta[level]
        return min(quantum, self.max_num_scheduled_tokens)

    def _autellix_reorder_running(self) -> None:
        def sort_key(request: Request) -> tuple[int, float]:
            q_idx = int(getattr(request, "_autellix_q_idx", 0))
            return (q_idx, request.arrival_time)

        self.running.sort(key=sort_key)

    def _autellix_on_scheduled(self, request: Request, *,
                               num_scheduled_tokens: int, quantum: int) -> None:
        request._autellix_model_tokens = int(
            getattr(request, "_autellix_model_tokens", 0)) + int(
                num_scheduled_tokens)
        request._autellix_quanta_left = int(
            getattr(request, "_autellix_quanta_left", quantum)) - int(
                num_scheduled_tokens)

        remaining_after = (
            request.num_tokens_with_spec + request.num_output_placeholders -
            request.num_computed_tokens - num_scheduled_tokens)
        if num_scheduled_tokens >= quantum and remaining_after > 0:
            current_level = int(getattr(request, "_autellix_q_idx", 0))
            max_level = len(self._autellix_level_quanta) - 1
            new_level = min(current_level + 1, max_level)
            request._autellix_q_idx = new_level
            request._autellix_quanta_left = self._autellix_quantum_for_level(
                new_level)

    def _autellix_apply_antistarvation(self) -> None:
        if self.policy != SchedulingPolicy.AUTELLIX:
            return

        def should_promote(request: Request) -> bool:
            current_level = int(getattr(request, "_autellix_q_idx", 0))
            if current_level <= 0:
                return False
            pid = request.job_id or request.request_id
            wait_total = self._autellix_program_wait[pid] + int(
                getattr(request, "_autellix_wait_tokens", 0))
            service_total = self._autellix_program_service[pid] + int(
                getattr(request, "_autellix_model_tokens", 0))
            if service_total <= 0:
                return False
            return (wait_total / service_total) >= self._autellix_beta

        def promote(request: Request) -> None:
            request._autellix_q_idx = 0
            request._autellix_quanta_left = self._autellix_quantum_for_level(0)
            request._autellix_wait_tokens = 0
            request._autellix_model_tokens = 0

        waiting_to_promote = [r for r in self.waiting if should_promote(r)]
        for r in waiting_to_promote:
            self.waiting.remove_request(r)
            promote(r)
            self.waiting.add_request(r)

        for r in self.running:
            if should_promote(r):
                promote(r)

    def _autellix_maybe_switch_to_fcfs(self) -> None:
        if self.policy != SchedulingPolicy.AUTELLIX:
            return
        if self._autellix_fcfs_mode:
            return
        if self._autellix_fcfs_after_n_finished <= 0:
            return
        if len(self._autellix_finished_job_ids
               ) < self._autellix_fcfs_after_n_finished:
            return

        waiting_reqs = list(self.waiting)
        waiting_reqs.sort(key=lambda r: (r.arrival_time, r.request_id))
        new_waiting = create_request_queue(SchedulingPolicy.FCFS)
        for r in waiting_reqs:
            new_waiting.add_request(r)
        self.waiting = new_waiting

        self.running.sort(key=lambda r: (r.arrival_time, r.request_id))
        self.policy = SchedulingPolicy.FCFS
        self._autellix_fcfs_mode = True
        logger.info(
            "Autellix tail fallback: switched to FCFS after %d finished jobs "
            "(threshold=%d).",
            len(self._autellix_finished_job_ids),
            self._autellix_fcfs_after_n_finished,
        )
    
    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
        
        self.unpin_requests_regular()
        if self.policy == SchedulingPolicy.AUTELLIX:
            self._autellix_maybe_switch_to_fcfs()
        
        #Qiuyang (DEBUG) logging all running queue jobs and waiting queue jobs
        logger.debug(f"Running queue jobs: {[req.request_id for req in self.running]}")
        logger.debug(f"Waiting queue jobs: {[req.request_id for req in self.waiting]}")


        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # Admit overflow requests, then reorder running queue.
        if self.policy == SchedulingPolicy.MARS:
            self._mars_admit_overflow()
            if self.running:
                self._mars_reorder_running(scheduled_timestamp)

        elif self.policy == SchedulingPolicy.AUTELLIX:
            self._autellix_apply_antistarvation()
            if self.running:
                self._autellix_reorder_running()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)


            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            mars_quantum = None
            if self.policy == SchedulingPolicy.MARS:
                effective_level = self._mars_effective_level(
                    request, scheduled_timestamp)
                mars_quantum = self._mars_quantum_for_level(effective_level)
                # Apply the quantum cap during prefill to prevent a single
                # long prompt from consuming the whole batch token budget.
                if request.num_computed_tokens < request.num_prompt_tokens:
                    num_new_tokens = min(num_new_tokens, mars_quantum)

            autellix_quantum = None
            if self.policy == SchedulingPolicy.AUTELLIX:
                q_idx = int(getattr(request, "_autellix_q_idx", 0))
                quanta_left = int(getattr(request, "_autellix_quanta_left", 0))
                # Be defensive: this can become <=0 if a request was "unscheduled"
                # by preemption after _autellix_on_scheduled already decremented
                # it. vLLM's KV allocator requires num_new_tokens > 0.
                if quanta_left <= 0:
                    quanta_left = self._autellix_quantum_for_level(q_idx)
                    request._autellix_quanta_left = quanta_left
                autellix_quantum = min(
                    quanta_left,
                    self.max_num_scheduled_tokens,
                )
                num_new_tokens = min(num_new_tokens, autellix_quantum)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget
                 ) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_compute_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue
            
            logger.debug(
                f"Trying to schedule request {request.request_id} for {num_new_tokens} tokens"
            )
            attempt_tokens = num_new_tokens
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    attempt_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is not None:
                    logger.debug(f"New blocks: {new_blocks}")
                else:
                    logger.debug(f"New blocks is None")
                
                if new_blocks is None:
                    # If KV slot allocation fails, MARS will reduce prefill tokens by 1/2 and retry.
                    if (self.policy == SchedulingPolicy.MARS
                            and request.num_computed_tokens <
                            request.num_prompt_tokens
                            and attempt_tokens > self.block_size):
                        attempt_tokens = max(self.block_size,
                                             attempt_tokens // 2)
                        continue
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    is_unpin = False
                    preempted_index: Optional[int] = None
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        preempted_index = self.running.index(preempted_req)
                        self.running.remove(preempted_req)
                        self.interception_recorder.request_evicted_from_running_queue(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)

                    elif self.policy in (SchedulingPolicy.CONTINUUM,
                                          SchedulingPolicy.CONTINUUM_DY):
                        # Prefer evicting a request that can resume later rather than a final-step request.
                        preempted_req, is_unpin = self.pop_running_request_based_on_last_step(request)
                        
                        #TODO (Hanchen) we need to add a check unpin requests with the same job id.
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                        
                        self.interception_recorder.request_evicted_from_running_queue(preempted_req)
                    elif self.policy == SchedulingPolicy.MARS:
                        # MARS prioritizes the eviction of low-priority requests that consume significant KV resources, 
                        # while striving to avoid evicting decode requests.
                        def preempt_key(r: Request) -> tuple[int, int, float]:
                            level = max(
                                self._mars_level_by_req_id.get(
                                    r.request_id, 0),
                                self._mars_remaining_level(r),
                            )
                            # Avoid preempting decode requests unless needed.
                            if r.num_computed_tokens >= r.num_prompt_tokens:
                                level = -1
                            blocks_est = (
                                min(r.num_computed_tokens, self.max_model_len) +
                                self.block_size - 1) // self.block_size
                            return (level, blocks_est, r.arrival_time)

                        preempted_req = max(self.running, key=preempt_key)
                        preempted_index = self.running.index(preempted_req)
                        self.running.remove(preempted_req)
                        self.interception_recorder.request_evicted_from_running_queue(
                            preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    elif self.policy == SchedulingPolicy.AUTELLIX:
                        def preempt_key(r: Request) -> tuple[int, int, float]:
                            q_idx = int(getattr(r, "_autellix_q_idx", 0))
                            blocks_est = (
                                min(r.num_computed_tokens, self.max_model_len) +
                                self.block_size - 1) // self.block_size
                            return (q_idx, blocks_est, r.arrival_time)

                        preempted_req = max(self.running, key=preempt_key)
                        preempted_index = self.running.index(preempted_req)
                        self.running.remove(preempted_req)
                        self.interception_recorder.request_evicted_from_running_queue(
                            preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()
                        self.interception_recorder.request_evicted_from_running_queue(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)

                    # If we preempted a request that had already been scheduled
                    # in this step, we must "unschedule" it to keep
                    # total_num_scheduled_tokens consistent with the actual
                    # scheduled request set.
                    preempted_req_id = preempted_req.request_id
                    preempted_scheduled_tokens = num_scheduled_tokens.pop(
                        preempted_req_id, None)
                    if preempted_scheduled_tokens is not None:
                        token_budget += preempted_scheduled_tokens
                    req_to_new_blocks.pop(preempted_req_id, None)
                    scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                    scheduled_encoder_inputs.pop(preempted_req_id, None)

                    if (preempted_index is not None
                            and preempted_index < req_index):
                        req_index -= 1

                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    if is_unpin:
                        pass
                    else:
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
                        if self.log_stats:
                            preempted_req.record_event(
                                EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                        if self.policy == SchedulingPolicy.MARS:
                            self._mars_level_by_req_id[
                                preempted_req.request_id] = 0
                            self._mars_last_scheduled_by_req_id.pop(
                                preempted_req.request_id, None)

                        # Recorded for continuum_dy
                        self._continuum_evicted_at[preempted_req.request_id] = time.time()
                        self.waiting.prepend_request(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    num_new_tokens = attempt_tokens
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            if self.policy == SchedulingPolicy.MARS:
                # Reset the priority level and status of a MARS request after it is preempted.
                self._mars_on_scheduled(
                    request,
                    num_scheduled_tokens=num_new_tokens,
                    quantum=(mars_quantum if mars_quantum is not None else
                             num_new_tokens),
                    now=scheduled_timestamp,
                )
            elif self.policy == SchedulingPolicy.AUTELLIX:
                self._autellix_on_scheduled(
                    request,
                    num_scheduled_tokens=num_new_tokens,
                    quantum=(autellix_quantum if autellix_quantum is not None
                             else num_new_tokens),
                )

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        # TODO (Hanchen) need to add scheduling logic for returns from functions. It should not be FCFS
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break
                
                if self.policy == SchedulingPolicy.FCFS:
                    request = self.waiting.peek_request()
                elif self.policy == SchedulingPolicy.PRIORITY:
                    request = self.waiting.peek_request()
                elif self.policy == SchedulingPolicy.MARS:
                    # MARS currently utilizes peek_request().
                    request = self.waiting.peek_request()
                elif self.policy in (SchedulingPolicy.CONTINUUM,
                                     SchedulingPolicy.CONTINUUM_DY):
                    #The current implementation is basically giving priority to jobs with less prefill tokens.
                    request = self.waiting.peek_request(self.pinned_requests, self.kv_cache_manager, self.connector)
                elif self.policy in (SchedulingPolicy.AUTELLIX,
                                     SchedulingPolicy.INFERCEPT):
                    request = self.waiting.peek_request()
                else:
                    raise ValueError(f"Invalid policy: {self.policy}")

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        if self.policy in (SchedulingPolicy.CONTINUUM,
                                           SchedulingPolicy.CONTINUUM_DY):
                            self.waiting.pop_request(self.pinned_requests, self.kv_cache_manager, self.connector)
                        else:
                            self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        if self.policy in (SchedulingPolicy.CONTINUUM,
                                           SchedulingPolicy.CONTINUUM_DY):
                            self.waiting.pop_request(self.pinned_requests, self.kv_cache_manager, self.connector)
                        else:
                            self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # NOTE (Hanchen) The logic here is that we will see if the connector can get the tokens. 
                    # If it can, we will use them.
                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                        # NOTE (Hanchen) this will not be called in cpu offloading.
                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    mars_quantum_waiting = None
                    autellix_quantum_waiting = None
                    if self.policy == SchedulingPolicy.MARS:
                        # Apply MARS effective level and quantum to calculate num_new_tokens
                        effective_level = self._mars_effective_level(
                            request,
                            scheduled_timestamp,
                            num_computed_tokens=num_computed_tokens,
                        )
                        mars_quantum_waiting = self._mars_quantum_for_level(
                            effective_level)
                        if num_computed_tokens < request.num_prompt_tokens:
                            num_new_tokens = min(num_new_tokens,
                                                 mars_quantum_waiting)
                    elif self.policy == SchedulingPolicy.AUTELLIX:
                        q_idx = int(getattr(request, "_autellix_q_idx", 0))
                        quanta_left = int(
                            getattr(request, "_autellix_quanta_left", 0))
                        # Same defensive normalization as the RUNNING path.
                        if quanta_left <= 0:
                            quanta_left = self._autellix_quantum_for_level(
                                q_idx)
                            request._autellix_quanta_left = quanta_left
                        autellix_quantum_waiting = min(
                            quanta_left,
                            self.max_num_scheduled_tokens,
                        )
                        num_new_tokens = min(num_new_tokens,
                                             autellix_quantum_waiting)

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_compute_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_compute_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    assert ("whisper"
                            in self.vllm_config.model_config.model.lower()), (
                                "Whisper is the only supported "
                                "encoder-decoder model.")
                    num_encoder_tokens = MULTIMODAL_REGISTRY.\
                        get_encdec_max_encoder_len(
                        self.vllm_config.model_config)
                else:
                    num_encoder_tokens = 0

                # Allocate slots after the policy has selected this request.
                attempt_tokens = num_new_tokens
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        attempt_tokens + num_external_computed_tokens,
                        num_new_local_computed_tokens,
                        new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        delay_cache_blocks=load_kv_async,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                    if new_blocks is not None:
                        num_new_tokens = attempt_tokens
                        break

                    # When MARS cannot allocate KV slots, it reduces the prefill chunk by 1/2.
                    if (self.policy == SchedulingPolicy.MARS
                            and not load_kv_async
                            and num_computed_tokens < request.num_prompt_tokens
                            and attempt_tokens > self.block_size):
                        attempt_tokens = max(self.block_size,
                                             attempt_tokens // 2)
                        continue
                    break

                if new_blocks is None:
                    #print(f"Request {request.request_id} cannot be scheduled due to no slots")
                    # The request cannot be scheduled.
                    # TODO (Hanchen) need to add preemption logic here for CONTINUUM
                    if len(self.running) == 0 and self.pinned_requests:
                        if self.policy in (SchedulingPolicy.CONTINUUM,
                                           SchedulingPolicy.CONTINUUM_DY,
                                           SchedulingPolicy.INFERCEPT):
                            preempted_req, _ = self.pop_running_request_based_on_last_step(request)
                            if preempted_req in scheduled_running_reqs:
                                scheduled_running_reqs.remove(preempted_req)
                            self.kv_cache_manager.free(preempted_req)
                            self.encoder_cache_manager.free(preempted_req)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.

                self.waiting.remove_request(request)

                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    self.interception_recorder.request_waiting_to_running(
                        request, 
                        prompt_length=request.num_prompt_tokens,
                        hit_length=num_computed_tokens
                    )
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    self.interception_recorder.request_evicted_to_running(
                        request,
                        prompt_length=request.num_prompt_tokens,
                        hit_length=num_computed_tokens
                    )
                    # Recorded for continuum_dy.
                    evicted_at = self._continuum_evicted_at.pop(
                        request.request_id, None)
                    if evicted_at is not None:
                        self._continuum_queue_delay_samples.append(time.time() -
                                                                  evicted_at)
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                
                if self.policy == SchedulingPolicy.MARS:
                    # Updates MARS feedback state
                    self._mars_on_scheduled(
                        request,
                        num_scheduled_tokens=num_new_tokens,
                        quantum=(mars_quantum_waiting
                                 if mars_quantum_waiting is not None else
                                 num_new_tokens),
                        now=scheduled_timestamp,
                    )
                elif self.policy == SchedulingPolicy.AUTELLIX:
                    self._autellix_on_scheduled(
                        request,
                        num_scheduled_tokens=num_new_tokens,
                        quantum=(autellix_quantum_waiting
                                 if autellix_quantum_waiting is not None else
                                 num_new_tokens),
                    )

                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        if self.policy == SchedulingPolicy.AUTELLIX and total_num_scheduled_tokens > 0:
            scheduled_ids = set(num_scheduled_tokens.keys())
            for r in self.running:
                if r.request_id not in scheduled_ids:
                    r._autellix_wait_tokens = int(
                        getattr(r, "_autellix_wait_tokens",
                                0)) + total_num_scheduled_tokens
            for r in self.waiting:
                if r.request_id not in scheduled_ids:
                    r._autellix_wait_tokens = int(
                        getattr(r, "_autellix_wait_tokens",
                                0)) + total_num_scheduled_tokens
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(self.running,
                                     scheduled_spec_decode_tokens))

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )
        #print(f"scheduler_output: {scheduler_output}")
    
        # NOTE (Hanchen) this will handle the KVConnector
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[Optional[tuple[list[int], ...]]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used.")
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_hashes[i] in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(
                        request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                    request, i, encoder_compute_budget,
                    num_tokens_to_schedule):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_hashes[i])
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
        )

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to its index in the batch.
        # This will help us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[req.request_id] = i

        if not structured_output_request_ids:
            bitmask = None
        else:
            bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
        return structured_output_request_ids, bitmask

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
                draft_token_ids.req_ids,
                draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if not spec_token_ids:
                # NOTE(woosuk): request.spec_token_ids should be updated.
                request.spec_token_ids.clear()
            elif self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids)
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        if self.policy == SchedulingPolicy.MARS:
            # MARS has overflow requests.
            return len(self.running), len(self.waiting) + len(
                self._mars_overflow)
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.tool_call_estimator.request_arrives(request)
        self.interception_recorder.request_arrives(request)

        if request.job_id is not None:
            k = self._continuum_program_served[request.job_id]
            self._continuum_program_k_history.setdefault(request.job_id,
                                                        []).append(k)
        if self.policy == SchedulingPolicy.AUTELLIX:
            pid = request.job_id or request.request_id
            service = self._autellix_program_service[pid]
            request._autellix_service = service
            request._autellix_wait_tokens = 0
            request._autellix_model_tokens = 0
            q_idx = self._autellix_level_for_service(service)
            request._autellix_q_idx = q_idx
            request._autellix_quanta_left = self._autellix_quantum_for_level(
                q_idx)

        #print(f"Adding request {request.job_id} to waiting queue")
        #print(f"Request last_func_call: {request.last_func_call}")
        #print(f"Request is_last_step: {request.is_last_step}")
        #print(f"Request this_func_call: {request.this_func_call}")
        # Track the first entry time for this job_id if not already recorded
        if request.job_id not in self.running_job_id_first_entry_time:
            self.running_job_id_first_entry_time[request.job_id] = request.arrival_time

        if self.policy == SchedulingPolicy.MARS:
            # MARS admission window
            self._mars_level_by_req_id[request.request_id] = 0
            self._mars_last_scheduled_by_req_id.pop(request.request_id, None)
            active_cnt = len(self.running) + len(self.waiting)
            if active_cnt < self._mars_active_window_size:
                self.waiting.add_request(request)
            else:
                self._mars_overflow.add_request(request)
        else:
            self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            if self.policy == SchedulingPolicy.MARS:
                # Remove all requests.
                self._mars_overflow.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()
        self.tool_call_estimator.request_finished(request)
        self.interception_recorder.request_finished(request)

        if request.job_id is not None:
            self._continuum_program_served[request.job_id] += 1
            if request.is_last_step:
                total = self._continuum_program_served[request.job_id]
                ks = self._continuum_program_k_history.pop(request.job_id, [])
                for k in ks:
                    x = float(k)
                    y = float(total - k)
                    self._continuum_eta_n += 1
                    self._continuum_eta_sum_x += x
                    self._continuum_eta_sum_y += y
                    self._continuum_eta_sum_x2 += x * x
                    self._continuum_eta_sum_y2 += y * y
                    self._continuum_eta_sum_xy += x * y
                del self._continuum_program_served[request.job_id]

        if self.policy == SchedulingPolicy.AUTELLIX:
            pid = request.job_id or request.request_id
            service_at_arrival = int(getattr(request, "_autellix_service", 0))
            model_tokens = int(getattr(request, "_autellix_model_tokens", 0))
            wait_tokens = int(getattr(request, "_autellix_wait_tokens", 0))
            self._autellix_program_service[pid] = max(
                self._autellix_program_service[pid],
                service_at_arrival + model_tokens,
            )
            self._autellix_program_wait[pid] += wait_tokens
            if request.is_last_step:
                self._autellix_program_service.pop(pid, None)
                self._autellix_program_wait.pop(pid, None)
            if self._autellix_fcfs_after_n_finished > 0:
                # Count a "finished job" by job_id. Best-effort: treat an
                # explicit last-step marker as completion; otherwise, if the
                # model output does not look like a tool call, assume the job
                # will not resume.
                job_id = request.job_id or request.request_id
                is_done = bool(getattr(request, "is_last_step", False))
                if (not is_done and request.job_id is not None
                        and getattr(request, "this_func_call", None) is None):
                    is_done = True
                if is_done:
                    self._autellix_finished_job_ids.add(job_id)
        # NOTE (Hanchen) in unpin, we need to make sure it is not delay free blocks because it could be still waiting for transfer, 
        # need to copy something similar to the kv_xfer_params
        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
    
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        #NOTE (Hanchen) this is called when the request is finished
        for req, end_time in list(self.pinned_requests):
            if req.job_id == request.job_id:
                self.unpin_request(req, end_time)

        self._mars_level_by_req_id.pop(request.request_id, None)
        self._mars_last_scheduled_by_req_id.pop(request.request_id, None)
        
        # TODO (Hanchen) check if we want to pin this memory here for how long, pin them on scheduler level.
        if (self.policy in (SchedulingPolicy.CONTINUUM,
                            SchedulingPolicy.CONTINUUM_DY)
                and not request.is_last_step):
            if self.policy == SchedulingPolicy.CONTINUUM:
                length_of_pin = self.tool_call_estimator.set_up_pin(request)
            else:
                # Continuum_Dy: Calculate the optimal pinning time
                # 
                # Equation - Variable Name: 
                # 
                # Prefill-Reload(r): prefill_reload_s
                # η: eta
                # T: queue_delay_s
                # τ: tau
                # f: request.this_func_call
                # P(τ, f): p
                # 
                # Benefit(r) = T * η + Prefill-Reload(r): 
                # base = float(queue_delay_s) * float(eta) + float(prefill_reload_s)
                # 
                # P(τ, f) = 1 / |S[f]| * Σ I[t <= τ]:
                # p = sum(1.0 for t in records if t <= tau) / n
                # 
                # reward(τ): Optimal tau value
                # reward(τ) = P(τ, f) * (T * η + Prefill-Reload(r)) - τ

                length_of_pin = self.tool_call_estimator.calc_ttl_continuum_dynamic(
                    tool=request.this_func_call,
                    prefill_reload_s=self.tool_call_estimator.
                    estimate_prefill_reload_s(
                        request.num_tokens,
                        env_prefix="CONTINUUM",
                        fallback_env_prefixes=(),
                    ),
                    queue_delay_s=self._continuum_queue_delay_s(),
                    eta=self._continuum_eta(),
                )

            if length_of_pin > 0.01:
                self.pin_request(request, length_of_pin)
                del self.requests[request.request_id]
                return

        if self._infercept_should_pin(request):
            length_of_pin = self._infercept_pin_ttl_s(request)
            if length_of_pin > 0.01:
                self.pin_request(request, length_of_pin)
                del self.requests[request.request_id]
                return

        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        if self.policy == SchedulingPolicy.MARS:
            return len(self.waiting) + len(self.running) + len(
                self._mars_overflow)
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=(len(self.waiting) + len(self._mars_overflow)
                              if self.policy == SchedulingPolicy.MARS else
                              len(self.waiting)),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted
                                   for req in self.running),
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        try:
            self.interception_recorder.print_history()
        except Exception as exc:
            logger.warning("Failed to flush scheduler history: %s",
                           exc)
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less than one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in (kv_connector_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (kv_connector_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])
