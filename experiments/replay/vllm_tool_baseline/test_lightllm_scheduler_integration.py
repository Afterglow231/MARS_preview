from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = REPO_ROOT / "vllm-continuum"
sys.path.insert(0, str(VLLM_ROOT))

from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheBlocks  # noqa: E402
from vllm.v1.core.kv_cache_utils import KVCacheBlock  # noqa: E402
from vllm.v1.core.sched.lightllm_policy import (  # noqa: E402
    LightLLMAdmissionConfig,
    LightLLMAdmissionController,
)
from vllm.v1.core.sched.output import CachedRequestData  # noqa: E402
from vllm.v1.core.sched.request_queue import (  # noqa: E402
    FCFSRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched import scheduler as scheduler_module  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus  # noqa: E402


def _request(
    request_id: str,
    *,
    prompt_tokens: int,
    max_tokens: int = 8,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


class _FakeKVCacheManager:

    def __init__(self, *args, **kwargs) -> None:
        self.block_pool = SimpleNamespace(
            num_gpu_blocks=1001,
            get_num_free_blocks=lambda: 1000,
            touch=Mock(),
            free_blocks=Mock(),
        )
        self.allocate_calls: list[tuple[Request, int]] = []
        self._blocks = KVCacheBlocks(([KVCacheBlock(1)], ))
        self.usage = 0.0
        self.make_prefix_cache_stats = Mock(return_value=Mock())

    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[KVCacheBlocks, int]:
        return KVCacheBlocks(([], )), 0

    def create_empty_block_list(self) -> KVCacheBlocks:
        return KVCacheBlocks(([], ))

    def allocate_slots(self, request: Request, num_tokens: int, *args,
                       **kwargs) -> KVCacheBlocks:
        self.allocate_calls.append((request, num_tokens))
        return self._blocks

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return self._blocks

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running: int,
    ) -> list[int]:
        return [0]

    def take_events(self) -> list[object]:
        return []


def _scheduler(*, token_budget: int = 16, chunk_size: int = 4) -> Scheduler:
    """Build only the state used by the synchronous text schedule path."""

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.policy = SchedulingPolicy.LIGHTLLM
    scheduler.running = []
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.requests = {}
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_scheduled_tokens = token_budget
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_model_len = 4096
    scheduler.block_size = 16
    scheduler.num_lookahead_tokens = 0
    scheduler.use_pp = False
    scheduler.log_stats = False
    scheduler.lora_config = None
    scheduler.connector = None
    scheduler.is_encoder_decoder = False
    scheduler.pinned_requests = []
    scheduler.finished_req_ids = set()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=(object(), ))
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0,
        chunked_prefill_enabled=True,
    )
    scheduler.kv_cache_manager = _FakeKVCacheManager()
    scheduler.encoder_cache_manager = Mock()
    scheduler.encoder_cache_manager.get_freed_mm_hashes.return_value = []
    scheduler.interception_recorder = Mock()
    scheduler.kv_event_publisher = Mock()
    scheduler._continuum_evicted_at = {}
    scheduler._continuum_queue_delay_samples = []
    scheduler._lightllm_chunk_size = chunk_size
    scheduler._lightllm_controller = LightLLMAdmissionController(
        LightLLMAdmissionConfig(
            chunk_size=chunk_size,
            batch_max_tokens=token_budget,
        ),
        max_total_tokens=16_000,
        running_max_req_size=scheduler.max_num_running_reqs,
    )
    scheduler._lightllm_deferred = {}

    # These helpers are orthogonal to the scheduling decisions under test.
    scheduler.unpin_requests_regular = Mock()
    scheduler._make_cached_request_data = Mock(
        return_value=CachedRequestData.make_empty())
    scheduler.get_grammar_bitmask = Mock(return_value=({}, None))
    scheduler._make_forward_step_profile_meta = Mock(return_value=None)
    scheduler._update_after_schedule = Mock()
    return scheduler


def _vllm_config_for_init() -> tuple[object, object]:
    scheduler_config = SimpleNamespace(
        max_num_seqs=16,
        max_num_batched_tokens=16,
        max_model_len=4096,
        policy="lightllm",
        autellix_tail_fcfs_after_finished=0,
        runner_type="generate",
        is_multimodal_model=False,
        async_scheduling=False,
        lightllm_chunked_prefill_size=4,
        lightllm_router_token_ratio=0.85,
        lightllm_router_max_wait_tokens=1,
        chunked_prefill_enabled=True,
        long_prefill_token_threshold=0,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        cache_config=SimpleNamespace(
            num_gpu_blocks=1001,
            block_size=16,
            enable_prefix_caching=True,
        ),
        lora_config=None,
        kv_events_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            tokenizer="unused-by-test",
            tokenizer_mode="auto",
            trust_remote_code=False,
            tokenizer_revision=None,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=None,  # Runtime type is irrelevant to this control-plane test.
        use_mla=False,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=(KVCacheGroupSpec([], attention_spec), ),
    )
    return vllm_config, kv_cache_config


def _free_request_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.policy = SchedulingPolicy.LIGHTLLM
    scheduler._lightllm_controller = Mock()
    scheduler._lightllm_deferred = {}
    scheduler.tool_call_estimator = Mock()
    scheduler.interception_recorder = Mock()
    scheduler._infercept_prepare_finished_request = Mock()
    scheduler._connector_finished = Mock(return_value=(False, None))
    scheduler.encoder_cache_manager = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_blocks = Mock()
    return scheduler


class LightLLMSchedulerIntegrationTest(unittest.TestCase):

    def test_policy_maps_to_lightllm_and_uses_fcfs_queue(self) -> None:
        self.assertIs(
            SchedulingPolicy("lightllm"),
            SchedulingPolicy.LIGHTLLM,
        )
        queue = create_request_queue(SchedulingPolicy.LIGHTLLM)
        self.assertIsInstance(queue, FCFSRequestQueue)

        first = _request("first", prompt_tokens=1)
        second = _request("second", prompt_tokens=1)
        queue.add_request(first)
        queue.add_request(second)
        self.assertIs(queue.peek_request(), first)
        self.assertIs(queue.pop_request(), first)
        self.assertIs(queue.pop_request(), second)

        vllm_config, kv_cache_config = _vllm_config_for_init()
        with (
            patch.object(
                scheduler_module,
                "InterceptionRecorder",
                return_value=Mock(),
            ),
            patch.object(
                scheduler_module,
                "ToolCallEstimator",
                return_value=Mock(),
            ),
            patch.object(
                scheduler_module,
                "compute_encoder_budget",
                return_value=(0, 0),
            ),
            patch.object(
                scheduler_module,
                "EncoderCacheManager",
                return_value=Mock(),
            ),
            patch.object(
                scheduler_module,
                "KVCacheManager",
                _FakeKVCacheManager,
            ),
            patch.object(
                scheduler_module.EventPublisherFactory,
                "create",
                return_value=Mock(),
            ),
        ):
            scheduler = Scheduler(vllm_config, kv_cache_config, Mock())

        self.assertIs(scheduler.policy, SchedulingPolicy.LIGHTLLM)
        self.assertIsInstance(scheduler.waiting, FCFSRequestQueue)

    def test_unsupported_parallel_policy_configurations_are_rejected(
            self) -> None:
        with (
            patch.object(
                scheduler_module,
                "InterceptionRecorder",
                return_value=Mock(),
            ),
            patch.object(
                scheduler_module,
                "ToolCallEstimator",
                return_value=Mock(),
            ),
            patch.object(
                scheduler_module,
                "compute_encoder_budget",
                return_value=(0, 0),
            ),
            patch.object(
                scheduler_module,
                "EncoderCacheManager",
                return_value=Mock(),
            ),
            patch.object(
                scheduler_module,
                "KVCacheManager",
                _FakeKVCacheManager,
            ),
            patch.object(
                scheduler_module.EventPublisherFactory,
                "create",
                return_value=Mock(),
            ),
        ):
            for attribute, message in (
                ("pipeline_parallel_size", "pipeline parallelism"),
                ("data_parallel_size", "data parallelism"),
            ):
                with self.subTest(attribute=attribute):
                    vllm_config, kv_cache_config = _vllm_config_for_init()
                    setattr(vllm_config.parallel_config, attribute, 2)
                    with self.assertRaisesRegex(ValueError, message):
                        Scheduler(vllm_config, kv_cache_config, Mock())

    def test_prefill_first_skips_decode_and_caps_running_chunk(self) -> None:
        scheduler = _scheduler(token_budget=8, chunk_size=4)
        decode = _request("decode", prompt_tokens=2)
        decode.append_output_token_ids(99)
        decode.num_computed_tokens = decode.num_prompt_tokens
        decode.status = RequestStatus.RUNNING
        prefill = _request("prefill", prompt_tokens=8)
        prefill.status = RequestStatus.RUNNING
        scheduler.running = [decode, prefill]
        scheduler.requests = {
            decode.request_id: decode,
            prefill.request_id: prefill,
        }

        self.assertTrue(scheduler._lightllm_prefill_step())
        output = scheduler.schedule()

        self.assertEqual(output.num_scheduled_tokens, {"prefill": 4})
        self.assertEqual(
            [(request.request_id, tokens)
             for request, tokens in scheduler.kv_cache_manager.allocate_calls],
            [("prefill", 4)],
        )

    def test_prefill_packer_does_not_shrink_chunk_to_leftover_budget(
            self) -> None:
        scheduler = _scheduler(token_budget=8, chunk_size=4)
        requests = [
            _request("three", prompt_tokens=3),
            _request("four-a", prompt_tokens=4),
            _request("four-b", prompt_tokens=4),
        ]
        for request in requests:
            request.status = RequestStatus.RUNNING
        scheduler.running = requests
        scheduler.requests = {
            request.request_id: request for request in requests
        }

        output = scheduler.schedule()

        self.assertEqual(
            output.num_scheduled_tokens,
            {"three": 3, "four-a": 4},
        )

    def test_waiting_prefill_is_chunked_after_admission(self) -> None:
        scheduler = _scheduler(token_budget=16, chunk_size=4)
        request = _request("waiting", prompt_tokens=12)
        scheduler.waiting.add_request(request)
        scheduler.requests[request.request_id] = request

        output = scheduler.schedule()

        self.assertEqual(output.num_scheduled_tokens, {"waiting": 4})
        self.assertEqual(
            [(queued.request_id, tokens)
             for queued, tokens in scheduler.kv_cache_manager.allocate_calls],
            [("waiting", 4)],
        )

    def test_prefix_hits_do_not_discount_later_same_batch_admission(
            self) -> None:
        scheduler = _scheduler(token_budget=16, chunk_size=4)
        # Prefix lookup is deliberately after LightLLM's router admission.
        # With a 4-token router chunk and doubled batch budget of 32, the
        # upstream FCFS scan admits exactly 8 requests.  A 16-token aligned
        # vLLM cache hit on each 17-token prompt must reduce execution work
        # without retroactively reducing that
        # provisional router charge for later candidates in the same scan.
        scheduler.kv_cache_manager.get_computed_blocks = Mock(
            return_value=(KVCacheBlocks(([], )), 16))
        for index in range(16):
            request = _request(f"waiting-{index}", prompt_tokens=17)
            scheduler.waiting.add_request(request)
            scheduler.requests[request.request_id] = request

        output = scheduler.schedule()

        self.assertEqual(len(output.num_scheduled_tokens), 8)
        self.assertEqual(set(output.num_scheduled_tokens.values()), {1})
        self.assertEqual(len(scheduler.running), 8)
        self.assertEqual(len(scheduler.waiting), 8)
        self.assertEqual(
            scheduler.kv_cache_manager.get_computed_blocks.call_count,
            8,
        )

    def test_full_prefix_hit_runs_in_decode_step_with_existing_decode(
            self) -> None:
        scheduler = _scheduler(token_budget=8, chunk_size=4)
        decode = _request("decode", prompt_tokens=2)
        decode.append_output_token_ids(99)
        decode.num_computed_tokens = decode.num_prompt_tokens
        decode.status = RequestStatus.RUNNING
        full_hit = _request("full-block-hit", prompt_tokens=17)
        scheduler.running = [decode]
        scheduler.waiting.add_request(full_hit)
        scheduler.requests = {
            decode.request_id: decode,
            full_hit.request_id: full_hit,
        }
        scheduler.kv_cache_manager.get_computed_blocks = Mock(
            return_value=(KVCacheBlocks(([], )), 16))

        output = scheduler.schedule()

        self.assertEqual(
            output.num_scheduled_tokens,
            {"decode": 1, "full-block-hit": 1},
        )

    def test_prefill_step_scans_past_admitted_decode_and_restores_fcfs(
            self) -> None:
        scheduler = _scheduler(token_budget=8, chunk_size=4)
        decode = _request("decode", prompt_tokens=2)
        decode.append_output_token_ids(99)
        decode.num_computed_tokens = decode.num_prompt_tokens
        decode.status = RequestStatus.RUNNING
        full_hit = _request("full-block-hit", prompt_tokens=17)
        prefill = _request("prefill", prompt_tokens=8)
        scheduler.running = [decode]
        scheduler.waiting.add_request(full_hit)
        scheduler.waiting.add_request(prefill)
        scheduler.requests = {
            request.request_id: request
            for request in (decode, full_hit, prefill)
        }

        def prefix_lookup(request: Request) -> tuple[KVCacheBlocks, int]:
            hit_tokens = 16 if request is full_hit else 0
            return KVCacheBlocks(([], )), hit_tokens

        scheduler.kv_cache_manager.get_computed_blocks = Mock(
            side_effect=prefix_lookup)

        output = scheduler.schedule()

        self.assertEqual(output.num_scheduled_tokens, {"prefill": 4})
        self.assertEqual(
            [request.request_id for request in scheduler.waiting],
            ["full-block-hit"],
        )

    def test_deferred_prefix_hit_is_held_and_reused_across_steps(
            self) -> None:
        scheduler = _scheduler(token_budget=8, chunk_size=4)
        active_prefill = _request("active-prefill", prompt_tokens=4)
        active_prefill.status = RequestStatus.RUNNING
        full_hit = _request("full-block-hit", prompt_tokens=17)
        scheduler.running = [active_prefill]
        scheduler.waiting.add_request(full_hit)
        scheduler.requests = {
            active_prefill.request_id: active_prefill,
            full_hit.request_id: full_hit,
        }
        prefix_block = KVCacheBlock(7)
        scheduler.kv_cache_manager.get_computed_blocks = Mock(
            return_value=(KVCacheBlocks(([prefix_block], )), 16))

        first_output = scheduler.schedule()

        self.assertEqual(
            first_output.num_scheduled_tokens,
            {"active-prefill": 4},
        )
        self.assertEqual(scheduler.get_request_counts(), (2, 0))
        scheduler.log_stats = True
        stats = scheduler.make_stats()
        assert stats is not None
        self.assertEqual(stats.num_running_reqs, 2)
        self.assertEqual(stats.num_waiting_reqs, 0)
        scheduler.log_stats = False
        self.assertIn(full_hit.request_id, scheduler._lightllm_deferred)
        scheduler.kv_cache_manager.block_pool.touch.assert_called_once()
        scheduler.kv_cache_manager.block_pool.free_blocks.assert_not_called()

        # Apply the relevant model-output state transition: the prompt was
        # computed and its first output token was produced.
        active_prefill.num_computed_tokens = 4
        active_prefill.append_output_token_ids(99)
        second_output = scheduler.schedule()

        self.assertEqual(
            second_output.num_scheduled_tokens,
            {"active-prefill": 1, "full-block-hit": 1},
        )
        self.assertEqual(
            scheduler.kv_cache_manager.get_computed_blocks.call_count,
            1,
        )
        scheduler.kv_cache_manager.block_pool.free_blocks.assert_called_once()
        self.assertNotIn(full_hit.request_id, scheduler._lightllm_deferred)

    def test_rejected_fcfs_head_blocks_tail_before_kv_allocation(self) -> None:
        scheduler = _scheduler(token_budget=16, chunk_size=4)
        # This controller makes the head inadmissible by token capacity while
        # the tiny tail would be admissible if the scheduler looked past it.
        scheduler._lightllm_controller = LightLLMAdmissionController(
            LightLLMAdmissionConfig(chunk_size=4, batch_max_tokens=16),
            max_total_tokens=100,
            running_max_req_size=16,
        )
        head = _request("head", prompt_tokens=1000, max_tokens=1)
        tail = _request("tail", prompt_tokens=1, max_tokens=1)
        scheduler.waiting.add_request(head)
        scheduler.waiting.add_request(tail)
        scheduler.requests = {"head": head, "tail": tail}

        self.assertFalse(
            scheduler._lightllm_evaluate_candidate(head).admitted)
        self.assertTrue(
            scheduler._lightllm_controller.evaluate(
                [],
                scheduler._lightllm_request_state(tail),
                active_kv_tokens=0,
            ).admitted)
        output = scheduler.schedule()

        self.assertEqual(output.num_scheduled_tokens, {})
        self.assertEqual(
            [request.request_id for request in scheduler.waiting],
            ["head", "tail"],
        )
        self.assertEqual(scheduler.kv_cache_manager.allocate_calls, [])

    def test_normal_completion_updates_output_ema(self) -> None:
        scheduler = _free_request_scheduler()
        request = _request("complete", prompt_tokens=1)
        request.append_output_token_ids([1, 2, 3])
        request.status = RequestStatus.FINISHED_STOPPED

        scheduler._free_request(request)

        scheduler._lightllm_controller.update_output_ema.assert_called_once_with(
            3)

    def test_abort_does_not_update_output_ema(self) -> None:
        scheduler = _free_request_scheduler()
        request = _request("aborted", prompt_tokens=1)
        request.append_output_token_ids([1, 2])
        request.status = RequestStatus.FINISHED_ABORTED

        scheduler._free_request(request)

        scheduler._lightllm_controller.update_output_ema.assert_not_called()


if __name__ == "__main__":
    unittest.main()
