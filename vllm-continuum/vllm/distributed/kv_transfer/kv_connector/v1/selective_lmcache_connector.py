# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selective/on-demand LMCache connector for InferCept swap.

The stock LMCacheConnectorV1 can save KV during normal forward steps.  For the
InferCept swap baseline we want a narrower behavior: keep LMCache loads
enabled, but save KV only after the scheduler has explicitly decided that a
finished tool-call request should be swapped to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    ReqMeta,
    SaveSpec,
)

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class _LazySaveRequest:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    request_configs: Optional[dict[str, Any]] = None


class SelectiveLMCacheConnectorV1(KVConnectorBase_V1):
    """LMCache V1 connector with request-finished lazy stores.

    Scheduler-side:
      * normal scheduled requests may load from LMCache;
      * normal scheduled requests do not save into LMCache;
      * a finished request with ``lmcache.lazy_save`` or
        ``request._infercept_kv_action == "swap"`` is kept alive and appended as
        save-only metadata until a worker forward step stores it.

    Worker-side:
      * delegates all load/store mechanics to LMCacheConnectorV1Impl;
      * after ``wait_for_save()`` returns, any save-only metadata request is
        reported through ``finished_sending`` so the scheduler can free blocks.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._lmcache_engine = LMCacheConnectorV1Impl(vllm_config, role, self)
        self._block_size = int(vllm_config.cache_config.block_size)
        self._lazy_saves: dict[str, _LazySaveRequest] = {}
        self._finished_lazy_saves: set[str] = set()

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._lmcache_engine.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)
        # vLLM runs a no-forward connector step to clear finished requests.
        # The stock adapter skips saving there because wait_for_save=False in
        # that path. For lazy swap, this is the ideal moment to push KV to CPU:
        # it happens immediately after the tool-call request finishes, while
        # the external tool is still running.
        if getattr(forward_context, "attn_metadata", None) is None:
            lazy_req_ids = self._lazy_req_ids_in_metadata()
            if lazy_req_ids:
                use_layerwise = bool(
                    getattr(self._lmcache_engine, "use_layerwise", False))
                if use_layerwise:
                    logger.warning(
                        "Lazy LMCache no-forward save cannot run with "
                        "layerwise=True; request ids will wait for a forward "
                        "step: %s", sorted(lazy_req_ids))
                else:
                    self._lmcache_engine.wait_for_save()
                    self._finished_lazy_saves.update(lazy_req_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        self._lmcache_engine.save_kv_layer(layer_name, kv_layer, attn_metadata,
                                           **kwargs)

    def wait_for_save(self):
        lazy_req_ids = self._lazy_req_ids_in_metadata()
        self._lmcache_engine.wait_for_save()
        self._finished_lazy_saves.update(lazy_req_ids)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        finished_sending, finished_recving = self._lmcache_engine.get_finished(
            finished_req_ids)

        sending = set(finished_sending or ())
        if self._finished_lazy_saves:
            sending.update(self._finished_lazy_saves)
            self._finished_lazy_saves.clear()

        return (sending or None), finished_recving

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens), False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        self._lmcache_engine.update_state_after_alloc(request,
                                                      num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = self._lmcache_engine.build_connector_meta(scheduler_output)
        assert isinstance(meta, LMCacheConnectorMetadata)

        # Keep LMCache loads, but make normal request scheduling save-free.
        for req_meta in meta.requests:
            if req_meta.save_spec is not None:
                req_meta.save_spec.can_save = False

        for req_id, lazy_req in list(self._lazy_saves.items()):
            lazy_meta = self._make_lazy_save_meta(lazy_req)
            if lazy_meta is None:
                logger.warning("Skipping empty lazy LMCache save for %s",
                               req_id)
                self._lazy_saves.pop(req_id, None)
                continue
            meta.add_request(lazy_meta)

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        _delay, kv_transfer_params = self._lmcache_engine.request_finished(
            request, block_ids)

        if not self._should_lazy_save(request):
            self._lazy_saves.pop(request.request_id, None)
            return False, kv_transfer_params

        token_ids = list(request.all_token_ids)
        if not token_ids or not block_ids:
            logger.warning(
                "Cannot lazy-save request %s to LMCache: tokens=%d blocks=%d",
                request.request_id,
                len(token_ids),
                len(block_ids),
            )
            return False, kv_transfer_params

        self._lazy_saves[request.request_id] = _LazySaveRequest(
            req_id=request.request_id,
            token_ids=token_ids,
            block_ids=list(block_ids),
            request_configs=self._extract_request_configs(request),
        )
        logger.info("Queued lazy LMCache save for request %s tokens=%d "
                    "blocks=%d",
                    request.request_id, len(token_ids), len(block_ids))
        return True, kv_transfer_params

    def update_connector_output(self, connector_output):
        update = getattr(self._lmcache_engine, "update_connector_output", None)
        if update is not None:
            update(connector_output)
        for req_id in (getattr(connector_output, "finished_sending", None)
                       or ()):
            self._lazy_saves.pop(req_id, None)

    def shutdown(self):
        shutdown = getattr(self._lmcache_engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    @classmethod
    def get_required_kvcache_layout(
            cls, vllm_config: "VllmConfig") -> Optional[str]:
        return None

    def _lazy_req_ids_in_metadata(self) -> set[str]:
        try:
            metadata = self._get_connector_metadata()
        except Exception:
            return set()
        if not isinstance(metadata, LMCacheConnectorMetadata):
            return set()
        req_ids: set[str] = set()
        for req_meta in metadata.requests:
            save_spec = getattr(req_meta, "save_spec", None)
            if save_spec is not None and bool(
                    getattr(save_spec, "can_save", False)):
                req_ids.add(str(req_meta.req_id))
        return req_ids

    @staticmethod
    def _should_lazy_save(request: "Request") -> bool:
        if getattr(request, "_infercept_kv_action", None) == "swap":
            return True
        params = getattr(request, "kv_transfer_params", None) or {}
        return bool(params.get("lmcache.lazy_save"))

    @staticmethod
    def _extract_request_configs(request: "Request") -> Optional[dict[str, Any]]:
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None)
        if not isinstance(extra_args, dict):
            return None
        params = extra_args.get("kv_transfer_params")
        if not isinstance(params, dict):
            return None
        request_configs = {
            str(k): v
            for k, v in params.items()
            if str(k).startswith("lmcache.")
            and str(k) not in {"lmcache.lazy_save", "lmcache.skip_save"}
        }
        return request_configs or None

    def _make_lazy_save_meta(
            self, lazy_req: _LazySaveRequest) -> Optional[ReqMeta]:
        token_ids = list(lazy_req.token_ids)
        block_ids = list(lazy_req.block_ids)
        max_tokens = len(block_ids) * self._block_size
        if max_tokens <= 0:
            return None
        if len(token_ids) > max_tokens:
            logger.warning(
                "Truncating lazy LMCache save for %s from %d to %d tokens "
                "because only %d blocks are allocated",
                lazy_req.req_id,
                len(token_ids),
                max_tokens,
                len(block_ids),
            )
            token_ids = token_ids[:max_tokens]
        if not token_ids:
            return None

        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size)) +
            block_ids_tensor.reshape((len(block_ids), 1)) * self._block_size)
        slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        assert slot_mapping.dtype == torch.long

        return ReqMeta(
            req_id=lazy_req.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=True,
            save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
            load_spec=None,
            disagg_spec=None,
            request_configs=lazy_req.request_configs,
        )
