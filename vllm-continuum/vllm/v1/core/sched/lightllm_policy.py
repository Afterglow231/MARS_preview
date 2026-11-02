# SPDX-License-Identifier: Apache-2.0
"""Pure scheduling-policy port of LightLLM's chunked-prefill admission.

This module ports the request token estimate, FCFS admission checks, and
output-length EMA from ModelTC/LightLLM commit ``5d59e490``.  The relevant
upstream sources are:

* ``lightllm/server/core/objs/req.py`` (``ChunkedPrefillReq``)
* ``lightllm/server/router/req_queue/chunked_prefill/impl.py``
* ``lightllm/server/router/stats.py``
* ``lightllm/server/router/model_infer/mode_backend/base_backend.py``
* ``lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py``
* ``lightllm/server/router/model_infer/mode_backend/chunked_prefill/control_state.py``

Only the policy mathematics is adapted here.  The code deliberately depends
on the Python standard library only; it does not copy or emulate LightLLM's
execution engine, memory manager, or kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


ADMITTED = "admitted"
TOKEN_CAPACITY = "token_capacity"
REQUEST_LIMIT = "request_limit"
PREFILL_BUDGET = "prefill_budget"
ADDED_OUTPUT_LEN = 16
INITIAL_EMA_OUTPUT_LEN = 2048
INITIAL_EMA_ALPHA = 0.5
EMA_ALPHA_DECAY = 0.8
MIN_EMA_ALPHA = 0.04
LIGHTLLM_UPSTREAM_COMMIT = "5d59e4907dc9b5338426b56723bdee42d8af9309"


@dataclass(frozen=True)
class LightLLMAdmissionConfig:
    """Parameters used by LightLLM's normal chunked-prefill queue."""

    router_token_ratio: float = 0.85
    router_max_wait_tokens: int = 1
    chunk_size: int = 8192
    batch_max_tokens: int = 16384

    def __post_init__(self) -> None:
        if not 0.0 <= self.router_token_ratio <= 1.0:
            raise ValueError("router_token_ratio must be in [0, 1]")
        if self.router_max_wait_tokens < 0:
            raise ValueError("router_max_wait_tokens must be non-negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.batch_max_tokens <= 0:
            raise ValueError("batch_max_tokens must be positive")
        if self.chunk_size > self.batch_max_tokens:
            raise ValueError(
                "chunk_size must not exceed batch_max_tokens, matching "
                "LightLLM's startup invariant")

    @property
    def prefill_admission_budget(self) -> int:
        """Router budget; upstream doubles ``batch_max_tokens`` once."""

        return self.batch_max_tokens * 2


@dataclass(frozen=True)
class LightLLMRequestState:
    """The request fields consumed by the ported admission mathematics.

    ``kv_len`` corresponds to LightLLM's ``shm_cur_kv_len`` and
    ``output_len`` to ``shm_cur_output_len``.  ``request_id`` is diagnostic
    only and has no effect on admission.
    """

    input_len: int
    output_len: int
    kv_len: int
    max_new_tokens: int
    ignore_eos: bool = False
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("input_len", "output_len", "kv_len",
                     "max_new_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def token_tuple(
        self,
        *,
        is_busy: bool,
        ema_output_len: int,
        config: LightLLMAdmissionConfig,
    ) -> tuple[int, int]:
        """Return upstream's ``(a_len, b_len)`` lifetime estimate."""

        if self.ignore_eos or is_busy:
            estimated_max_output = self.max_new_tokens
        else:
            observed_growth = int(1.1 * self.output_len)
            estimated_max_output = min(
                self.max_new_tokens,
                max(observed_growth, ema_output_len),
            )

        a_len = max(
            self.input_len + self.output_len + 1,
            self.kv_len + 1,
        )
        remaining_prefill = (
            self.input_len + self.output_len - self.kv_len)
        prefill_chunks = (
            remaining_prefill + config.chunk_size - 1) // config.chunk_size
        b_len = (
            prefill_chunks * (config.router_max_wait_tokens + 1)
            + estimated_max_output
            - self.output_len
            - 1
        )
        b_len = max(0, b_len) + ADDED_OUTPUT_LEN
        return a_len, b_len

    def decode_need_tokens(self, chunk_size: int) -> int:
        """Tokens the active request can consume in the next router step."""

        remaining = self.input_len + self.output_len - self.kv_len
        return min(max(0, remaining), chunk_size)

    def first_router_need_tokens(self, chunk_size: int) -> int:
        """First-step tokens charged when this waiting request is admitted."""

        return min(self.input_len + self.output_len, chunk_size)

    def is_decode_step(self) -> bool:
        """Match LightLLM model-side prefill/decode classification.

        Chunked-prefill mode keeps the final token uncomputed so the forward
        pass can produce logits.  Consequently ``kv_len == input_len - 1``
        is already a decode step for a new request, including a full prefix
        cache hit.
        """

        return self.kv_len + 1 == self.input_len + self.output_len


@dataclass(frozen=True)
class LightLLMAdmissionDecision:
    """Result for one FCFS head candidate."""

    admitted: bool
    reason: str
    peak_tokens: int
    prefill_tokens: int
    is_busy: bool


def has_prefill_work(states: Iterable[LightLLMRequestState]) -> bool:
    """Whether LightLLM would place any request in its prefill class."""

    return any(not state.is_decode_step() for state in states)


class LightLLMAdmissionController:
    """Stateful EMA plus pure, one-candidate FCFS admission decisions.

    Callers must evaluate only the current queue head and stop scanning after
    a rejection.  That preserves the upstream queue's head-of-line behavior.
    """

    def __init__(
        self,
        config: LightLLMAdmissionConfig,
        max_total_tokens: int,
        running_max_req_size: int,
    ) -> None:
        if max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if running_max_req_size <= 0:
            raise ValueError("running_max_req_size must be positive")
        self.config = config
        self.max_total_tokens = max_total_tokens
        self.running_max_req_size = running_max_req_size
        self.ema_output_len = INITIAL_EMA_OUTPUT_LEN
        self.ema_alpha = INITIAL_EMA_ALPHA

    def is_busy(self, active_kv_tokens: int) -> bool:
        """Match LightLLM's inclusive busy-ratio threshold."""

        if active_kv_tokens < 0:
            raise ValueError("active_kv_tokens must be non-negative")
        return (
            active_kv_tokens / self.max_total_tokens
            >= self.config.router_token_ratio
        )

    def estimate_peak_tokens(
        self,
        states: Sequence[LightLLMRequestState],
        *,
        is_busy: bool,
    ) -> int:
        """Compute ``max(b * i + cumulative_a)`` after descending-b sort."""

        token_tuples = [
            state.token_tuple(
                is_busy=is_busy,
                ema_output_len=self.ema_output_len,
                config=self.config,
            )
            for state in states
        ]
        token_tuples.sort(key=lambda item: item[1], reverse=True)

        peak_tokens = 0
        cumulative_a = 0
        for index, (a_len, b_len) in enumerate(token_tuples, start=1):
            cumulative_a += a_len
            peak_tokens = max(
                peak_tokens,
                b_len * index + cumulative_a,
            )
        return peak_tokens

    def evaluate(
        self,
        active_states: Iterable[LightLLMRequestState],
        candidate: LightLLMRequestState,
        active_kv_tokens: int,
    ) -> LightLLMAdmissionDecision:
        """Evaluate one FCFS queue-head candidate without mutating state."""

        active = tuple(active_states)
        is_busy = self.is_busy(active_kv_tokens)
        proposed = active + (candidate, )
        peak_tokens = self.estimate_peak_tokens(
            proposed,
            is_busy=is_busy,
        )
        prefill_tokens = sum(
            state.decode_need_tokens(self.config.chunk_size)
            for state in active
        )
        prefill_tokens += candidate.first_router_need_tokens(
            self.config.chunk_size)

        # Keep both the strict capacity boundary and the check ordering from
        # the upstream conjunction.  The ordering only selects a useful,
        # deterministic diagnostic when more than one constraint fails.
        if peak_tokens >= self.max_total_tokens:
            admitted = False
            reason = TOKEN_CAPACITY
        elif len(proposed) > self.running_max_req_size:
            admitted = False
            reason = REQUEST_LIMIT
        elif prefill_tokens > self.config.prefill_admission_budget:
            admitted = False
            reason = PREFILL_BUDGET
        else:
            admitted = True
            reason = ADMITTED

        return LightLLMAdmissionDecision(
            admitted=admitted,
            reason=reason,
            peak_tokens=peak_tokens,
            prefill_tokens=prefill_tokens,
            is_busy=is_busy,
        )

    def update_output_ema(self, output_tokens: int) -> int:
        """Update and return LightLLM's completed-output-length EMA."""

        if output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        observed = max(output_tokens, 64)
        self.ema_output_len = int(
            self.ema_output_len * (1 - self.ema_alpha)
            + observed * self.ema_alpha
        )
        self.ema_output_len = max(64, self.ema_output_len)
        self.ema_alpha = max(
            MIN_EMA_ALPHA,
            self.ema_alpha * EMA_ALPHA_DECAY,
        )
        return self.ema_output_len


__all__ = [
    "ADDED_OUTPUT_LEN",
    "ADMITTED",
    "EMA_ALPHA_DECAY",
    "INITIAL_EMA_ALPHA",
    "INITIAL_EMA_OUTPUT_LEN",
    "LIGHTLLM_UPSTREAM_COMMIT",
    "MIN_EMA_ALPHA",
    "PREFILL_BUDGET",
    "REQUEST_LIMIT",
    "TOKEN_CAPACITY",
    "LightLLMAdmissionConfig",
    "LightLLMAdmissionController",
    "LightLLMAdmissionDecision",
    "LightLLMRequestState",
    "has_prefill_work",
]
