from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = (
    REPO_ROOT
    / "vllm-continuum/vllm/v1/core/sched/lightllm_policy.py"
)
SPEC = importlib.util.spec_from_file_location("lightllm_policy", POLICY_PATH)
assert SPEC is not None and SPEC.loader is not None
lightllm_policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lightllm_policy
SPEC.loader.exec_module(lightllm_policy)

ADMITTED = lightllm_policy.ADMITTED
PREFILL_BUDGET = lightllm_policy.PREFILL_BUDGET
REQUEST_LIMIT = lightllm_policy.REQUEST_LIMIT
TOKEN_CAPACITY = lightllm_policy.TOKEN_CAPACITY
LIGHTLLM_UPSTREAM_COMMIT = lightllm_policy.LIGHTLLM_UPSTREAM_COMMIT
LightLLMAdmissionConfig = lightllm_policy.LightLLMAdmissionConfig
LightLLMAdmissionController = lightllm_policy.LightLLMAdmissionController
LightLLMRequestState = lightllm_policy.LightLLMRequestState
has_prefill_work = lightllm_policy.has_prefill_work


def _state(
    *,
    input_len: int = 1,
    output_len: int = 0,
    kv_len: int = 0,
    max_new_tokens: int = 1,
    request_id: str = "req",
) -> object:
    return LightLLMRequestState(
        input_len=input_len,
        output_len=output_len,
        kv_len=kv_len,
        max_new_tokens=max_new_tokens,
        request_id=request_id,
    )


class LightLLMPolicyTest(unittest.TestCase):

    def test_chunk_size_cannot_exceed_batch_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_max_tokens"):
            LightLLMAdmissionConfig(
                chunk_size=17,
                batch_max_tokens=16,
            )

    def test_policy_commit_matches_native_lock(self) -> None:
        lock_path = (
            REPO_ROOT / "experiments" / "baselines" / "lightllm"
            / "upstream.env")
        match = re.search(
            r'^LIGHTLLM_COMMIT="([0-9a-f]{40})"$',
            lock_path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), LIGHTLLM_UPSTREAM_COMMIT)

    def test_busy_switches_from_ema_to_full_output_bound(self) -> None:
        controller = LightLLMAdmissionController(
            LightLLMAdmissionConfig(),
            max_total_tokens=10_000,
            running_max_req_size=8,
        )
        candidate = _state(
            input_len=100,
            output_len=100,
            kv_len=200,
            max_new_tokens=4096,
        )

        not_busy = controller.evaluate([], candidate, active_kv_tokens=8499)
        busy = controller.evaluate([], candidate, active_kv_tokens=8500)

        self.assertFalse(not_busy.is_busy)
        self.assertTrue(busy.is_busy)
        self.assertEqual(not_busy.peak_tokens, 2164)
        self.assertEqual(busy.peak_tokens, 4212)
        self.assertTrue(not_busy.admitted)
        self.assertTrue(busy.admitted)

    def test_capacity_check_is_strict(self) -> None:
        candidate = _state(input_len=10, max_new_tokens=10)
        at_capacity = LightLLMAdmissionController(
            LightLLMAdmissionConfig(),
            max_total_tokens=38,
            running_max_req_size=8,
        ).evaluate([], candidate, active_kv_tokens=0)
        below_capacity = LightLLMAdmissionController(
            LightLLMAdmissionConfig(),
            max_total_tokens=39,
            running_max_req_size=8,
        ).evaluate([], candidate, active_kv_tokens=0)

        self.assertEqual(at_capacity.peak_tokens, 38)
        self.assertFalse(at_capacity.admitted)
        self.assertEqual(at_capacity.reason, TOKEN_CAPACITY)
        self.assertTrue(below_capacity.admitted)
        self.assertEqual(below_capacity.reason, ADMITTED)

    def test_fcfs_head_rejection_does_not_authorize_lookahead(self) -> None:
        controller = LightLLMAdmissionController(
            LightLLMAdmissionConfig(),
            max_total_tokens=100,
            running_max_req_size=8,
        )
        waiting = [
            _state(input_len=1000, request_id="head"),
            _state(input_len=1, request_id="tail"),
        ]

        head_decision = controller.evaluate(
            [], waiting[0], active_kv_tokens=0)

        self.assertFalse(head_decision.admitted)
        self.assertEqual(head_decision.reason, TOKEN_CAPACITY)
        # The tail demonstrates why the caller must break on the rejected
        # head, exactly as LightLLM's waiting-list loop does.
        self.assertTrue(
            controller.evaluate([], waiting[1], active_kv_tokens=0).admitted)

    def test_request_count_limit(self) -> None:
        controller = LightLLMAdmissionController(
            LightLLMAdmissionConfig(),
            max_total_tokens=100_000,
            running_max_req_size=1,
        )

        decision = controller.evaluate(
            [_state(request_id="active")],
            _state(request_id="candidate"),
            active_kv_tokens=1,
        )

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, REQUEST_LIMIT)

    def test_prefill_budget_is_doubled_and_inclusive(self) -> None:
        controller = LightLLMAdmissionController(
            LightLLMAdmissionConfig(),
            max_total_tokens=1_000_000,
            running_max_req_size=10,
        )
        full_chunk = _state(input_len=8192)

        exactly_budget = controller.evaluate(
            [full_chunk] * 3,
            full_chunk,
            active_kv_tokens=0,
        )
        over_budget = controller.evaluate(
            [full_chunk] * 4,
            full_chunk,
            active_kv_tokens=0,
        )

        self.assertEqual(
            controller.config.prefill_admission_budget,
            2 * controller.config.batch_max_tokens,
        )
        self.assertEqual(exactly_budget.prefill_tokens, 32768)
        self.assertTrue(exactly_budget.admitted)
        self.assertEqual(over_budget.prefill_tokens, 40960)
        self.assertFalse(over_budget.admitted)
        self.assertEqual(over_budget.reason, PREFILL_BUDGET)

    def test_output_ema_clamps_short_samples_and_decays_alpha(self) -> None:
        controller = LightLLMAdmissionController(
            LightLLMAdmissionConfig(),
            max_total_tokens=100_000,
            running_max_req_size=8,
        )

        self.assertEqual(controller.update_output_ema(0), 1056)
        self.assertAlmostEqual(controller.ema_alpha, 0.4)
        self.assertEqual(controller.update_output_ema(256), 736)
        self.assertAlmostEqual(controller.ema_alpha, 0.32)
        for _ in range(20):
            controller.update_output_ema(64)
        self.assertAlmostEqual(controller.ema_alpha, 0.04)
        self.assertGreaterEqual(controller.ema_output_len, 64)

    def test_prefill_classification_reserves_last_token_for_decode(self) -> None:
        self.assertFalse(has_prefill_work([
            _state(input_len=8, kv_len=7),
        ]))
        self.assertTrue(has_prefill_work([
            _state(input_len=8, output_len=3, kv_len=8),
        ]))


if __name__ == "__main__":
    unittest.main()
