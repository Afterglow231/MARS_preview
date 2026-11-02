from __future__ import annotations

from request_bootstrap import build_user_request


def test_terminalbench_prompt_keeps_hint_and_container_workflow() -> None:
    prompt = build_user_request(
        {
            "benchmark": "terminalbench",
            "payload": {
                "instruction": "Fix the service.",
                "hint": "Inspect the pinned dependency first.",
            },
        }
    )

    assert "Fix the service." in prompt
    assert "Inspect the pinned dependency first." in prompt
    assert 'python tb_task.py exec "<cmd>"' in prompt
    assert "python tb_task.py test" in prompt
    assert "standard OpenHands tools" not in prompt
