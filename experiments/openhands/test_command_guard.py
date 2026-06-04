from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import command_guard


class CommandGuardTest(unittest.TestCase):
    def test_blocks_broad_process_control_commands(self) -> None:
        cases = [
            ("pkill -f python", "broad process-control command is blocked: pkill"),
            ("killall python", "broad process-control command is blocked: killall"),
            ("kill -9 -1", "broad process-control command is blocked"),
            ("kill -- -1234", "process-group kill command is blocked"),
            ("kill 0", "broad process-control command is blocked"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for command, reason_substring in cases:
                with self.subTest(command=command):
                    reason = command_guard.blocked_terminal_command_reason(command, workspace)
                    self.assertIsNotNone(reason)
                    assert reason is not None
                    self.assertIn(reason_substring, reason)

    def test_allows_scoped_or_non_process_commands(self) -> None:
        cases = [
            "kill 1234",
            "kill -TERM 1234",
            "kill -l",
            "echo kill",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for command in cases:
                with self.subTest(command=command):
                    self.assertIsNone(
                        command_guard.blocked_terminal_command_reason(command, workspace)
                    )

    def test_cli_reports_blocked_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = command_guard.main(
                    [
                        "check-terminal",
                        "--workspace-dir",
                        tmp,
                        "--command",
                        "pkill -f python",
                    ]
                )
        self.assertEqual(rc, 10)
        self.assertIn("broad process-control command is blocked: pkill", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
