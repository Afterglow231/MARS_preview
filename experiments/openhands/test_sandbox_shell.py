from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SHELL_SCRIPT = Path(__file__).resolve().with_name("sandbox_shell.sh")


def _run_guarded_shell(base_dir: Path, command: str) -> subprocess.CompletedProcess[str]:
    workspace = base_dir / "workspace"
    runtime = workspace / ".mars_runtime"
    home_dir = runtime / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "MARS_WORKSPACE_DIR": str(workspace),
            "MARS_RUNTIME_DIR": str(runtime),
            "MARS_HOST_HOME": "",
            "MARS_HOST_CONDA_ROOT": "",
            "MARS_HOST_VENVS_ROOT": "",
            "MARS_HOST_MODELS_ROOT": "",
        }
    )
    return subprocess.run(
        [str(SHELL_SCRIPT), "-lc", command],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class SandboxShellTest(unittest.TestCase):
    def test_blocks_broad_process_control_command(self) -> None:
        command = "pkill -f mars_guard_test_process_name_that_should_not_exist"
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_guarded_shell(Path(tmp), command)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Command blocked by MARS runtime guard.", proc.stderr)
        self.assertIn(command, proc.stderr)

    def test_allows_safe_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_guarded_shell(Path(tmp), "printf ok")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "ok")


if __name__ == "__main__":
    unittest.main()
