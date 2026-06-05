from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SHELL_SCRIPT = Path(__file__).resolve().with_name("sandbox_shell.sh")


def _run_guarded_shell(
    base_dir: Path,
    command: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
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
            "MARS_LOCAL_INPUTS_DIR": "",
        }
    )
    if extra_env:
        env.update(extra_env)
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

    def test_exposes_local_inputs_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local_inputs = base / "out" / "_localized_inputs"
            local_inputs.mkdir(parents=True)
            asset = local_inputs / "asset.txt"
            asset.write_text("input\n", encoding="utf-8")

            read_proc = _run_guarded_shell(
                base,
                'cat "$MARS_LOCAL_INPUTS_DIR/asset.txt"',
                extra_env={"MARS_LOCAL_INPUTS_DIR": str(local_inputs)},
            )
            self.assertEqual(read_proc.returncode, 0, read_proc.stderr)
            self.assertEqual(read_proc.stdout, "input\n")

            write_proc = _run_guarded_shell(
                base,
                'printf changed > "$MARS_LOCAL_INPUTS_DIR/asset.txt"',
                extra_env={"MARS_LOCAL_INPUTS_DIR": str(local_inputs)},
            )
            self.assertNotEqual(write_proc.returncode, 0)
            self.assertEqual(asset.read_text(encoding="utf-8"), "input\n")


if __name__ == "__main__":
    unittest.main()
