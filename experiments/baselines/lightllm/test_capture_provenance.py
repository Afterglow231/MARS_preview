from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.baselines.lightllm.capture_provenance import (
    _client_source_fingerprint,
    _python_info,
)


class PythonInfoTests(unittest.TestCase):

    def test_invokes_requested_venv_path_without_resolving_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            python_link = Path(temp_dir) / "bin" / "python"
            python_link.parent.mkdir()
            python_link.symlink_to(sys.executable)
            commands: list[list[str]] = []

            def fake_run(command: list[str]) -> str:
                commands.append(command)
                if command[1:3] == ["-m", "pip"]:
                    return "openhands-sdk==1.14.0\npip==25.1"
                if command[-1] == "import sys; print(sys.prefix)":
                    return str(python_link.parent.parent)
                return "3.12.11"

            with patch(
                "experiments.baselines.lightllm.capture_provenance._run_text",
                side_effect=fake_run,
            ):
                info = _python_info(str(python_link), include_packages=True)

            requested = str(python_link.absolute())
            self.assertEqual(info["python"], requested)
            self.assertEqual(info["python_realpath"], str(python_link.resolve()))
            self.assertEqual(info["python_prefix"], str(python_link.parent.parent))
            self.assertTrue(commands)
            self.assertTrue(all(command[0] == requested for command in commands))
            self.assertIn("openhands-sdk==1.14.0", info["packages"])

    def test_package_inspection_failure_is_fatal(self) -> None:
        with patch(
            "experiments.baselines.lightllm.capture_provenance._run_text",
            side_effect=("3.12.11", "/env/openhands", None),
        ):
            with self.assertRaisesRegex(SystemExit, "Cannot inspect Python packages"):
                _python_info(os.fsdecode(sys.executable), include_packages=True)


class ClientSourceFingerprintTests(unittest.TestCase):

    def test_fingerprint_is_label_sorted_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text("first\n", encoding="utf-8")
            second.write_text("second\n", encoding="utf-8")

            fingerprint = _client_source_fingerprint(
                [f"z={second}", f"a={first}"])
            changed = _client_source_fingerprint(
                [f"a={first}", f"z={second}"])
            self.assertEqual(fingerprint, changed)
            self.assertEqual(
                [entry["label"] for entry in fingerprint["files"]],
                ["a", "z"],
            )

            second.write_text("changed\n", encoding="utf-8")
            self.assertNotEqual(
                fingerprint["digest"],
                _client_source_fingerprint(
                    [f"a={first}", f"z={second}"])["digest"],
            )


if __name__ == "__main__":
    unittest.main()
