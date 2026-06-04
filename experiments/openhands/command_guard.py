#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR_NAME = ".mars_runtime"


def _split_shell_segments(cmd: str) -> list[str]:
    parts = re.split(r"(?:&&|\|\||;|\|)", cmd)
    return [part.strip() for part in parts if part.strip()]


def _is_within_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _expand_path_token(token: str) -> Optional[Path]:
    text = str(token).strip()
    if not text:
        return None
    if text.startswith("${HOME}"):
        text = text.replace("${HOME}", os.path.expanduser("~"), 1)
    if not (
        text.startswith("/")
        or text.startswith("~/")
        or text == "~"
        or text.startswith("$HOME/")
        or text == "$HOME"
    ):
        return None
    expanded = os.path.expandvars(os.path.expanduser(text))
    try:
        return Path(expanded).resolve(strict=False)
    except Exception:
        return Path(expanded)


def _relative_parent_escape(token: str) -> bool:
    text = str(token).strip()
    if not text:
        return False
    if text.startswith("/"):
        return False
    return ".." in Path(text).parts


def _blocked_workspace_escape_reason(target: str, workspace_dir: Path, *, action: str) -> Optional[str]:
    expanded = _expand_path_token(target)
    workspace = workspace_dir.resolve(strict=False)

    if expanded is not None:
        if _is_within_path(expanded, workspace):
            return None
        return f"{action} outside workspace is blocked: {target}"

    if _relative_parent_escape(target):
        return f"{action} with parent-directory escape is blocked: {target}"

    return None


def _protected_roots() -> tuple[Path, ...]:
    home = Path.home().resolve(strict=False)
    roots = [
        REPO_ROOT.resolve(strict=False),
        home.resolve(strict=False),
        (home / "miniconda3").resolve(strict=False),
        (home / ".venvs").resolve(strict=False),
        (home / ".conda").resolve(strict=False),
        Path("/usr"),
        Path("/etc"),
        Path("/var"),
        Path("/opt"),
        Path("/bin"),
        Path("/sbin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/boot"),
        Path("/dev"),
        Path("/proc"),
        Path("/sys"),
        Path("/home"),
    ]
    return tuple(roots)


def _blocked_delete_target_reason(target: str, workspace_dir: Path) -> Optional[str]:
    generic = _blocked_workspace_escape_reason(
        target,
        workspace_dir,
        action="deleting path",
    )
    if generic is not None:
        return generic
    expanded = _expand_path_token(target)
    if expanded is None:
        return None
    for root in _protected_roots():
        if expanded == root or _is_within_path(expanded, root):
            return f"deleting protected path outside workspace: {target}"
    return None


def _blocked_cd_target_reason(target: str, workspace_dir: Path) -> Optional[str]:
    return _blocked_workspace_escape_reason(
        target,
        workspace_dir,
        action="changing directory",
    )


def _blocked_mutating_target_reason(target: str, workspace_dir: Path) -> Optional[str]:
    return _blocked_workspace_escape_reason(
        target,
        workspace_dir,
        action="writing path",
    )


def blocked_file_editor_path_reason(path: str, workspace_dir: Path) -> Optional[str]:
    target = str(path).strip()
    if not target:
        return "file_editor path is empty"
    try:
        expanded = Path(os.path.expandvars(os.path.expanduser(target))).resolve(strict=False)
    except Exception:
        expanded = Path(target)
    workspace = workspace_dir.resolve(strict=False)
    if _is_within_path(expanded, workspace):
        return None
    return f"file_editor path outside workspace is blocked: {path}"


def _iter_redirection_targets(segment: str) -> list[str]:
    targets: list[str] = []
    for match in re.finditer(r"(?:^|[\s])(?:\d*>>?|\d*>|>>?)\s*([^\s;&|]+)", segment):
        target = match.group(1).strip()
        if target:
            targets.append(target.strip("'\""))
    return targets


def _kill_operands(args: list[str]) -> list[str]:
    operands: list[str] = []
    idx = 0
    signal_spec_consumed = False
    while idx < len(args):
        token = str(args[idx]).strip()
        if not token:
            idx += 1
            continue
        lower = token.lower()
        if token == "--":
            operands.extend(item for item in args[idx + 1 :] if str(item).strip())
            break
        if lower in {"-l", "-L"}:
            return []
        if lower in {"-s", "-n"}:
            signal_spec_consumed = True
            idx += 2
            continue
        if re.fullmatch(r"-[A-Za-z][A-Za-z0-9-]*", token) and not signal_spec_consumed:
            signal_spec_consumed = True
            idx += 1
            continue
        if re.fullmatch(r"-\d+", token) and not signal_spec_consumed and not operands:
            signal_spec_consumed = True
            idx += 1
            continue
        operands.append(token)
        idx += 1
    return operands


def _blocked_process_control_reason(segment: str, exe_name: str, tokens: list[str]) -> Optional[str]:
    if exe_name in {"pkill", "killall"}:
        return f"broad process-control command is blocked: {exe_name}"

    if exe_name != "kill":
        return None

    seg_l = segment.lower()
    if (
        "pgrep" in seg_l or "pidof" in seg_l
    ) and ("$(" in segment or "`" in segment or "| xargs" in seg_l or "|xargs" in seg_l):
        return "dynamic process-selection kill command is blocked"

    operands = _kill_operands(tokens[1:])
    if not operands:
        return None
    if any(item in {"-1", "0"} for item in operands):
        return "broad process-control command is blocked: kill target"
    if any(re.fullmatch(r"-\d+", item) for item in operands):
        return "process-group kill command is blocked"
    return None


def blocked_terminal_command_reason(cmd: str, workspace_dir: Path) -> Optional[str]:
    if str(os.environ.get("MARS_ALLOW_UNSAFE_TERMINAL_CMD", "")).strip() == "1":
        return None

    segments = _split_shell_segments(cmd)
    for segment in segments:
        try:
            tokens = shlex.split(segment, posix=True)
        except Exception:
            continue
        if not tokens:
            continue

        exe = str(tokens[0]).lower()
        exe_name = Path(exe).name.lower()
        seg_l = segment.lower()

        process_control_reason = _blocked_process_control_reason(segment, exe_name, tokens)
        if process_control_reason is not None:
            return process_control_reason

        if exe_name in {"sudo", "doas"} or seg_l.startswith("sudo "):
            return "privilege escalation command is not allowed"

        if exe_name == "cd" and len(tokens) >= 2:
            reason = _blocked_cd_target_reason(tokens[1], workspace_dir)
            if reason is not None:
                return reason

        for target in _iter_redirection_targets(segment):
            reason = _blocked_mutating_target_reason(target, workspace_dir)
            if reason is not None:
                return reason

        if exe_name in {"conda", "mamba", "micromamba"}:
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"install", "remove", "update", "create"} for item in tail):
                return "conda environment mutation is blocked"

        is_pip_like = exe_name in {"pip", "pip3"}
        if exe_name in {"python", "python3"} and len(tokens) >= 3:
            is_pip_like = tokens[1] == "-m" and tokens[2].lower() == "pip"
        if is_pip_like:
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"install", "uninstall"} for item in tail):
                return "pip environment mutation is blocked"

        if exe_name == "uv" and len(tokens) >= 3:
            tail = [item.lower() for item in tokens[1:]]
            if tokens[1].lower() == "pip" and any(
                item in {"install", "uninstall", "sync"} for item in tail[1:]
            ):
                return "uv pip environment mutation is blocked"

        if exe_name in {"poetry", "pdm", "rye"}:
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"install", "add", "remove", "update", "sync", "lock"} for item in tail):
                return f"{exe_name} environment mutation is blocked"

        if exe_name == "cargo":
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"install", "new", "init"} for item in tail):
                return "cargo toolchain mutation is blocked"

        if exe_name == "rustup":
            tail = [item.lower() for item in tokens[1:]]
            if any(
                item in {
                    "install",
                    "update",
                    "default",
                    "override",
                    "toolchain",
                    "component",
                    "target",
                }
                for item in tail
            ):
                return "rustup toolchain mutation is blocked"

        if exe_name == "opam":
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"init", "install", "switch", "update", "upgrade", "reinstall"} for item in tail):
                return "opam toolchain mutation is blocked"

        if exe_name == "elan":
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"default", "install", "update", "toolchain"} for item in tail):
                return "elan toolchain mutation is blocked"

        if exe_name == "deno":
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"install", "upgrade"} for item in tail):
                return "deno toolchain mutation is blocked"

        if exe_name in {"npm", "pnpm", "yarn"}:
            tail = [item.lower() for item in tokens[1:]]
            if any(item in {"install", "add", "remove", "update"} for item in tail):
                if "-g" in tail or "--global" in tail or "global" in tail:
                    return f"{exe_name} global package mutation is blocked"

        if "nvm " in seg_l and any(word in seg_l for word in (" install ", " use ", " alias ", " uninstall ")):
            return "nvm toolchain mutation is blocked"

        if (
            "miniconda3" in seg_l or ".venvs" in seg_l or ".conda" in seg_l
        ) and any(word in seg_l for word in ("rm ", "mv ", "cp ", "ln ", "chmod ", "chown ")):
            return "mutating protected Python environment paths is blocked"

        if exe_name in {"rm", "rmdir", "unlink"}:
            targets = [item for item in tokens[1:] if not item.startswith("-")]
            for target in targets:
                if target in {"/", "/*", "~", "$HOME", "${HOME}"}:
                    return f"dangerous rm target is blocked: {target}"
                reason = _blocked_delete_target_reason(target, workspace_dir)
                if reason is not None:
                    return reason

        if exe_name in {"mv", "cp", "ln", "touch", "mkdir", "install", "chmod", "chown", "tee"}:
            targets = [item for item in tokens[1:] if not item.startswith("-")]
            for target in targets:
                if exe_name == "tee" and target in {"-", "/dev/stdout", "/dev/stderr"}:
                    continue
                reason = _blocked_mutating_target_reason(target, workspace_dir)
                if reason is not None:
                    return reason

        if exe_name == "git":
            if len(tokens) >= 3 and tokens[1].lower() == "reset" and "--hard" in tokens[2:]:
                return "destructive git reset is blocked"
            if len(tokens) >= 3 and tokens[1].lower() == "clean" and any(
                flag in tokens[2:] for flag in ("-fd", "-xdf", "-fdx", "-ffd", "-ffdx")
            ):
                return "destructive git clean is blocked"
            if len(tokens) >= 3 and tokens[1].lower() in {"checkout", "restore"} and "--" in tokens[2:]:
                return "destructive git checkout/restore is blocked"

    return None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MARS OpenHands command guard helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    check_terminal = sub.add_parser(
        "check-terminal",
        help="Print the blocked reason for a terminal command, if any.",
    )
    check_terminal.add_argument("--workspace-dir", type=str, required=True)
    check_terminal.add_argument("--command", type=str, required=True)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.cmd == "check-terminal":
        reason = blocked_terminal_command_reason(
            str(args.command),
            Path(args.workspace_dir).expanduser().resolve(),
        )
        if reason is None:
            return 0
        print(reason)
        return 10
    raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
