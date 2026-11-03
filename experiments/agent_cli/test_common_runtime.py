from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _run_readiness(tmp_path: Path, *, kv_payload: dict) -> subprocess.CompletedProcess[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = (
                {"data": [{"id": "model"}]}
                if self.path == "/v1/models"
                else kv_payload
            )
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        repo = Path(__file__).resolve().parents[2]
        runtime = repo / "experiments" / "scripts" / "common_runtime.sh"
        snapshot = tmp_path / "kv_ready.json"
        command = (
            f"source {shlex.quote(str(runtime))}; "
            f"mars_wait_for_vllm {shlex.quote(sys.executable)} "
            f"http://127.0.0.1:{server.server_address[1]}/v1 0.8 '' '' 1 "
            f"{shlex.quote(str(snapshot))}"
        )
        return subprocess.run(
            ["bash", "-c", command],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_mars_readiness_requires_structured_kv_stats(tmp_path: Path) -> None:
    result = _run_readiness(
        tmp_path,
        kv_payload={
            "block_size": 16,
            "num_gpu_blocks": 100,
            "num_free_blocks": 100,
            "usage": 0.0,
        },
    )

    assert result.returncode == 0, result.stderr
    snapshot = json.loads((tmp_path / "kv_ready.json").read_text())
    assert snapshot["stats"]["num_gpu_blocks"] == 100


def test_mars_readiness_rejects_http_200_error_payload(tmp_path: Path) -> None:
    result = _run_readiness(tmp_path, kv_payload={"error": "utility RPC failed"})

    assert result.returncode != 0
    assert not (tmp_path / "kv_ready.json").exists()
