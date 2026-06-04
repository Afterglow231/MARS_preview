from vllm.v1.request import Request
from typing import Optional
import json
import math
import os
import time
import re
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

logger = init_logger(__name__)

FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds

class InterceptionRecorder:
    """Record request lifecycle timestamps for Infercept."""

    def __init__(self):
        self.job_id_to_history = {}
        # Track scheduling operation timing.
        self.scheduling_times = []  # List of {start_time, end_time, duration}

    def print_history(self):
        from pathlib import Path
        import tempfile

        output_dir = Path(os.environ.get("RUN_OUTPUT_DIR") or "./scheduler_exp").expanduser()
        try:
            output_dir = output_dir.resolve()
        except Exception:
            output_dir = output_dir.absolute()

        final_path = output_dir / "scheduler_timestamps"
        tmp_path: Path | None = None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(output_dir),
                prefix="scheduler_timestamps.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                json.dump(self.job_id_to_history, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
        except Exception as exc:
            logger.warning("Failed to persist scheduler history to %s: %s",
                           final_path, exc)
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def request_arrives(self, request: Request):
        if request.job_id not in self.job_id_to_history:
            self.job_id_to_history[request.job_id] = []
        self.job_id_to_history[request.job_id].append({"Request_arrival_time": time.time()})
    
    def request_finished(self, request: Request):
        self.job_id_to_history[request.job_id].append({"Request_departure_time": time.time()})

    def request_evicted_from_running_queue(self, request: Request):
        self.job_id_to_history[request.job_id].append({"Request_evicted_from_running_queue_time": time.time()})

    def request_pinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"pinned_time": time.time()})

    def request_unpinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"unpinned_time": time.time()})

    def request_waiting_to_running(self, request: Request, prompt_length: int, hit_length: int = 0):
        self.job_id_to_history[request.job_id].append({
            "waiting_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })
    
    def request_evicted_to_running(self, request: Request, prompt_length: int, hit_length: int):
        self.job_id_to_history[request.job_id].append({
            "evicted_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })

class ToolCallParser:
    """Parser for extracting function calls from LLM output.

    Uses the same parsing logic as mini-swe-agent to extract bash commands
    from markdown code blocks and identify the function call.

    Also supports the MARS tool baseline JSON action format:
      {"action":"tool","tool":"run_cmd","commands":["python3 -m venv .venv"]}

    This can be extended for other datasets with different parsing logic.
    """

    def _extract_json_obj(self, text: str) -> Optional[dict]:
        s = (text or "").strip()
        if not s:
            return None

        # Strip fenced code blocks like ```json ... ``` or ``` ... ```.
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        decoder = json.JSONDecoder()
        starts = [i for i, ch in enumerate(s) if ch == "{"][:50]
        for start in starts:
            try:
                obj, _end = decoder.raw_decode(s[start:])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def parse(self, text: str) -> Optional[str]:
        """Parse LLM output and extract the function call name.

        Args:
            text: Output text from the LLM

        Returns:
            The function call name (e.g., "ls", "cd", "git"), or None if not found
        """
        # MARS action JSON.
        obj = self._extract_json_obj(text)
        if obj is not None:
            action = obj.get("action")
            if action is None and "tool" in obj:
                action = "tool"
            if str(action) == "tool":
                tool = obj.get("tool")
                cmds = obj.get("commands")
                first_cmd = None
                if isinstance(cmds, str) and cmds.strip():
                    first_cmd = cmds.strip()
                elif isinstance(cmds, list) and cmds:
                    first_cmd = str(cmds[0]).strip()
                if first_cmd:
                    words = first_cmd.split()
                    if words:
                        return words[0]
                if tool:
                    return str(tool)

        # Qwen3 Coder / native OpenAI-compatible XML tool calls.
        function_names = re.findall(r"<function=([^>\n]+)>", text or "")
        if function_names:
            for raw_name in function_names:
                tool_name = str(raw_name).strip()
                if not tool_name or tool_name in {"finish", "think"}:
                    continue
                return tool_name

        # Generic JSON function/tool names from native tool calling.
        for pattern in (
            r'"name"\s*:\s*"([^"]+)"',
            r'"tool_name"\s*:\s*"([^"]+)"',
        ):
            matches = re.findall(pattern, text or "")
            if not matches:
                continue
            for raw_name in matches:
                tool_name = str(raw_name).strip()
                if not tool_name or tool_name in {"finish", "think"}:
                    continue
                return tool_name

        # Bash fenced code blocks.
        actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)

        if len(actions) == 1:
            bash_action = actions[0].strip()
            # Extract the first word/command from the action
            words = bash_action.split()
            if words:
                return words[0]

        return None

class ToolCallEstimator:
    def __init__(
        self,
        tokenizer: Optional[AnyTokenizer] = None,
        model_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: Optional[str] = None,
        parser: Optional[ToolCallParser] = None,
    ):
        self.func_call_to_exec_time: dict[str, float] = {}
        self.record_func_call_to_exec_time: dict[str, list[float]] = {}
        self.record_all_exec_times: list[float] = []

        self.job_to_history: dict[str, list[dict[str, float]]] = {}

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            try:
                self.tokenizer = get_tokenizer(
                    tokenizer_name=model_name,
                    tokenizer_mode=tokenizer_mode,
                    trust_remote_code=trust_remote_code,
                    revision=tokenizer_revision,
                )
                logger.info(f"Initialized tokenizer for model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer for {model_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Initialize parser (can be customized for different datasets)
        self.parser = parser if parser is not None else ToolCallParser()

    def _history_key(self, request: Request) -> str:
        """Return a stable per-program key for estimator history.

        Agent benchmarks pass `job_id` through the API to connect multiple LLM turns from the same program.
        """
        return request.job_id or request.request_id

    def get_func_call_exec_time(self, func: str) -> Optional[float]:
        if func not in self.func_call_to_exec_time:
            return None
        return self.func_call_to_exec_time[func]
    
    def update_func_call_exec_time(self, job_id: str) -> None:
        """Update the moving average for the tool that just returned."""
        if job_id not in self.job_to_history or not self.job_to_history[job_id]:
            return

        last_record = self.job_to_history[job_id][-1]
        last_departure_time = last_record.get("departure_time")
        func = last_record.get("func_call")
        if last_departure_time is None or func is None:
            return
        exec_time = time.time() - last_departure_time

        if func not in self.record_func_call_to_exec_time:
            self.record_func_call_to_exec_time[func] = [exec_time]
        else:
            self.record_func_call_to_exec_time[func].append(exec_time)
        self.record_all_exec_times.append(exec_time)
        self.func_call_to_exec_time[func] = sum(self.record_func_call_to_exec_time[func]) / len(self.record_func_call_to_exec_time[func])
        return 

    def estimate_prefill_reload_s(
        self,
        prompt_tokens: int,
        *,
        env_prefix: str = "INFERCEPT",
        fallback_env_prefixes: tuple[str, ...] = ("CONTINUUM",),
    ) -> float:
        """Estimate prefill reload time in seconds.
        -InferCept uses this as the Tfwd(C_i) term in its memory-waste comparison. 
        -Continuum callers pass env_prefix="CONTINUUM" to preserve their existing configuration path. 
        """
        x = float(max(0, prompt_tokens))

        prefixes: list[str] = []
        for prefix in (env_prefix, *fallback_env_prefixes):
            if prefix and prefix not in prefixes:
                prefixes.append(prefix)

        for prefix in prefixes:
            quad = os.environ.get(f"{prefix}_PREFILL_RELOAD_QUAD")
            if not quad:
                continue
            try:
                a_str, b_str, c_str = quad.split(",")
                a = float(a_str.strip())
                b = float(b_str.strip())
                c = float(c_str.strip())
                return max(0.0, a * x * x + b * x + c)
            except Exception:
                logger.warning(
                    "Ignoring invalid %s_PREFILL_RELOAD_QUAD=%r", prefix, quad)

        for prefix in prefixes:
            s_per_tok_raw = os.environ.get(
                f"{prefix}_PREFILL_RELOAD_S_PER_TOKEN")
            if s_per_tok_raw is None:
                continue
            try:
                return max(0.0, float(s_per_tok_raw) * x)
            except ValueError:
                logger.warning(
                    "Ignoring invalid %s_PREFILL_RELOAD_S_PER_TOKEN=%r",
                    prefix,
                    s_per_tok_raw,
                )

        return max(0.0, 0.0002 * x)

    def calc_ttl_continuum_dynamic(
        self,
        *,
        tool: Optional[str],
        prefill_reload_s: float,
        queue_delay_s: float,
        eta: float,
        record_threshold_k: int = 100,
        default_ttl_s: float = FIXED_THRESHOLD_CONTINUUM,
    ) -> float:
        """
        Compute Continuum dynamic TTL (Sec. 4.2, Eq. (2)).

        Cold-start handling follows the paper:
          - If |S| <= K: use fixed TTL (T_default).
          - Else if |S[f]| <= K: use global records S.
          - Else: use fine-grained tool records S[f].
        """
        if tool is None:
            return 0.0

        base = float(queue_delay_s) * float(eta) + float(prefill_reload_s)
        if base <= 0.0:
            return 0.0

        global_records = self.record_all_exec_times
        tool_records = self.record_func_call_to_exec_time.get(tool, [])

        # Cold start.
        if len(global_records) <= record_threshold_k:
            return max(0.0, float(default_ttl_s))

        records = tool_records if len(tool_records) > record_threshold_k else global_records
        if not records:
            return 0.0

        candidates = sorted(set([0.0] + [float(t) for t in records if t >= 0.0]))
        if not candidates:
            return 0.0

        best_tau = 0.0
        best_reward = -math.inf
        n = float(len(records))
        for tau in candidates:
            #P(τ, f) = (1/|S[f]|) * sum I[t <= τ]
            p = sum(1.0 for t in records if t <= tau) / n
            reward = p * base - tau
            if reward > best_reward:
                best_reward = reward
                best_tau = tau

        if best_reward <= 0.0:
            return 0.0
        return float(best_tau)
    
    # Functions below will be called by outside functions.
    def set_up_pin(self, request: Request) -> float:
        if request.this_func_call is None:
            return 0
        
        this_func_call_exec_time = self.get_func_call_exec_time(request.this_func_call) or 0.0

        if this_func_call_exec_time > FIXED_THRESHOLD_CONTINUUM:
            return 0
        
        return FIXED_THRESHOLD_CONTINUUM

    def request_arrives(self, request: Request) -> None:
        history_key = self._history_key(request)
        logger.info(f"Request job id arriving: {request.job_id}, time is {time.time()}")
        # this is called when a job arrives in scheduler.py, if job is new, create an entry,
        if history_key not in self.job_to_history:
            self.job_to_history[history_key] = []
            if request.last_func_call is not None:
                logger.warning(
                    "Missing estimator history for request %s (job_id=%s) with "
                    "pre-populated last_func_call=%s; reinitializing history.",
                    request.request_id,
                    request.job_id,
                    request.last_func_call,
                )
            self.job_to_history[history_key].append({"arrival_time": request.arrival_time})
            return

        last_record = self.job_to_history[history_key][-1]
        derived_last_func_call = last_record.get("func_call")
        if request.last_func_call is None:
            request.last_func_call = derived_last_func_call
        elif derived_last_func_call is not None and request.last_func_call != derived_last_func_call:
            logger.info(
                "Request %s (job_id=%s) supplied last_func_call=%s overriding estimator history value=%s",
                request.request_id,
                request.job_id,
                request.last_func_call,
                derived_last_func_call,
            )
        logger.info(f"Request job id: {request.job_id}, last func call: {request.last_func_call}")

        self.update_func_call_exec_time(history_key)

        self.job_to_history[history_key].append({"arrival_time": request.arrival_time})
        return
    
    def request_finished(self, request: Request) -> None:
        history_key = self._history_key(request)
        logger.info(f"Request job id finishing: {request.job_id}, time is {time.time()}")

        # Detokenize output and parse function call
        this_func_call = request.this_func_call
        if this_func_call is None and self.tokenizer is not None and len(request.output_token_ids) > 0:
            try:
                # Detokenize the output tokens
                output_text = self.tokenizer.decode(
                    request.output_token_ids,
                    skip_special_tokens=True
                )

                # Parse function call using the parser
                this_func_call = self.parser.parse(output_text)

                if this_func_call:
                    logger.info(f"Extracted func_call: {this_func_call} from output")
                else:
                    logger.debug(f"No function call found in output: {output_text[:200]}")
            except Exception as e:
                logger.warning(f"Error detokenizing/parsing output for request {request.request_id}: {e}")

        request.this_func_call = this_func_call
        self.job_to_history.setdefault(history_key, []).append({
            "departure_time": time.time(),
            "func_call": request.this_func_call
        })
        return
