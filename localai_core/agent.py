"""Autonomous /goal Agent Engine with zero max turn limit.

Coordinates multi-turn reasoning and tool calling to achieve open-ended goals.
Supports OpenAI tool calling protocol, tag-based fallback extraction,
and interactive tool confirmation.
"""

import json
import re
import sys
from typing import Any, Callable, Dict, List, Optional

from .client import LocalAIClient
from .tools import TOOLS, dispatch_tool

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous AI agent working to achieve a user-specified objective.\n"
    "You have access to the following built-in tools:\n"
    "- execute_command(command: str): Run bash shell commands locally.\n"
    "- read_file(path: str): Read file contents.\n"
    "- write_file(path: str, content: str): Create or overwrite files.\n\n"
    "To call a tool, use function calling or output a JSON object like:\n"
    '{"tool_call": {"name": "<tool_name>", "arguments": { ... }}}\n\n'
    "Operating Guidelines:\n"
    "1. Plan your actions and execute tools step-by-step to fulfill the objective.\n"
    "2. Check command output and verify your changes carefully.\n"
    "3. When the goal is completely achieved and verified, you MUST output the completion token: <!-- GOAL_COMPLETE -->.\n"
)


def _parse_dict_for_tool(data: Any, call_id_prefix: str = "call") -> List[Dict[str, Any]]:
    """Helper to parse a dictionary for tool calls across diverse LLM formats."""
    if not isinstance(data, dict):
        return []

    # Format: {"tool_calls": [...]}
    if "tool_calls" in data and isinstance(data["tool_calls"], list):
        res = []
        for i, item in enumerate(data["tool_calls"]):
            sub = _parse_dict_for_tool(item, f"{call_id_prefix}_{i + 1}")
            res.extend(sub)
        if res:
            return res

    # Format: {"tool_call": {...}}
    if "tool_call" in data and isinstance(data["tool_call"], dict):
        data = data["tool_call"]

    name = None
    args = None
    if "name" in data:
        name = data["name"]
        args = data.get("arguments") or data.get("parameters") or {}
    elif "function" in data and isinstance(data["function"], dict):
        fn = data["function"]
        name = fn.get("name")
        args = fn.get("arguments") or fn.get("parameters") or {}
    elif "action" in data:
        name = data["action"]
        args = data.get("action_input") or data.get("arguments") or {}
    else:
        for known_tool in ("execute_command", "read_file", "write_file"):
            if known_tool in data:
                name = known_tool
                args = data[known_tool]
                break

    if not name:
        return []

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"raw": args}
    elif not isinstance(args, dict):
        args = {}

    return [{"id": call_id_prefix, "name": name, "arguments": args}]


def extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from an assistant response message.

    Supports:
    1. Standard OpenAI 'tool_calls' structure
    2. Fallback XML tag extraction (<tool_call>...</tool_call>)
    3. Markdown code blocks containing JSON
    4. Raw JSON in message content (e.g. {"tool_call": ...} or {"name": ...})

    Args:
        message: Assistant message dictionary.

    Returns:
        List of dictionaries with keys: 'id', 'name', 'arguments'.
    """
    calls: List[Dict[str, Any]] = []

    # 1. Standard OpenAI tool_calls structure
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls and isinstance(raw_tool_calls, list):
        for idx, item in enumerate(raw_tool_calls):
            parsed = _parse_dict_for_tool(item, item.get("id") or f"call_{idx + 1}")
            calls.extend(parsed)

        if calls:
            return calls

    # 2. Content-based extraction
    content = message.get("content") or ""
    if not isinstance(content, str) or not content.strip():
        return calls

    # 2a. Fallback tag extraction: <tool_call>...</tool_call>
    if "<tool_call>" in content:
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)
        for idx, match in enumerate(matches):
            match_str = match.strip()
            if match_str.startswith("```"):
                lines = match_str.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                match_str = "\n".join(lines).strip()

            try:
                data = json.loads(match_str)
                calls.extend(_parse_dict_for_tool(data, f"call_fallback_{idx + 1}"))
            except Exception:
                continue

        if calls:
            return calls

    # 2b. Markdown code block extraction: ```json ... ``` or ``` ... ```
    if "```" in content:
        code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        matches = re.findall(code_block_pattern, content)
        for idx, match in enumerate(matches):
            try:
                data = json.loads(match.strip())
                calls.extend(_parse_dict_for_tool(data, f"call_code_{idx + 1}"))
            except Exception:
                continue

        if calls:
            return calls

    # 2c. Direct raw JSON object in content
    trimmed = content.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            data = json.loads(trimmed)
            calls.extend(_parse_dict_for_tool(data, "call_json_1"))
            if calls:
                return calls
        except Exception:
            pass

    # 2d. Substring JSON search in content
    json_candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
    for idx, cand in enumerate(json_candidates):
        try:
            data = json.loads(cand)
            parsed = _parse_dict_for_tool(data, f"call_sub_{idx + 1}")
            calls.extend(parsed)
        except Exception:
            continue

    return calls


class GoalRunner:
    """Autonomous agent engine for executing goals via multi-turn tool calling."""

    def __init__(
        self,
        client: LocalAIClient,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        out_stream: Any = sys.stdout,
        in_stream: Any = None,
    ) -> None:
        """Initialize GoalRunner.

        Args:
            client: LocalAIClient instance to communicate with.
            model: Optional model name override.
            system_prompt: Optional custom system prompt.
            out_stream: Output stream for logging progress (default: stdout).
            in_stream: Input stream for interactive confirmations (default: None).
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.out_stream = out_stream
        self.in_stream = in_stream

    def run_goal(
        self,
        objective: str,
        model: Optional[str] = None,
        max_turns: int = 0,
        autonomous: bool = False,
        confirm_cb: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ) -> bool:
        """Execute autonomous goal loop until completion or interrupt.

        Args:
            objective: High-level goal or task description.
            model: Optional model override. Defaults to instance model or config default.
            max_turns: Max turns before aborting. 0 means unlimited turns (default 0).
            autonomous: When True, runs tools without per-turn confirmation.
            confirm_cb: Optional callback (tool_name, tool_args) -> bool to confirm execution.

        Returns:
            bool: True if goal completed (emitted <!-- GOAL_COMPLETE -->), False otherwise.
        """
        target_model = model or self.model
        if not target_model and hasattr(self.client, "config_manager") and self.client.config_manager:
            target_model = self.client.config_manager.get_default_model()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": objective},
        ]

        self.out_stream.write(f"Starting goal execution: {objective}\n")
        self.out_stream.write(f"Model: {target_model} | Autonomous: {autonomous} | Max turns: {max_turns}\n\n")
        self.out_stream.flush()

        turn = 0
        while True:
            if max_turns > 0 and turn >= max_turns:
                self.out_stream.write(f"\n[Reached maximum turn limit of {max_turns}]\n")
                self.out_stream.flush()
                return False

            turn += 1
            self.out_stream.write(f"--- Turn {turn} ---\n")
            self.out_stream.flush()

            try:
                resp = self.client.chat_completion(
                    messages=messages,
                    model=target_model,
                    stream=False,
                    tools=TOOLS,
                )
            except KeyboardInterrupt:
                self.out_stream.write("\n[Goal execution interrupted by user]\n")
                self.out_stream.flush()
                return False
            except Exception as exc:
                self.out_stream.write(f"\n[Error querying model: {exc}]\n")
                self.out_stream.flush()
                return False

            choices = resp.get("choices", [])
            if not choices:
                self.out_stream.write("\n[Error: Received empty response choices from model]\n")
                self.out_stream.flush()
                return False

            choice = choices[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""

            is_complete = "<!-- GOAL_COMPLETE -->" in content
            tool_calls = extract_tool_calls(msg)

            if content:
                self.out_stream.write(f"\n[Agent]: {content}\n")
                self.out_stream.flush()

            if not tool_calls:
                if is_complete:
                    self.out_stream.write("\n[Goal Complete]\n")
                    self.out_stream.flush()
                    return True

                # Model responded with text without tool calls or completion token
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Please continue working towards the goal using the available tools. Emit <!-- GOAL_COMPLETE --> when finished.",
                })
                continue

            # Tool calls are present
            raw_tool_calls = msg.get("tool_calls")
            if raw_tool_calls and isinstance(raw_tool_calls, list):
                messages.append(msg)
            else:
                synthetic_tool_calls = [
                    {
                        "id": tc.get("id") or f"call_{turn}_{i + 1}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                            if isinstance(tc["arguments"], (dict, list))
                            else str(tc["arguments"]),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ]
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": synthetic_tool_calls,
                })

            for idx, call in enumerate(tool_calls):
                tool_name = call["name"]
                tool_args = call["arguments"]
                call_id = call.get("id") or f"call_{turn}_{idx + 1}"

                self.out_stream.write(f"\n[Tool Request]: {tool_name}({json.dumps(tool_args)})\n")
                self.out_stream.flush()

                confirmed = True
                if not autonomous:
                    if confirm_cb is not None:
                        try:
                            confirmed = confirm_cb(tool_name, tool_args)
                        except KeyboardInterrupt:
                            self.out_stream.write("\n[Goal execution interrupted by user]\n")
                            self.out_stream.flush()
                            return False
                        except Exception as exc:
                            self.out_stream.write(f"\n[Confirmation callback error: {exc}]\n")
                            confirmed = False
                    else:
                        prompt = f"Execute tool '{tool_name}' with arguments {json.dumps(tool_args)}? [y/N]: "
                        self.out_stream.write(prompt)
                        self.out_stream.flush()
                        try:
                            if self.in_stream is not None:
                                line = self.in_stream.readline()
                                if not line:
                                    confirmed = False
                                else:
                                    confirmed = line.strip().lower() in ("y", "yes")
                            else:
                                line = input()
                                confirmed = line.strip().lower() in ("y", "yes")
                        except (KeyboardInterrupt, EOFError):
                            self.out_stream.write("\n[Goal execution interrupted by user]\n")
                            self.out_stream.flush()
                            return False

                if confirmed:
                    try:
                        result = dispatch_tool(tool_name, tool_args)
                    except KeyboardInterrupt:
                        self.out_stream.write("\n[Goal execution interrupted by user]\n")
                        self.out_stream.flush()
                        return False
                    except Exception as exc:
                        result = f"Error executing tool '{tool_name}': {exc}"
                else:
                    result = f"Tool execution for '{tool_name}' was rejected by user."

                self.out_stream.write(f"[Tool Result]:\n{result}\n")
                self.out_stream.flush()

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": result,
                })

                # Loop safeguard: detect repetitive duplicate calls
                call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                if getattr(self, "_last_call_sig", None) == call_sig:
                    self._consecutive_dupes = getattr(self, "_consecutive_dupes", 0) + 1
                    if self._consecutive_dupes >= 3:
                        messages.append({
                            "role": "user",
                            "content": f"System Alert: Tool '{tool_name}' has been repeated {self._consecutive_dupes} times with identical arguments. The action is already complete. If the goal has been achieved, you MUST output <!-- GOAL_COMPLETE --> now.",
                        })
                else:
                    self._last_call_sig = call_sig
                    self._consecutive_dupes = 1

            if is_complete:
                self.out_stream.write("\n[Goal Complete]\n")
                self.out_stream.flush()
                return True
