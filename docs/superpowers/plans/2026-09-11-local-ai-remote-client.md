# LocalAI Remote Client CLI & Autonomous Goal Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, lightweight CLI tool installed at `/home/deck/bin/local-ai` configured to run models on a remote LocalAI server (starting at `http://94.130.18.206:8080`) on dynamic cloud IPs, with chat, model management, and an autonomous `/goal` mode with zero max turn limit.

**Architecture:** Python 3 module structure located at `/home/deck/.local/share/local-ai/` with an executable launcher symlinked/installed at `/home/deck/bin/local-ai`. Uses a resilient HTTP client supporting SSE streaming, a config manager supporting dynamic IP updates and priority resolution, and an autonomous agent loop with OpenAI tool-calling protocol for `--max-turns 0` goal execution.

**Tech Stack:** Python 3, `requests` (for HTTP & SSE streaming), `unittest` / `pytest` for test-driven development.

**Spec:** `docs/superpowers/specs/2026-09-11-local-ai-remote-client-design.md`

## Global Constraints
- Executable path: `/home/deck/bin/local-ai`
- Base directory for source code: `/home/deck/.local/share/local-ai`
- Configuration file path: `/home/deck/.config/local-ai/config.json`
- Initial remote endpoint: `http://94.130.18.206:8080`
- Initial default model: `Hermes-3-Llama-3.2-3B-Q4_K_M.gguf`
- All created files owned by `deck:deck`
- Turn limit for `/goal` mode defaults to unlimited (`--max-turns 0`)
- Completion token: `<!-- GOAL_COMPLETE -->`
- Mandatory code review before completion

---

### Task 1: Configuration Management & Dynamic IP Resolution

**Files:**
- Create: `/home/deck/.local/share/local-ai/localai_core/config.py`
- Test: `/home/deck/.local/share/local-ai/tests/test_config.py`

**Interfaces:**
- Produces: `ConfigManager`
  - `get_url(cli_override: str = None) -> str`
  - `get_default_model(cli_override: str = None) -> str`
  - `set_url(new_url: str) -> None`
  - `set_default_model(new_model: str) -> None`
  - `load_config() -> dict`
  - `save_config(data: dict) -> None`

- [ ] **Step 1: Write the failing test for ConfigManager**
```python
# /home/deck/.local/share/local-ai/tests/test_config.py
import unittest
import os
import json
import tempfile
from localai_core.config import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = os.path.join(self.temp_dir.name, "config.json")
        self.manager = ConfigManager(config_path=self.config_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_config_created(self):
        url = self.manager.get_url()
        self.assertEqual(url, "http://94.130.18.206:8080")
        self.assertTrue(os.path.exists(self.config_file))

    def test_cli_override_priority(self):
        os.environ["LOCALAI_URL"] = "http://env-url:8080"
        try:
            url = self.manager.get_url(cli_override="http://cli-url:8080")
            self.assertEqual(url, "http://cli-url:8080")
        finally:
            del os.environ["LOCALAI_URL"]

    def test_env_var_priority_over_file(self):
        os.environ["LOCALAI_URL"] = "http://env-url:8080"
        try:
            url = self.manager.get_url()
            self.assertEqual(url, "http://env-url:8080")
        finally:
            del os.environ["LOCALAI_URL"]

    def test_set_url(self):
        self.manager.set_url("http://dynamic-ip:9000")
        self.assertEqual(self.manager.get_url(), "http://dynamic-ip:9000")
        with open(self.config_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data.get("url"), "http://dynamic-ip:9000")

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python3 -m unittest /home/deck/.local/share/local-ai/tests/test_config.py`
Expected: FAIL (ModuleNotFoundError: No module named 'localai_core')

- [ ] **Step 3: Implement ConfigManager**
Implement `/home/deck/.local/share/local-ai/localai_core/config.py` handling priority resolution (CLI flag > ENV var > config file > default) and atomic persistence.

- [ ] **Step 4: Run test to verify it passes**
Run: `PYTHONPATH=/home/deck/.local/share/local-ai python3 -m unittest discover -s /home/deck/.local/share/local-ai/tests`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add /home/deck/.local/share/local-ai/
git commit -m "feat(config): add ConfigManager with dynamic IP resolution"
```

---

### Task 2: Resilient LocalAI HTTP Client with Streaming

**Files:**
- Create: `/home/deck/.local/share/local-ai/localai_core/client.py`
- Test: `/home/deck/.local/share/local-ai/tests/test_client.py`

**Interfaces:**
- Consumes: `ConfigManager` from Task 1
- Produces: `LocalAIClient`
  - `list_models() -> List[str]`
  - `chat_completion(messages: List[dict], model: str = None, stream: bool = False, tools: List[dict] = None) -> Iterator[str] | dict`
  - `health_check() -> Tuple[bool, str, float]`

- [ ] **Step 1: Write the failing test for LocalAIClient**
Create `/home/deck/.local/share/local-ai/tests/test_client.py` mocking HTTP responses for `/v1/models` and `/v1/chat/completions` (including SSE stream parsing and retry behavior).

- [ ] **Step 2: Run test to verify it fails**
Run: `PYTHONPATH=/home/deck/.local/share/local-ai python3 -m unittest /home/deck/.local/share/local-ai/tests/test_client.py`
Expected: FAIL

- [ ] **Step 3: Implement LocalAIClient**
Write `/home/deck/.local/share/local-ai/localai_core/client.py` with retry logic, SSE token generator for streaming, model listing, and tool-call payload support.

- [ ] **Step 4: Run test to verify it passes**
Run: `PYTHONPATH=/home/deck/.local/share/local-ai python3 -m unittest /home/deck/.local/share/local-ai/tests/test_client.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add /home/deck/.local/share/local-ai/localai_core/client.py /home/deck/.local/share/local-ai/tests/test_client.py
git commit -m "feat(client): implement LocalAIClient with streaming and retries"
```

---

### Task 3: Interactive Chat & Single-Shot Query Handlers

**Files:**
- Create: `/home/deck/.local/share/local-ai/localai_core/chat.py`
- Create: `/home/deck/.local/share/local-ai/localai_core/runner.py`
- Test: `/home/deck/.local/share/local-ai/tests/test_chat.py`

**Interfaces:**
- Consumes: `LocalAIClient`, `ConfigManager`
- Produces:
  - `run_prompt(client: LocalAIClient, prompt: str, model: str = None) -> int`
  - `start_chat_repl(client: LocalAIClient, model: str = None) -> None`

- [ ] **Step 1: Write test for single-shot prompt runner and slash commands**
Write unit tests validating slash command dispatching (`/help`, `/models`, `/model`, `/status`, `/clear`) and input handling.

- [ ] **Step 2: Run test to verify it fails**
Expected: FAIL

- [ ] **Step 3: Implement runner.py and chat.py**
Implement streaming single-shot runner and terminal REPL with command history and slash commands.

- [ ] **Step 4: Run test to verify it passes**
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git commit -m "feat(chat): implement interactive REPL and single-shot runner"
```

---

### Task 4: Autonomous `/goal` Agent Engine (No Max Turn Limit)

**Files:**
- Create: `/home/deck/.local/share/local-ai/localai_core/agent.py`
- Create: `/home/deck/.local/share/local-ai/localai_core/tools.py`
- Test: `/home/deck/.local/share/local-ai/tests/test_agent.py`

**Interfaces:**
- Consumes: `LocalAIClient`
- Produces: `GoalRunner`
  - `run_goal(objective: str, model: str = None, max_turns: int = 0, autonomous: bool = False) -> bool`
  - Built-in tools: `execute_command`, `read_file`, `write_file`

- [ ] **Step 1: Write tests for GoalRunner and tools**
Test tool execution (running bash, reading/writing files) and autonomous loop termination upon receiving `<!-- GOAL_COMPLETE -->`.

- [ ] **Step 2: Run test to verify it fails**
Expected: FAIL

- [ ] **Step 3: Implement tools.py and agent.py**
Implement agent execution loop with `--max-turns 0` default (unlimited turns), tool schema, JSON/OpenAI tool call extraction, and safe execution.

- [ ] **Step 4: Run test to verify it passes**
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git commit -m "feat(agent): implement autonomous goal runner with unlimited turns"
```

---

### Task 5: Main CLI Entrypoint & Packaging (`local-ai`)

**Files:**
- Create: `/home/deck/.local/share/local-ai/localai_core/cli.py`
- Create: `/home/deck/bin/local-ai` (executable bash launcher)
- Test: `/home/deck/.local/share/local-ai/tests/test_cli.py`

**Interfaces:**
- Produces: CLI interface supporting `run`, `chat`, `goal`, `config`, `models`, `status`, `ping`.

- [ ] **Step 1: Write test for CLI argument parsing**
Test CLI options, defaults, subcommands, and flags.

- [ ] **Step 2: Run test to verify it fails**
Expected: FAIL

- [ ] **Step 3: Implement cli.py and launcher `/home/deck/bin/local-ai`**
Add executable permissions `chmod 755 /home/deck/bin/local-ai` and set ownership to `deck:deck`.

- [ ] **Step 4: Run test to verify it passes**
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git commit -m "feat(cli): add CLI entrypoint and install /home/deck/bin/local-ai"
```

---

### Task 6: End-to-End Live Integration Verification

**Files:**
- Create: `/home/deck/.local/share/local-ai/tests/test_live_integration.py`

- [ ] **Step 1: Run live health and model check**
`local-ai status` and `local-ai models` against `94.130.18.206:8080`.
- [ ] **Step 2: Run live prompt completion**
`local-ai run "Say 'LocalAI client online' in 5 words or less"`
- [ ] **Step 3: Run live autonomous `/goal` test**
`local-ai goal "Write 'Hello from LocalAI Goal' to /tmp/local_ai_goal_test.txt and verify it exists" --autonomous`
Verify `/tmp/local_ai_goal_test.txt` exists and contains the expected text.
- [ ] **Step 4: Test dynamic IP config change**
`local-ai config set-url http://94.130.18.206:8080` and verify config persistence.

---

### Task 7: Mandatory Code Review & Final Verification

- [ ] **Step 1: Dispatch code reviewer subagent via `requesting-code-review`**
- [ ] **Step 2: Address all Critical / Important issues raised by the reviewer**
- [ ] **Step 3: Run complete test suite**
- [ ] **Step 4: Final verification before claiming task complete**
