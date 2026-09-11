# local-ai-cli

A fast, resilient command-line interface and autonomous agent runner for [LocalAI](https://localai.io) instances running locally or on dynamic remote cloud servers.

---

## Features

- **Dynamic IP & Port Management**: Effortlessly adapt when your cloud host IP or port changes with `local-ai config set-url`.
- **Autonomous `/goal` Runner (Zero Max Turn Limit)**: Execute multi-turn agentic goals with tools (`execute_command`, `read_file`, `write_file`) running indefinitely (`--max-turns 0`) until explicit completion (`<!-- GOAL_COMPLETE -->`).
- **Interactive Streaming Chat**: Terminal REPL with live token streaming, persistent readline command history, and in-chat slash commands (`/help`, `/models`, `/model <name>`, `/status`, `/clear`, `/goal`).
- **Single-Shot & Piped Execution**: Stream completions directly to stdout or pipe context through standard input (`cat file.py | local-ai run "review"`).
- **Network Resilience**: Automatic retry with exponential backoff and randomized jitter on transient cloud network drops.
- **Safety & Permissions**: Automatically executes tools with atomic write semantics, non-blocking pipeline handling, and user privilege preservation.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/1456319/local-ai-cli.git ~/.local/share/local-ai

# Symlink launcher into your PATH
mkdir -p ~/bin
ln -sf ~/.local/share/local-ai/bin/local-ai ~/bin/local-ai
ln -sf ~/bin/local-ai ~/bin/localai

# Add alias to ~/.bashrc if desired
echo "alias localai='local-ai'" >> ~/.bashrc
```

Ensure `~/bin` is in your `PATH`.

---

## Quick Start

### 1. Check Server Status
```bash
local-ai status
```

### 2. Configure Dynamic Remote Host
```bash
local-ai config set-url http://94.130.18.206:8080
local-ai config show
```

### 3. List Available Models
```bash
local-ai models
```

### 4. Interactive Chat
```bash
local-ai chat
```
Inside chat, you can use:
- `/models`: list available models
- `/model <name>`: switch active model
- `/status`: check latency and connection health
- `/clear`: clear context
- `/help`: show command reference

### 5. Single-Shot & Piped Execution
```bash
# Direct prompt
local-ai run "Explain quantum computing in 10 words"

# Piped input
echo "def hello(): print('world')" | local-ai run "add type hints to this function"
```

### 6. Autonomous Goal Loop
Run an open-ended autonomous agent with tools (`execute_command`, `read_file`, `write_file`) running until task completion:
```bash
# Interactive tool confirmation
local-ai goal "Inspect system memory and summarize top processes"

# Fully autonomous mode with unlimited turns (default: --max-turns 0)
local-ai goal "Create a deployment script and test its syntax" --autonomous
```

---

## Running Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

---

## License

MIT License.
