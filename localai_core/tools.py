"""Built-in system tools for LocalAI autonomous agent.

Provides execution primitives for bash commands, file reading, and file writing,
along with OpenAI-compatible tool calling schemas and tool dispatching.
"""

import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional


def _ensure_deck_ownership(target_path: str) -> None:
    """Set deck:deck ownership on target_path if on POSIX and deck user exists."""
    if hasattr(os, "chown") and os.name == "posix":
        try:
            import grp
            import pwd

            deck_uid = pwd.getpwnam("deck").pw_uid
            deck_gid = grp.getgrnam("deck").gr_gid
            os.chown(target_path, deck_uid, deck_gid)
        except Exception:
            pass


def execute_command(command: str, timeout: int = 120) -> str:
    """Execute a bash shell command and capture stdout, stderr, and exit code.

    If running as root, preserves user context by executing as the 'deck' user.

    Args:
        command: The bash command string to execute.
        timeout: Maximum duration in seconds before timing out (default 120).

    Returns:
        Formatted string containing exit code, stdout, and stderr.
    """
    run_kwargs: Dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }

    if getattr(os, "geteuid", None) and os.geteuid() == 0:
        try:
            import pwd

            pw = pwd.getpwnam("deck")
            run_kwargs["user"] = pw.pw_uid
            run_kwargs["group"] = pw.pw_gid
            env = os.environ.copy()
            env["USER"] = "deck"
            env["LOGNAME"] = "deck"
            env["HOME"] = pw.pw_dir
            run_kwargs["env"] = env

            cwd = os.getcwd()
            if cwd.startswith("/root"):
                run_kwargs["cwd"] = pw.pw_dir
        except Exception:
            pass

    try:
        res = subprocess.run(["bash", "-c", command], **run_kwargs)
        parts = [f"Exit code: {res.returncode}"]
        if res.stdout:
            parts.append(f"STDOUT:\n{res.stdout}")
        if res.stderr:
            parts.append(f"STDERR:\n{res.stderr}")
        if not res.stdout and not res.stderr:
            parts.append("(no output)")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as exc:
        return f"Error executing command: {exc}"


def read_file(path: str) -> str:
    """Read the UTF-8 text content of a file.

    Args:
        path: Path to the target file.

    Returns:
        The text content of the file or an error message.
    """
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(abs_path):
            return f"Error: File not found: '{path}'"
        if os.path.isdir(abs_path):
            return f"Error: Path is a directory, not a file: '{path}'"

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as exc:
        return f"Error reading file '{path}': {exc}"


def write_file(path: str, content: str) -> str:
    """Write text content to a file atomically, ensuring parent directories exist and setting deck ownership.

    Args:
        path: Path to the target file.
        content: Text content to write.

    Returns:
        Success message with path and character count, or an error message.
    """
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and not os.path.exists(parent_dir):
            missing_dirs = []
            curr = parent_dir
            while curr and not os.path.exists(curr):
                missing_dirs.append(curr)
                parent = os.path.dirname(curr)
                if parent == curr:
                    break
                curr = parent

            os.makedirs(parent_dir, exist_ok=True)
            for d in missing_dirs:
                _ensure_deck_ownership(d)

        target_dir = parent_dir if parent_dir else "."

        target_mode = 0o644
        if os.path.exists(abs_path):
            try:
                target_mode = os.stat(abs_path).st_mode & 0o777
            except OSError:
                target_mode = 0o644

        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=target_dir,
                prefix=".tmp_write_",
                suffix=".tmp",
            )
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            try:
                os.chmod(temp_path, target_mode)
            except OSError:
                pass

            _ensure_deck_ownership(temp_path)
            os.replace(temp_path, abs_path)
            _ensure_deck_ownership(abs_path)
            return f"Successfully wrote {len(content)} characters to {abs_path}"
        finally:
            if temp_path and temp_path != abs_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    except Exception as exc:
        return f"Error writing to file '{path}': {exc}"


def dispatch_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch a tool invocation by name and arguments dictionary.

    Args:
        name: Tool function name ('execute_command', 'read_file', 'write_file').
        arguments: Dictionary of arguments for the tool.

    Returns:
        String execution result or diagnostic error message.
    """
    if not isinstance(arguments, dict):
        return f"Error: Arguments must be a dictionary, got {type(arguments).__name__}"

    if name == "execute_command":
        cmd = arguments.get("command")
        if cmd is None:
            return "Error: Missing required argument 'command' for execute_command"
        timeout_arg = arguments.get("timeout", 120)
        try:
            timeout_val = int(timeout_arg)
        except (ValueError, TypeError):
            timeout_val = 120
        return execute_command(command=str(cmd), timeout=timeout_val)

    elif name == "read_file":
        path = arguments.get("path")
        if path is None:
            return "Error: Missing required argument 'path' for read_file"
        return read_file(path=str(path))

    elif name == "write_file":
        path = arguments.get("path")
        content = arguments.get("content")
        if path is None:
            return "Error: Missing required argument 'path' for write_file"
        if content is None:
            return "Error: Missing required argument 'content' for write_file"
        return write_file(path=str(path), content=str(content))

    else:
        return f"Error: Unknown tool '{name}'"


# OpenAI-compatible function calling schemas
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a bash shell command locally and capture stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash shell command line to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default: 120).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the UTF-8 text content of a local file at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a local file, creating any missing parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]

TOOLS = TOOL_DEFINITIONS
