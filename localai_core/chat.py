"""Interactive Chat REPL and session management for LocalAI remote client."""

import os
import sys
from typing import Any, Dict, List, Optional

from .client import LocalAIClient

DEFAULT_HISTORY_PATH: str = "/home/deck/.config/local-ai/history"
DEFAULT_HISTORY_FILE: str = DEFAULT_HISTORY_PATH


class ChatSession:
    """Interactive multi-turn chat session with slash command support."""

    def __init__(
        self,
        client: LocalAIClient,
        model: Optional[str] = None,
        system: Optional[str] = None,
        out_stream: Any = sys.stdout,
        hist_file: Optional[str] = None,
    ) -> None:
        self.client = client
        if model:
            self.model = model
        elif hasattr(client, "config_manager") and client.config_manager:
            self.model = client.config_manager.get_default_model()
        else:
            self.model = "default"
        self.system = system
        self.out_stream = out_stream
        self.hist_file = (
            os.path.abspath(os.path.expanduser(hist_file))
            if hist_file
            else DEFAULT_HISTORY_PATH
        )
        self.messages: List[Dict[str, str]] = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

        self.load_history()

    @staticmethod
    def _ensure_deck_ownership(path: str) -> None:
        """Set deck:deck ownership on path if running as root and deck user exists."""
        if hasattr(os, "chown") and os.name == "posix":
            try:
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    import grp
                    import pwd

                    deck_uid = pwd.getpwnam("deck").pw_uid
                    deck_gid = grp.getgrnam("deck").gr_gid
                    os.chown(path, deck_uid, deck_gid)
            except Exception:
                pass

    def load_history(self) -> None:
        """Load history from hist_file if it exists."""
        try:
            import readline
            if os.path.exists(self.hist_file):
                readline.read_history_file(self.hist_file)
        except Exception:
            pass

    def save_history(self) -> None:
        """Save history to hist_file on exit."""
        try:
            import readline
            hist_dir = os.path.dirname(self.hist_file)
            if hist_dir and not os.path.exists(hist_dir):
                os.makedirs(hist_dir, exist_ok=True)
                self._ensure_deck_ownership(hist_dir)

            readline.write_history_file(self.hist_file)
            self._ensure_deck_ownership(self.hist_file)
        except Exception:
            pass

    def handle_command(self, cmd_line: str) -> bool:
        """Handle slash command.

        Returns:
            bool: True to continue session, False to exit session.
        """
        parts = cmd_line.strip().split(None, 1)
        if not parts:
            return True

        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            self.out_stream.write("Goodbye!\n")
            self.out_stream.flush()
            return False

        elif cmd == "/help":
            help_text = (
                "Available commands:\n"
                "  /help             Show this help message\n"
                "  /models           List available models on remote server\n"
                "  /model [<name>]   View or switch active model\n"
                "  /status           Show server connection status, latency, and active model\n"
                "  /clear            Clear conversation history\n"
                "  /exit, /quit      Exit the chat session\n"
            )
            self.out_stream.write(help_text)
            self.out_stream.flush()
            return True

        elif cmd == "/models":
            try:
                models = self.client.list_models()
                if not models:
                    self.out_stream.write("No models found on server.\n")
                else:
                    self.out_stream.write("Available models:\n")
                    for m in models:
                        active_marker = " (active)" if m == self.model else ""
                        self.out_stream.write(f"  - {m}{active_marker}\n")
            except Exception as e:
                self.out_stream.write(f"Failed to fetch models: {e}\n")
            self.out_stream.flush()
            return True

        elif cmd == "/model":
            if arg:
                self.model = arg
                self.out_stream.write(f"Model switched to: {self.model}\n")
            else:
                self.out_stream.write(f"Active model: {self.model}\n")
            self.out_stream.flush()
            return True

        elif cmd == "/status":
            try:
                is_healthy, status_msg, latency = self.client.health_check()
                health_str = "Healthy" if is_healthy else "Unhealthy"
                base_url = getattr(self.client, "base_url", "Unknown")
                latency_str = (
                    f"{latency:.4f}s"
                    if isinstance(latency, (int, float))
                    else f"{latency}"
                )
                self.out_stream.write(
                    f"Server URL: {base_url}\n"
                    f"Active Model: {self.model}\n"
                    f"Status: {health_str} ({status_msg})\n"
                    f"Latency: {latency_str}\n"
                )
            except Exception as e:
                self.out_stream.write(f"Failed to check status: {e}\n")
            self.out_stream.flush()
            return True

        elif cmd == "/clear":
            self.messages.clear()
            if self.system:
                self.messages.append({"role": "system", "content": self.system})
            self.out_stream.write("Conversation history cleared.\n")
            self.out_stream.flush()
            return True

        else:
            self.out_stream.write(
                f"Unknown command: {cmd}. Type /help for available commands.\n"
            )
            self.out_stream.flush()
            return True

    def send_message(self, user_text: str) -> str:
        """Send a user message, stream response tokens to out_stream, and update history.

        Args:
            user_text: Prompt text from user.

        Returns:
            str: Full assembled assistant response string.
        """
        self.messages.append({"role": "user", "content": user_text})
        chunks: List[str] = []

        messages_snapshot = list(self.messages)
        try:
            stream = self.client.chat_completion(
                messages=messages_snapshot,
                model=self.model,
                stream=True,
            )
            for chunk in stream:
                self.out_stream.write(chunk)
                self.out_stream.flush()
                chunks.append(chunk)

            self.out_stream.write("\n")
            self.out_stream.flush()
            full_response = "".join(chunks)
            self.messages.append({"role": "assistant", "content": full_response})
            return full_response

        except KeyboardInterrupt:
            self.out_stream.write("\n[Generation interrupted]\n")
            self.out_stream.flush()
            if chunks:
                full_response = "".join(chunks)
                self.messages.append({"role": "assistant", "content": full_response})
                return full_response
            else:
                self.messages.pop()
                return ""

        except Exception as exc:
            prefix = "\n" if chunks else ""
            self.out_stream.write(f"{prefix}Error: {exc}\n")
            self.out_stream.flush()
            if chunks:
                full_response = "".join(chunks)
                self.messages.append({"role": "assistant", "content": full_response})
            else:
                self.messages.pop()
            raise

    def start(self, in_stream: Any = None) -> None:
        """Start the interactive chat REPL loop."""
        try:
            import readline  # Enable arrow keys, line editing, and history
        except ImportError:
            pass

        self.out_stream.write(f"Starting chat with model '{self.model}'.\n")
        self.out_stream.write(
            "Type /help for commands, Ctrl+C to interrupt, Ctrl+D or /exit to quit.\n\n"
        )
        self.out_stream.flush()

        try:
            while True:
                try:
                    if in_stream is not None:
                        self.out_stream.write(">>> ")
                        self.out_stream.flush()
                        line = in_stream.readline()
                        if not line:
                            self.out_stream.write("\nGoodbye!\n")
                            self.out_stream.flush()
                            break
                        user_input = line.strip()
                    else:
                        self.out_stream.flush()
                        user_input = input(">>> ").strip()
                except EOFError:
                    self.out_stream.write("\nGoodbye!\n")
                    self.out_stream.flush()
                    break
                except KeyboardInterrupt:
                    self.out_stream.write("\n")
                    self.out_stream.flush()
                    continue

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                try:
                    self.send_message(user_input)
                except KeyboardInterrupt:
                    pass
                except Exception:
                    pass
        finally:
            self.save_history()


def start_chat_repl(
    client: LocalAIClient,
    model: Optional[str] = None,
    system: Optional[str] = None,
    out_stream: Any = sys.stdout,
    in_stream: Any = None,
    hist_file: Optional[str] = None,
) -> None:
    """Entrypoint to launch the interactive ChatSession REPL."""
    kwargs: Dict[str, Any] = {"client": client, "model": model}
    if system is not None:
        kwargs["system"] = system
    if out_stream is not sys.stdout:
        kwargs["out_stream"] = out_stream
    if hist_file is not None:
        kwargs["hist_file"] = hist_file
    session = ChatSession(**kwargs)
    session.start(in_stream=in_stream)
