"""Main CLI Entrypoint for local-ai remote client and autonomous agent runner."""

import argparse
import io
import os
import sys
from typing import Any, List, Optional

from .agent import GoalRunner
from .chat import start_chat_repl
from .client import LocalAIClient, LocalAIError
from .config import ConfigManager
from .runner import run_prompt


def build_parser() -> argparse.ArgumentParser:
    """Build and return top-level ArgumentParser with subcommands and flags."""
    parser = argparse.ArgumentParser(
        prog="local-ai",
        description="LocalAI remote client & autonomous agent runner.",
    )
    parser.add_argument(
        "-u",
        "--url",
        help="Override remote LocalAI server URL (e.g. http://host:port)",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Available subcommands",
    )

    # run subcommand
    parser_run = subparsers.add_parser(
        "run",
        help="Run a single-shot query and stream the response",
    )
    parser_run.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt text to send (optional if piped via stdin)",
    )
    parser_run.add_argument(
        "-m",
        "--model",
        help="Model name override",
    )
    parser_run.add_argument(
        "-s",
        "--system",
        help="Optional system prompt",
    )
    parser_run.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    # chat subcommand
    parser_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive chat REPL session",
    )
    parser_chat.add_argument(
        "-m",
        "--model",
        help="Model name override",
    )
    parser_chat.add_argument(
        "-s",
        "--system",
        help="Optional system prompt",
    )

    # goal subcommand
    parser_goal = subparsers.add_parser(
        "goal",
        help="Execute an autonomous goal loop",
    )
    parser_goal.add_argument(
        "objective",
        help="Objective or task to accomplish",
    )
    parser_goal.add_argument(
        "-m",
        "--model",
        help="Model name override",
    )
    parser_goal.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Maximum number of turns (0 for unlimited, default: 0)",
    )
    parser_goal.add_argument(
        "-y",
        "--autonomous",
        action="store_true",
        help="Autonomous mode (do not ask for confirmation on tool execution)",
    )

    # models subcommand (with list-models alias)
    subparsers.add_parser(
        "models",
        aliases=["list-models"],
        help="List available models on remote server",
    )

    # status subcommand (with ping alias)
    subparsers.add_parser(
        "status",
        aliases=["ping"],
        help="Check remote server status, latency, and health",
    )

    # config subcommand
    parser_config = subparsers.add_parser(
        "config",
        help="View or update configuration",
    )
    config_subparsers = parser_config.add_subparsers(
        dest="config_action",
        help="Config actions",
    )
    config_subparsers.add_parser(
        "show",
        help="Show current configuration",
    )
    parser_set_url = config_subparsers.add_parser(
        "set-url",
        help="Set remote LocalAI server URL",
    )
    parser_set_url.add_argument(
        "url",
        help="New server URL",
    )
    parser_set_model = config_subparsers.add_parser(
        "set-model",
        help="Set default model name",
    )
    parser_set_model.add_argument(
        "model",
        help="New default model name",
    )

    return parser


def _read_piped_input(in_stream: Any, has_prompt: bool) -> str:
    """Read piped input from in_stream if available without blocking.

    If piped input has prompt argument, read ready lines or limit read to avoid
    hanging on non-closing streams. If prompt is omitted and stdin is pipe,
    standard in_stream.read() is used.
    """
    if in_stream is None:
        return ""

    # If stream is a TTY, do not read piped stdin
    try:
        if hasattr(in_stream, "isatty") and in_stream.isatty():
            return ""
    except Exception:
        pass

    try:
        if not has_prompt:
            return in_stream.read()

        # Check if stream has a working OS fileno
        fd = None
        try:
            fd = in_stream.fileno()
        except (AttributeError, io.UnsupportedOperation, ValueError):
            fd = None

        if fd is not None:
            import select
            try:
                r, _, _ = select.select([fd], [], [], 0.0)
                if not r:
                    return ""
            except Exception:
                return ""

            try:
                orig_blocking = os.get_blocking(fd)
                os.set_blocking(fd, False)
                chunks = []
                try:
                    while True:
                        try:
                            chunk = os.read(fd, 65536)
                            if not chunk:
                                break
                            chunks.append(chunk)
                            if sum(len(c) for c in chunks) >= 10 * 1024 * 1024:
                                break
                        except (BlockingIOError, InterruptedError):
                            break
                finally:
                    os.set_blocking(fd, orig_blocking)
                return b"".join(chunks).decode("utf-8", errors="replace")
            except Exception:
                try:
                    return in_stream.read(65536)
                except Exception:
                    return ""
        else:
            return in_stream.read()
    except Exception:
        return ""


def main(
    argv: Optional[List[str]] = None,
    out_stream: Any = sys.stdout,
    err_stream: Any = sys.stderr,
    in_stream: Any = sys.stdin,
) -> int:
    """CLI entrypoint function.

    Args:
        argv: Optional list of command-line arguments. Defaults to sys.argv[1:].
        out_stream: Output stream for standard output (default: sys.stdout).
        err_stream: Output stream for standard error (default: sys.stderr).
        in_stream: Input stream for standard input (default: sys.stdin).

    Returns:
        int: Process exit code (0 for success, non-zero for error).
    """
    if out_stream is None:
        out_stream = sys.stdout
    if err_stream is None:
        err_stream = sys.stderr
    if in_stream is None:
        in_stream = sys.stdin

    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Default action if no subcommand is given: show help
    if not args.subcommand:
        parser.print_help(file=out_stream)
        out_stream.flush()
        return 0

    # Initialize ConfigManager and LocalAIClient
    config_mgr = ConfigManager(config_path=args.config)
    url = config_mgr.get_url(cli_override=args.url)
    client = LocalAIClient(config_manager=config_mgr, base_url=url)

    # Subcommand: run
    if args.subcommand == "run":
        piped_data = _read_piped_input(in_stream, has_prompt=bool(args.prompt))
        piped_data = piped_data.strip() if piped_data else ""

        if args.prompt and piped_data:
            prompt = f"{args.prompt}\n\n{piped_data}"
        elif args.prompt:
            prompt = args.prompt
        elif piped_data:
            prompt = piped_data
        else:
            err_stream.write(
                "Error: No prompt provided. Provide a prompt argument or pipe text via stdin.\n"
            )
            err_stream.flush()
            return 1

        return run_prompt(
            client=client,
            prompt=prompt,
            model=args.model,
            system=args.system,
            temperature=args.temperature,
            out_stream=out_stream,
            err_stream=err_stream,
        )

    # Subcommand: chat
    elif args.subcommand == "chat":
        chat_in_stream = None
        if in_stream is not sys.stdin and in_stream is not None:
            chat_in_stream = in_stream
        elif hasattr(in_stream, "isatty") and not in_stream.isatty():
            chat_in_stream = in_stream

        start_chat_repl(
            client=client,
            model=args.model,
            system=args.system,
            out_stream=out_stream,
            in_stream=chat_in_stream,
        )
        return 0

    # Subcommand: goal
    elif args.subcommand == "goal":
        goal_in_stream = None
        if in_stream is not sys.stdin and in_stream is not None:
            goal_in_stream = in_stream
        elif hasattr(in_stream, "isatty") and not in_stream.isatty():
            goal_in_stream = in_stream

        runner = GoalRunner(
            client=client,
            model=args.model,
            out_stream=out_stream,
            in_stream=goal_in_stream,
        )
        success = runner.run_goal(
            objective=args.objective,
            model=args.model,
            max_turns=args.max_turns,
            autonomous=args.autonomous,
        )
        return 0 if success else 1

    # Subcommand: models / list-models
    elif args.subcommand in ("models", "list-models"):
        try:
            models = client.list_models()
            if not models:
                out_stream.write("No models found on server.\n")
            else:
                out_stream.write("Available models:\n")
                for m in models:
                    out_stream.write(f"  - {m}\n")
            out_stream.flush()
            return 0
        except Exception as exc:
            err_stream.write(f"Error fetching models: {exc}\n")
            err_stream.flush()
            return 1

    # Subcommand: status / ping
    elif args.subcommand in ("status", "ping"):
        try:
            is_healthy, status_msg, latency = client.health_check()
            health_str = "Healthy" if is_healthy else "Unhealthy"
            latency_str = (
                f"{latency:.4f}s"
                if isinstance(latency, (int, float))
                else f"{latency}"
            )
            out_stream.write(f"Server URL: {client.base_url}\n")
            out_stream.write(f"Status:     {health_str} ({status_msg})\n")
            out_stream.write(f"Latency:    {latency_str}\n")
            out_stream.flush()
            return 0 if is_healthy else 1
        except Exception as exc:
            err_stream.write(f"Error checking status: {exc}\n")
            err_stream.flush()
            return 1

    # Subcommand: config
    elif args.subcommand == "config":
        action = getattr(args, "config_action", None)
        if action in (None, "show"):
            out_stream.write(f"Config path:   {config_mgr.config_path}\n")
            out_stream.write(f"Server URL:    {config_mgr.get_url()}\n")
            out_stream.write(f"Default Model: {config_mgr.get_default_model()}\n")
            out_stream.write(f"Timeout:       {config_mgr.get_timeout()}s\n")
            out_stream.write(f"Max Retries:   {config_mgr.get_max_retries()}\n")
            out_stream.flush()
            return 0
        elif action == "set-url":
            config_mgr.set_url(args.url)
            out_stream.write(f"Updated server URL to: {config_mgr.get_url()}\n")
            out_stream.flush()
            return 0
        elif action == "set-model":
            config_mgr.set_default_model(args.model)
            out_stream.write(
                f"Updated default model to: {config_mgr.get_default_model()}\n"
            )
            out_stream.flush()
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
