"""Unit tests for local-ai CLI entrypoint and argument parser."""

import io
import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from localai_core.cli import build_parser, main, _read_piped_input
from localai_core.client import LocalAIClient, LocalAIError
from localai_core.config import ConfigManager


class TestCLIParser(unittest.TestCase):
    """Test CLI argument parsing, subcommands, flags, and defaults."""

    def setUp(self):
        self.parser = build_parser()

    def test_root_flags(self):
        args = self.parser.parse_args(["--url", "http://custom-host:8080", "--config", "/tmp/cfg.json"])
        self.assertEqual(args.url, "http://custom-host:8080")
        self.assertEqual(args.config, "/tmp/cfg.json")

    def test_root_short_flags(self):
        args = self.parser.parse_args(["-u", "http://short-host:8080", "-c", "/tmp/short.json"])
        self.assertEqual(args.url, "http://short-host:8080")
        self.assertEqual(args.config, "/tmp/short.json")

    def test_no_subcommand_defaults_to_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.subcommand)

    def test_run_parser_defaults(self):
        args = self.parser.parse_args(["run", "Hello world"])
        self.assertEqual(args.subcommand, "run")
        self.assertEqual(args.prompt, "Hello world")
        self.assertIsNone(args.model)
        self.assertIsNone(args.system)
        self.assertEqual(args.temperature, 0.7)

    def test_run_parser_flags(self):
        args = self.parser.parse_args([
            "run",
            "Explain entropy",
            "-m", "custom-model",
            "-s", "You are a physics professor",
            "-t", "0.2",
        ])
        self.assertEqual(args.subcommand, "run")
        self.assertEqual(args.prompt, "Explain entropy")
        self.assertEqual(args.model, "custom-model")
        self.assertEqual(args.system, "You are a physics professor")
        self.assertEqual(args.temperature, 0.2)

    def test_run_parser_no_prompt_allowed(self):
        # Prompt can be omitted when piping via stdin
        args = self.parser.parse_args(["run"])
        self.assertEqual(args.subcommand, "run")
        self.assertIsNone(args.prompt)

    def test_chat_parser(self):
        args = self.parser.parse_args(["chat", "-m", "chat-model", "-s", "Chat system"])
        self.assertEqual(args.subcommand, "chat")
        self.assertEqual(args.model, "chat-model")
        self.assertEqual(args.system, "Chat system")

    def test_goal_parser_defaults(self):
        args = self.parser.parse_args(["goal", "Refactor module"])
        self.assertEqual(args.subcommand, "goal")
        self.assertEqual(args.objective, "Refactor module")
        self.assertIsNone(args.model)
        self.assertEqual(args.max_turns, 0)
        self.assertFalse(args.autonomous)

    def test_goal_parser_flags(self):
        args = self.parser.parse_args([
            "goal",
            "Write tests",
            "-m", "reasoning-model",
            "--max-turns", "15",
            "-y",
        ])
        self.assertEqual(args.subcommand, "goal")
        self.assertEqual(args.objective, "Write tests")
        self.assertEqual(args.model, "reasoning-model")
        self.assertEqual(args.max_turns, 15)
        self.assertTrue(args.autonomous)

    def test_goal_parser_missing_objective_fails(self):
        with self.assertRaises(SystemExit):
            with patch("sys.stderr", new_callable=io.StringIO):
                self.parser.parse_args(["goal"])

    def test_models_subcommand_and_alias(self):
        args1 = self.parser.parse_args(["models"])
        self.assertIn(args1.subcommand, ("models", "list-models"))

        args2 = self.parser.parse_args(["list-models"])
        self.assertIn(args2.subcommand, ("models", "list-models"))

    def test_status_subcommand_and_alias(self):
        args1 = self.parser.parse_args(["status"])
        self.assertIn(args1.subcommand, ("status", "ping"))

        args2 = self.parser.parse_args(["ping"])
        self.assertIn(args2.subcommand, ("status", "ping"))

    def test_config_subcommands(self):
        args_show = self.parser.parse_args(["config", "show"])
        self.assertEqual(args_show.subcommand, "config")
        self.assertEqual(args_show.config_action, "show")

        args_set_url = self.parser.parse_args(["config", "set-url", "http://new-ip:9000"])
        self.assertEqual(args_set_url.subcommand, "config")
        self.assertEqual(args_set_url.config_action, "set-url")
        self.assertEqual(args_set_url.url, "http://new-ip:9000")

        args_set_model = self.parser.parse_args(["config", "set-model", "Qwen-7B"])
        self.assertEqual(args_set_model.subcommand, "config")
        self.assertEqual(args_set_model.config_action, "set-model")
        self.assertEqual(args_set_model.model, "Qwen-7B")


class TestCLIMain(unittest.TestCase):
    """Test CLI main() dispatch, output, and exit codes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.json")
        self.out_stream = io.StringIO()
        self.err_stream = io.StringIO()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_main_no_subcommand_shows_help(self):
        code = main(
            ["-c", self.config_path],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        output = self.out_stream.getvalue()
        self.assertIn("usage:", output.lower())

    @patch("localai_core.cli.run_prompt")
    def test_main_run_with_prompt_arg(self, mock_run_prompt):
        mock_run_prompt.return_value = 0
        code = main(
            ["-c", self.config_path, "run", "Hello world", "-m", "custom-model", "-t", "0.5"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        mock_run_prompt.assert_called_once()
        _, kwargs = mock_run_prompt.call_args
        self.assertEqual(kwargs.get("prompt"), "Hello world")
        self.assertEqual(kwargs.get("model"), "custom-model")
        self.assertEqual(kwargs.get("temperature"), 0.5)

    @patch("localai_core.cli.run_prompt")
    def test_main_run_with_piped_stdin(self, mock_run_prompt):
        mock_run_prompt.return_value = 0
        in_stream = io.StringIO("Data from pipe")
        code = main(
            ["-c", self.config_path, "run"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
            in_stream=in_stream,
        )
        self.assertEqual(code, 0)
        mock_run_prompt.assert_called_once()
        _, kwargs = mock_run_prompt.call_args
        self.assertEqual(kwargs.get("prompt"), "Data from pipe")

    @patch("localai_core.cli.run_prompt")
    def test_main_run_combines_prompt_and_piped_stdin(self, mock_run_prompt):
        mock_run_prompt.return_value = 0
        in_stream = io.StringIO("Line 1\nLine 2")
        code = main(
            ["-c", self.config_path, "run", "Summarize this:"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
            in_stream=in_stream,
        )
        self.assertEqual(code, 0)
        mock_run_prompt.assert_called_once()
        _, kwargs = mock_run_prompt.call_args
        self.assertEqual(kwargs.get("prompt"), "Summarize this:\n\nLine 1\nLine 2")

    def test_main_run_no_prompt_and_empty_stdin_fails(self):
        in_stream = io.StringIO("")
        code = main(
            ["-c", self.config_path, "run"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
            in_stream=in_stream,
        )
        self.assertEqual(code, 1)
        err_out = self.err_stream.getvalue()
        self.assertIn("No prompt provided", err_out)

    def test_read_piped_input_non_closing_stream_with_prompt(self):
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b"ready line 1\nready line 2\n")
            with os.fdopen(r_fd, "r") as r_stream:
                data = _read_piped_input(r_stream, has_prompt=True)
                self.assertEqual(data, "ready line 1\nready line 2\n")
        finally:
            os.close(w_fd)

    def test_read_piped_input_without_prompt_reads_to_eof(self):
        r_fd, w_fd = os.pipe()
        os.write(w_fd, b"prompt data to eof\n")
        os.close(w_fd)
        with os.fdopen(r_fd, "r") as r_stream:
            data = _read_piped_input(r_stream, has_prompt=False)
            self.assertEqual(data, "prompt data to eof\n")

    def test_read_piped_input_tty_returns_empty(self):
        mock_tty = MagicMock()
        mock_tty.isatty.return_value = True
        self.assertEqual(_read_piped_input(mock_tty, has_prompt=False), "")
        self.assertEqual(_read_piped_input(mock_tty, has_prompt=True), "")

    @patch("localai_core.cli.start_chat_repl")
    def test_main_chat_dispatch(self, mock_chat_repl):
        code = main(
            ["-c", self.config_path, "chat", "-m", "chat-model", "-s", "System prompt"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        mock_chat_repl.assert_called_once()
        _, kwargs = mock_chat_repl.call_args
        self.assertEqual(kwargs.get("model"), "chat-model")
        self.assertEqual(kwargs.get("system"), "System prompt")

    @patch("localai_core.cli.GoalRunner")
    def test_main_goal_dispatch_success(self, mock_goal_runner_cls):
        mock_instance = MagicMock()
        mock_instance.run_goal.return_value = True
        mock_goal_runner_cls.return_value = mock_instance

        code = main(
            ["-c", self.config_path, "goal", "Test objective", "-y", "--max-turns", "5"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        mock_instance.run_goal.assert_called_once_with(
            objective="Test objective",
            model=None,
            max_turns=5,
            autonomous=True,
        )

    @patch("localai_core.cli.GoalRunner")
    def test_main_goal_dispatch_failure(self, mock_goal_runner_cls):
        mock_instance = MagicMock()
        mock_instance.run_goal.return_value = False
        mock_goal_runner_cls.return_value = mock_instance

        code = main(
            ["-c", self.config_path, "goal", "Unfinished objective"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 1)

    @patch.object(LocalAIClient, "list_models")
    def test_main_models_success(self, mock_list_models):
        mock_list_models.return_value = ["model-a", "model-b"]
        code = main(
            ["-c", self.config_path, "models"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("Available models:", out)
        self.assertIn("- model-a", out)
        self.assertIn("- model-b", out)

    @patch.object(LocalAIClient, "list_models")
    def test_main_models_empty(self, mock_list_models):
        mock_list_models.return_value = []
        code = main(
            ["-c", self.config_path, "list-models"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("No models found", out)

    @patch.object(LocalAIClient, "list_models")
    def test_main_models_error(self, mock_list_models):
        mock_list_models.side_effect = LocalAIError("Server unreachable")
        code = main(
            ["-c", self.config_path, "models"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 1)
        err = self.err_stream.getvalue()
        self.assertIn("Server unreachable", err)

    @patch.object(LocalAIClient, "health_check")
    def test_main_status_healthy(self, mock_health_check):
        mock_health_check.return_value = (True, "Healthy (2 models)", 0.045)
        code = main(
            ["-c", self.config_path, "status"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("Status:     Healthy", out)
        self.assertIn("0.045", out)

    @patch.object(LocalAIClient, "health_check")
    def test_main_ping_unhealthy(self, mock_health_check):
        mock_health_check.return_value = (False, "Connection refused", 5.0)
        code = main(
            ["-c", self.config_path, "ping"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 1)
        out = self.out_stream.getvalue()
        self.assertIn("Status:     Unhealthy", out)

    def test_main_config_show(self):
        code = main(
            ["-c", self.config_path, "config", "show"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("Config path:", out)
        self.assertIn("Server URL:", out)
        self.assertIn("Default Model:", out)

    def test_main_config_default_action_is_show(self):
        code = main(
            ["-c", self.config_path, "config"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("Config path:", out)

    def test_main_config_set_url(self):
        code = main(
            ["-c", self.config_path, "config", "set-url", "http://10.0.0.99:8080"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("http://10.0.0.99:8080", out)

        mgr = ConfigManager(config_path=self.config_path)
        self.assertEqual(mgr.get_url(), "http://10.0.0.99:8080")

    def test_main_config_set_model(self):
        code = main(
            ["-c", self.config_path, "config", "set-model", "Hermes-3-Custom"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("Hermes-3-Custom", out)

        mgr = ConfigManager(config_path=self.config_path)
        self.assertEqual(mgr.get_default_model(), "Hermes-3-Custom")

    def test_url_override_flag(self):
        with patch.object(LocalAIClient, "health_check") as mock_health:
            mock_health.return_value = (True, "OK", 0.01)
            code = main(
                ["--url", "http://override-host:9999", "-c", self.config_path, "status"],
                out_stream=self.out_stream,
                err_stream=self.err_stream,
            )
            self.assertEqual(code, 0)
            out = self.out_stream.getvalue()
            self.assertIn("http://override-host:9999", out)

    def test_timeout_override_flag(self):
        with patch.object(LocalAIClient, "__init__", return_value=None) as mock_init:
            with patch.object(LocalAIClient, "list_models", return_value=["m1"]):
                code = main(
                    ["--timeout", "240", "-c", self.config_path, "models"],
                    out_stream=self.out_stream,
                    err_stream=self.err_stream,
                )
                self.assertEqual(code, 0)
                _, kwargs = mock_init.call_args
                self.assertEqual(kwargs.get("timeout"), 240)

        with patch.object(LocalAIClient, "__init__", return_value=None) as mock_init:
            with patch("localai_core.cli.start_chat_repl"):
                main(
                    ["--timeout", "180", "-c", self.config_path, "chat"],
                    out_stream=self.out_stream,
                    err_stream=self.err_stream,
                )
                _, kwargs = mock_init.call_args
                self.assertEqual(kwargs.get("timeout"), 180)

        with patch.object(LocalAIClient, "__init__", return_value=None) as mock_init:
            with patch("localai_core.cli.start_chat_repl"):
                main(
                    ["-c", self.config_path, "chat", "--timeout", "90"],
                    out_stream=self.out_stream,
                    err_stream=self.err_stream,
                )
                _, kwargs = mock_init.call_args
                self.assertEqual(kwargs.get("timeout"), 90)

    def test_config_set_timeout(self):
        code = main(
            ["-c", self.config_path, "config", "set-timeout", "150"],
            out_stream=self.out_stream,
            err_stream=self.err_stream,
        )
        self.assertEqual(code, 0)
        out = self.out_stream.getvalue()
        self.assertIn("150s", out)
        mgr = ConfigManager(config_path=self.config_path)
        self.assertEqual(mgr.get_timeout(), 150)


class TestLauncherScript(unittest.TestCase):
    """Verify /home/deck/bin/local-ai script formatting, permissions, and execution."""

    LAUNCHER_PATH = "/home/deck/bin/local-ai"

    def test_launcher_exists_and_executable(self):
        self.assertTrue(
            os.path.exists(self.LAUNCHER_PATH),
            f"Launcher script not found at {self.LAUNCHER_PATH}",
        )
        self.assertTrue(
            os.access(self.LAUNCHER_PATH, os.X_OK),
            f"Launcher script is not executable: {self.LAUNCHER_PATH}",
        )

    def test_launcher_shebang_and_content(self):
        with open(self.LAUNCHER_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertTrue(content.startswith("#!/usr/bin/env bash"))
        self.assertIn("PYTHONPATH=", content)
        self.assertIn("localai_core.cli", content)

    def test_launcher_executes_help(self):
        result = subprocess.run(
            [self.LAUNCHER_PATH, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("local-ai", result.stdout)
        self.assertIn("run", result.stdout)
        self.assertIn("chat", result.stdout)
        self.assertIn("goal", result.stdout)


if __name__ == "__main__":
    unittest.main()
