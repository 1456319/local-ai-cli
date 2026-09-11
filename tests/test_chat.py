import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from localai_core.client import LocalAIClient, LocalAIError, LocalAIConnectionError, LocalAIHTTPError
from localai_core.runner import run_prompt
from localai_core.chat import ChatSession, start_chat_repl


class TestRunPrompt(unittest.TestCase):
    def setUp(self):
        self.mock_client = mock.create_autospec(LocalAIClient, instance=True)

    def test_run_prompt_success_streams_tokens(self):
        self.mock_client.chat_completion.return_value = iter(["Hello", " ", "world", "!"])
        out_buf = io.StringIO()

        exit_code = run_prompt(
            client=self.mock_client,
            prompt="Say hello",
            model="test-model",
            system="You are a helpful assistant.",
            temperature=0.5,
            out_stream=out_buf,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(out_buf.getvalue(), "Hello world!\n")
        self.mock_client.chat_completion.assert_called_once_with(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"},
            ],
            model="test-model",
            stream=True,
            temperature=0.5,
        )

    def test_run_prompt_default_parameters(self):
        self.mock_client.chat_completion.return_value = iter(["OK"])
        out_buf = io.StringIO()

        exit_code = run_prompt(
            client=self.mock_client,
            prompt="Check default",
            out_stream=out_buf,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(out_buf.getvalue(), "OK\n")
        self.mock_client.chat_completion.assert_called_once_with(
            messages=[{"role": "user", "content": "Check default"}],
            model=None,
            stream=True,
            temperature=0.7,
        )

    def test_run_prompt_connection_error_returns_1(self):
        self.mock_client.chat_completion.side_effect = LocalAIConnectionError("Connection failed")
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        exit_code = run_prompt(
            client=self.mock_client,
            prompt="Hello",
            out_stream=out_buf,
            err_stream=err_buf,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(out_buf.getvalue(), "")
        self.assertIn("Connection failed", err_buf.getvalue())

    def test_run_prompt_stream_exception_returns_1(self):
        def failing_stream():
            yield "Partial"
            raise LocalAIHTTPError(500, "Internal error")

        self.mock_client.chat_completion.return_value = failing_stream()
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        exit_code = run_prompt(
            client=self.mock_client,
            prompt="Hello",
            out_stream=out_buf,
            err_stream=err_buf,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(out_buf.getvalue(), "Partial")
        self.assertIn("Internal error", err_buf.getvalue())

    def test_run_prompt_keyboard_interrupt(self):
        def interrupted_stream():
            yield "First"
            raise KeyboardInterrupt()

        self.mock_client.chat_completion.return_value = interrupted_stream()
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        exit_code = run_prompt(
            client=self.mock_client,
            prompt="Hello",
            out_stream=out_buf,
            err_stream=err_buf,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("First\n", out_buf.getvalue())
        self.assertIn("Interrupted", err_buf.getvalue())



class TestChatSession(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.hist_file = os.path.join(self.temp_dir, "history")
        self.hist_patcher = mock.patch("localai_core.chat.DEFAULT_HISTORY_PATH", self.hist_file)
        self.hist_patcher.start()
        self.mock_client = mock.create_autospec(LocalAIClient, instance=True)
        self.mock_client.base_url = "http://94.130.18.206:8080"
        self.mock_config = mock.MagicMock()
        self.mock_config.get_default_model.return_value = "default-llama"
        self.mock_client.config_manager = self.mock_config

    def tearDown(self):
        self.hist_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_defaults(self):
        session = ChatSession(client=self.mock_client)
        self.assertEqual(session.model, "default-llama")
        self.assertEqual(session.messages, [])

    def test_init_with_custom_model_and_system(self):
        session = ChatSession(
            client=self.mock_client,
            model="custom-model",
            system="System directive",
        )
        self.assertEqual(session.model, "custom-model")
        self.assertEqual(session.messages, [{"role": "system", "content": "System directive"}])

    def test_handle_command_help(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        cont = session.handle_command("/help")
        self.assertTrue(cont)
        output = out_buf.getvalue()
        self.assertIn("/help", output)
        self.assertIn("/models", output)
        self.assertIn("/model", output)
        self.assertIn("/status", output)
        self.assertIn("/clear", output)
        self.assertIn("/exit", output)

    def test_handle_command_models_success(self):
        self.mock_client.list_models.return_value = ["hermes-3", "qwen-2.5"]
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, model="hermes-3", out_stream=out_buf)

        cont = session.handle_command("/models")
        self.assertTrue(cont)
        output = out_buf.getvalue()
        self.assertIn("hermes-3", output)
        self.assertIn("qwen-2.5", output)
        self.assertIn("(active)", output)

    def test_handle_command_models_empty(self):
        self.mock_client.list_models.return_value = []
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        cont = session.handle_command("/models")
        self.assertTrue(cont)
        self.assertIn("No models", out_buf.getvalue())

    def test_handle_command_models_error(self):
        self.mock_client.list_models.side_effect = LocalAIError("Failed to fetch")
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        cont = session.handle_command("/models")
        self.assertTrue(cont)
        self.assertIn("Failed to fetch", out_buf.getvalue())

    def test_handle_command_model_get_and_set(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, model="initial-model", out_stream=out_buf)

        # Get current model
        cont = session.handle_command("/model")
        self.assertTrue(cont)
        self.assertIn("initial-model", out_buf.getvalue())

        # Switch model
        out_buf.truncate(0)
        out_buf.seek(0)
        cont = session.handle_command("/model new-llama-model")
        self.assertTrue(cont)
        self.assertEqual(session.model, "new-llama-model")
        self.assertIn("new-llama-model", out_buf.getvalue())

    def test_handle_command_status(self):
        self.mock_client.health_check.return_value = (True, "Healthy (2 models)", 0.035)
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, model="hermes-3", out_stream=out_buf)

        cont = session.handle_command("/status")
        self.assertTrue(cont)
        output = out_buf.getvalue()
        self.assertIn("http://94.130.18.206:8080", output)
        self.assertIn("hermes-3", output)
        self.assertIn("Healthy", output)
        self.assertIn("0.035", output)

    def test_handle_command_status_error(self):
        self.mock_client.health_check.side_effect = LocalAIConnectionError("Timeout")
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        cont = session.handle_command("/status")
        self.assertTrue(cont)
        self.assertIn("Timeout", out_buf.getvalue())

    def test_handle_command_clear_without_system(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)
        session.messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        cont = session.handle_command("/clear")
        self.assertTrue(cont)
        self.assertEqual(session.messages, [])
        self.assertIn("cleared", out_buf.getvalue().lower())

    def test_handle_command_clear_preserves_system(self):
        out_buf = io.StringIO()
        session = ChatSession(
            client=self.mock_client,
            system="System rule",
            out_stream=out_buf,
        )
        session.messages.extend([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])

        cont = session.handle_command("/clear")
        self.assertTrue(cont)
        self.assertEqual(session.messages, [{"role": "system", "content": "System rule"}])

    def test_handle_command_exit_and_quit(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        self.assertFalse(session.handle_command("/exit"))
        self.assertFalse(session.handle_command("/quit"))

    def test_handle_command_unknown(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        cont = session.handle_command("/unknown_command")
        self.assertTrue(cont)
        self.assertIn("Unknown command", out_buf.getvalue())

    def test_send_message_multi_turn_history(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, model="chat-model", out_stream=out_buf)

        # Turn 1
        self.mock_client.chat_completion.return_value = iter(["Hello", " there!"])
        reply1 = session.send_message("Hi!")
        self.assertEqual(reply1, "Hello there!")
        self.assertEqual(out_buf.getvalue(), "Hello there!\n")
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello there!"},
            ],
        )

        # Turn 2
        out_buf.truncate(0)
        out_buf.seek(0)
        self.mock_client.chat_completion.return_value = iter(["I can ", "help."])
        reply2 = session.send_message("What can you do?")
        self.assertEqual(reply2, "I can help.")
        self.assertEqual(out_buf.getvalue(), "I can help.\n")
        self.assertEqual(len(session.messages), 4)
        self.assertEqual(session.messages[2], {"role": "user", "content": "What can you do?"})
        self.assertEqual(session.messages[3], {"role": "assistant", "content": "I can help."})

        # Ensure full history was passed to chat_completion on 2nd turn
        self.mock_client.chat_completion.assert_called_with(
            messages=session.messages[:3],
            model="chat-model",
            stream=True,
        )

    def test_send_message_error_cleans_up_user_message(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)
        self.mock_client.chat_completion.side_effect = LocalAIConnectionError("Network drop")

        with self.assertRaises(LocalAIConnectionError):
            session.send_message("Failed query")

        # User message should have been popped on failure
        self.assertEqual(session.messages, [])
        self.assertIn("Network drop", out_buf.getvalue())

    def test_send_message_error_with_partial_tokens_keeps_assistant_message(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        def partial_error_stream():
            yield "Beginning"
            raise LocalAIHTTPError(500, "Crash")

        self.mock_client.chat_completion.return_value = partial_error_stream()
        with self.assertRaises(LocalAIHTTPError):
            session.send_message("Explain")

        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0], {"role": "user", "content": "Explain"})
        self.assertEqual(session.messages[1], {"role": "assistant", "content": "Beginning"})
        self.assertIn("Beginning\nError: HTTP 500: Crash\n", out_buf.getvalue())

    def test_send_message_keyboard_interrupt_with_partial_tokens(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        def interrupted_stream():
            yield "Partial answer"
            raise KeyboardInterrupt()

        self.mock_client.chat_completion.return_value = interrupted_stream()
        response = session.send_message("Query")

        self.assertEqual(response, "Partial answer")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[1], {"role": "assistant", "content": "Partial answer"})
        self.assertIn("[Generation interrupted]", out_buf.getvalue())

    def test_send_message_keyboard_interrupt_before_tokens(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        def interrupted_stream():
            raise KeyboardInterrupt()
            yield "Never"

        self.mock_client.chat_completion.return_value = interrupted_stream()
        response = session.send_message("Query")

        self.assertEqual(response, "")
        self.assertEqual(len(session.messages), 0)
        self.assertIn("[Generation interrupted]", out_buf.getvalue())

    def test_handle_command_empty(self):
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)
        self.assertTrue(session.handle_command(""))
        self.assertEqual(out_buf.getvalue(), "")


class TestChatRepl(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.hist_file = os.path.join(self.temp_dir, "history")
        self.hist_patcher = mock.patch("localai_core.chat.DEFAULT_HISTORY_PATH", self.hist_file)
        self.hist_patcher.start()
        self.mock_client = mock.create_autospec(LocalAIClient, instance=True)
        self.mock_client.base_url = "http://94.130.18.206:8080"
        self.mock_config = mock.MagicMock()
        self.mock_config.get_default_model.return_value = "default-llama"
        self.mock_client.config_manager = self.mock_config

    def tearDown(self):
        self.hist_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_repl_exits_on_quit_command(self):
        in_buf = io.StringIO("/quit\n")
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        session.start(in_stream=in_buf)
        output = out_buf.getvalue()
        self.assertIn("Goodbye", output)

    def test_repl_exits_on_eof(self):
        in_buf = io.StringIO("")  # Simulates EOF (Ctrl+D)
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        session.start(in_stream=in_buf)
        output = out_buf.getvalue()
        self.assertIn("Goodbye", output)

    def test_repl_handles_keyboard_interrupt_prompt(self):
        # Simulates a prompt_func that raises KeyboardInterrupt on 1st call, then "/exit" on 2nd
        calls = [0]
        def mock_input(prompt):
            calls[0] += 1
            if calls[0] == 1:
                raise KeyboardInterrupt()
            return "/exit"

        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        with mock.patch("builtins.input", side_effect=mock_input):
            session.start()

        output = out_buf.getvalue()
        self.assertIn("Goodbye", output)
        self.assertEqual(calls[0], 2)

    def test_repl_handles_send_message_exception_gracefully(self):
        self.mock_client.chat_completion.side_effect = LocalAIConnectionError("Refused")
        in_buf = io.StringIO("Ask question\n/exit\n")
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        session.start(in_stream=in_buf)
        output = out_buf.getvalue()
        self.assertIn("Refused", output)
        self.assertIn("Goodbye", output)

    def test_repl_handles_command_then_message_then_exit(self):
        self.mock_client.chat_completion.return_value = iter(["General ", "Kenobi!"])
        in_buf = io.StringIO("/help\nHello there\n/exit\n")
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf)

        session.start(in_stream=in_buf)
        output = out_buf.getvalue()
        self.assertIn("/help", output)
        self.assertIn("General Kenobi!", output)
        self.assertEqual(len(session.messages), 2)

    @mock.patch("localai_core.chat.ChatSession")
    def test_start_chat_repl_instantiates_and_starts(self, mock_session_cls):
        mock_instance = mock.MagicMock()
        mock_session_cls.return_value = mock_instance

        start_chat_repl(client=self.mock_client, model="my-model")

        mock_session_cls.assert_called_once_with(client=self.mock_client, model="my-model")
        mock_instance.start.assert_called_once()

    @mock.patch("readline.read_history_file")
    def test_history_loaded_on_init_if_file_exists(self, mock_read_hist):
        with tempfile.NamedTemporaryFile() as tf:
            session = ChatSession(client=self.mock_client, hist_file=tf.name)
            mock_read_hist.assert_called_once_with(tf.name)

    @mock.patch("readline.read_history_file")
    def test_history_not_loaded_if_file_not_exists(self, mock_read_hist):
        non_existent = "/tmp/non_existent_history_file_12345"
        session = ChatSession(client=self.mock_client, hist_file=non_existent)
        mock_read_hist.assert_not_called()

    @mock.patch("readline.write_history_file")
    def test_history_saved_on_repl_exit(self, mock_write_hist):
        with tempfile.TemporaryDirectory() as td:
            hist_path = os.path.join(td, "history")
            in_buf = io.StringIO("/exit\n")
            out_buf = io.StringIO()
            session = ChatSession(client=self.mock_client, out_stream=out_buf, hist_file=hist_path)
            session.start(in_stream=in_buf)
            mock_write_hist.assert_called_once_with(hist_path)

    def test_save_history_creates_dir_and_file(self):
        with tempfile.TemporaryDirectory() as td:
            hist_path = os.path.join(td, "sub", "history")
            session = ChatSession(client=self.mock_client, hist_file=hist_path)
            session.save_history()
            self.assertTrue(os.path.exists(hist_path))

    @mock.patch("readline.set_history_length")
    def test_history_length_capped_to_1000(self, mock_set_len):
        session = ChatSession(client=self.mock_client, hist_file=self.hist_file)
        mock_set_len.assert_called_with(1000)
        session.save_history()
        self.assertGreaterEqual(mock_set_len.call_count, 2)

    def test_send_message_stream_interrupted_preserves_tokens_and_outputs_error(self):
        def interrupted_stream():
            yield "Partial "
            yield "token"
            raise LocalAIConnectionError("Stream interrupted: Socket read timed out")

        self.mock_client.chat_completion.return_value = interrupted_stream()
        out_buf = io.StringIO()
        session = ChatSession(client=self.mock_client, out_stream=out_buf, hist_file=self.hist_file)

        with self.assertRaises(LocalAIConnectionError):
            session.send_message("What is the weather?")

        out = out_buf.getvalue()
        self.assertIn("Partial token", out)
        self.assertIn("Error: Stream interrupted: Socket read timed out", out)
        self.assertEqual(session.messages[-1], {"role": "assistant", "content": "Partial token"})


if __name__ == "__main__":
    unittest.main()

