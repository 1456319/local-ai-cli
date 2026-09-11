"""Unit tests for LocalAIClient."""

import unittest
from unittest import mock
import requests

from localai_core.config import ConfigManager, DEFAULT_URL, DEFAULT_MODEL
from localai_core.client import (
    LocalAIClient,
    LocalAIError,
    LocalAIConnectionError,
    LocalAIHTTPError,
)


class TestLocalAIClient(unittest.TestCase):
    def setUp(self):
        self.mock_config = mock.MagicMock(spec=ConfigManager)
        self.mock_config.get_url.return_value = DEFAULT_URL
        self.mock_config.get_default_model.return_value = DEFAULT_MODEL
        self.mock_config.get_timeout.return_value = 60
        self.mock_config.get_max_retries.return_value = 3

    def test_init_defaults(self):
        client = LocalAIClient(config_manager=self.mock_config)
        self.assertEqual(client.base_url, DEFAULT_URL)
        self.assertEqual(client.timeout, 60)
        self.assertEqual(client.max_retries, 3)

    def test_init_custom_overrides(self):
        client = LocalAIClient(
            config_manager=self.mock_config,
            base_url="http://custom-host:9000/",
            timeout=30,
            max_retries=5,
            backoff_factor=0.5,
        )
        self.assertEqual(client.base_url, "http://custom-host:9000")
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.backoff_factor, 0.5)

    def test_list_models_parses_model_ids(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "data": [
                {"id": "Hermes-3-Llama-3.2-3B-Q4_K_M.gguf", "object": "model"},
                {"id": "qwen2.5-coder-7b-instruct", "object": "model"},
            ],
        }

        with mock.patch.object(client.session, "request", return_value=mock_resp) as mock_req:
            models = client.list_models()
            self.assertEqual(
                models,
                ["Hermes-3-Llama-3.2-3B-Q4_K_M.gguf", "qwen2.5-coder-7b-instruct"],
            )
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            self.assertEqual(call_args[0][0], "GET")
            self.assertTrue(call_args[0][1].endswith("/v1/models"))

    def test_list_models_empty(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            models = client.list_models()
            self.assertEqual(models, [])

    def test_list_models_null_data(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": None}

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            models = client.list_models()
            self.assertEqual(models, [])


    def test_chat_completion_non_streaming_returns_dict(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        expected_response = {
            "id": "chatcmpl-test-123",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        mock_resp.json.return_value = expected_response

        messages = [{"role": "user", "content": "Hi"}]
        with mock.patch.object(client.session, "request", return_value=mock_resp) as mock_req:
            res = client.chat_completion(messages=messages, stream=False)
            self.assertEqual(res, expected_response)
            mock_req.assert_called_once()
            call_args, call_kwargs = mock_req.call_args
            self.assertEqual(call_args[0], "POST")
            self.assertTrue(call_args[1].endswith("/v1/chat/completions"))
            sent_json = call_kwargs.get("json", {})
            self.assertEqual(sent_json["model"], DEFAULT_MODEL)
            self.assertEqual(sent_json["messages"], messages)
            self.assertFalse(sent_json["stream"])
            self.assertEqual(sent_json["temperature"], 0.7)

    def test_chat_completion_with_tools_and_custom_options(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "execute_command", "arguments": '{"command":"ls"}'},
                            }
                        ],
                    }
                }
            ]
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Run shell command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ]

        with mock.patch.object(client.session, "request", return_value=mock_resp) as mock_req:
            res = client.chat_completion(
                messages=[{"role": "user", "content": "list directory"}],
                model="custom-model",
                tools=tools,
                temperature=0.2,
                max_tokens=256,
            )
            self.assertIn("choices", res)
            call_args, call_kwargs = mock_req.call_args
            sent_json = call_kwargs["json"]
            self.assertEqual(sent_json["model"], "custom-model")
            self.assertEqual(sent_json["tools"], tools)
            self.assertEqual(sent_json["temperature"], 0.2)
            self.assertEqual(sent_json["max_tokens"], 256)

    def test_chat_completion_streaming_yields_tokens_from_sse(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            ': keepalive comment',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
            'data: {"choices":[{"delta":{"content":"should not appear"}}]}',
        ]
        mock_resp.iter_lines.return_value = sse_lines

        with mock.patch.object(client.session, "request", return_value=mock_resp) as mock_req:
            token_gen = client.chat_completion(
                messages=[{"role": "user", "content": "Greet me"}],
                stream=True,
            )
            tokens = list(token_gen)
            self.assertEqual(tokens, ["Hello", " world", "!"])
            mock_req.assert_called_once()
            call_kwargs = mock_req.call_args[1]
            self.assertTrue(call_kwargs.get("stream"))
            self.assertTrue(call_kwargs["json"].get("stream"))
            mock_resp.close.assert_called_once()

    def test_chat_completion_streaming_handles_bytes_and_empty_delta(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        sse_lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            b'data: {"choices":[{"delta":{"content":"Streaming"}}]}',
            b'data: {"choices":[{"delta":{"content":null}}]}',
            b'data: [DONE]',
        ]
        mock_resp.iter_lines.return_value = sse_lines

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            tokens = list(client.chat_completion([{"role": "user", "content": "test"}], stream=True))
            self.assertEqual(tokens, ["Streaming"])

    def test_chat_completion_streaming_terminates_on_finish_reason_without_done(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        # Simulated generator where lines after finish_reason should NOT be read
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Line 1"}}]}',
            'data: {"choices":[{"delta":{"content":" Line 2"},"finish_reason":"stop"}]}',
            'data: {"choices":[{"delta":{"content":" Should NOT appear"}}]}',
        ]
        mock_resp.iter_lines.return_value = sse_lines

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            tokens = list(client.chat_completion([{"role": "user", "content": "test"}], stream=True))
            self.assertEqual(tokens, ["Line 1", " Line 2"])

    def test_chat_completion_streaming_terminates_on_separate_finish_reason_chunk(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Part 1"}}]}',
            'data: {"choices":[{"delta":{"content":null},"finish_reason":"stop"}]}',
            'data: {"choices":[{"delta":{"content":" Should NOT appear"}}]}',
        ]
        mock_resp.iter_lines.return_value = sse_lines

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            tokens = list(client.chat_completion([{"role": "user", "content": "test"}], stream=True))
            self.assertEqual(tokens, ["Part 1"])

    def test_chat_completion_streaming_recovers_gracefully_on_timeout_after_tokens(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200

        def stream_with_timeout():
            yield 'data: {"choices":[{"delta":{"content":"Generated token"}}]}'
            raise requests.exceptions.ReadTimeout("Socket read timed out waiting for more data")

        mock_resp.iter_lines.return_value = stream_with_timeout()

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            tokens = list(client.chat_completion([{"role": "user", "content": "test"}], stream=True))
            self.assertEqual(tokens, ["Generated token"])

    def test_chat_completion_streaming_handles_flexible_done_tokens(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Token"}}]}',
            'data:[DONE]',
            'data: {"choices":[{"delta":{"content":" Extra"}}]}',
        ]
        mock_resp.iter_lines.return_value = sse_lines

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            tokens = list(client.chat_completion([{"role": "user", "content": "test"}], stream=True))
            self.assertEqual(tokens, ["Token"])

    def test_chat_completion_streaming_choice_text_fallback(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=0)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        sse_lines = [
            'data: {"choices":[{"text":"Hello from text field"}]}',
            'data: [DONE]',
        ]
        mock_resp.iter_lines.return_value = sse_lines

        with mock.patch.object(client.session, "request", return_value=mock_resp):
            tokens = list(client.chat_completion([{"role": "user", "content": "test"}], stream=True))
            self.assertEqual(tokens, ["Hello from text field"])

    def test_health_check_healthy(self):
        client = LocalAIClient(config_manager=self.mock_config)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "m1"}, {"id": "m2"}]}

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            is_healthy, message, latency = client.health_check()
            self.assertTrue(is_healthy)
            self.assertIn("2 models", message)
            self.assertIsInstance(latency, float)
            self.assertGreaterEqual(latency, 0.0)

    def test_health_check_http_error(self):
        client = LocalAIClient(config_manager=self.mock_config)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 502
        mock_resp.reason = "Bad Gateway"

        with mock.patch.object(client.session, "get", return_value=mock_resp):
            is_healthy, message, latency = client.health_check()
            self.assertFalse(is_healthy)
            self.assertIn("502", message)
            self.assertIsInstance(latency, float)

    def test_health_check_connection_error(self):
        client = LocalAIClient(config_manager=self.mock_config)

        with mock.patch.object(client.session, "get", side_effect=requests.exceptions.ConnectionError("Refused")):
            is_healthy, message, latency = client.health_check()
            self.assertFalse(is_healthy)
            self.assertIn("Connection failed", message)
            self.assertIsInstance(latency, float)

    def test_health_check_passes_headers_and_auth(self):
        client = LocalAIClient(config_manager=self.mock_config)
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "m1"}]}

        with mock.patch.dict("os.environ", {"LOCALAI_API_KEY": "secret-token"}):
            with mock.patch.object(client.session, "get", return_value=mock_resp) as mock_get:
                is_healthy, message, latency = client.health_check(timeout=3.0)
                self.assertTrue(is_healthy)
                mock_get.assert_called_once()
                call_kwargs = mock_get.call_args[1]
                self.assertEqual(call_kwargs.get("timeout"), 3.0)
                headers = call_kwargs.get("headers", {})
                self.assertEqual(headers.get("Authorization"), "Bearer secret-token")
                self.assertEqual(headers.get("Content-Type"), "application/json")


    @mock.patch("time.sleep")
    def test_retry_on_connection_error_then_success(self, mock_sleep):
        client = LocalAIClient(config_manager=self.mock_config, max_retries=3, backoff_factor=1.0)
        success_resp = mock.MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"data": [{"id": "model-1"}]}

        # Fails twice with ConnectionError, succeeds on 3rd attempt
        side_effects = [
            requests.exceptions.ConnectionError("Drop 1"),
            requests.exceptions.ConnectionError("Drop 2"),
            success_resp,
        ]

        with mock.patch.object(client.session, "request", side_effect=side_effects) as mock_req:
            models = client.list_models()
            self.assertEqual(models, ["model-1"])
            self.assertEqual(mock_req.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            # Verify exponential backoff delays with [0, 0.25] jitter:
            # 1.0 <= delay0 <= 1.25, 2.0 <= delay1 <= 2.25
            d1 = mock_sleep.call_args_list[0][0][0]
            d2 = mock_sleep.call_args_list[1][0][0]
            self.assertTrue(1.0 <= d1 <= 1.25, f"Expected d1 in [1.0, 1.25], got {d1}")
            self.assertTrue(2.0 <= d2 <= 2.25, f"Expected d2 in [2.0, 2.25], got {d2}")

    @mock.patch("time.sleep")
    def test_retry_on_5xx_server_error_then_success(self, mock_sleep):
        client = LocalAIClient(config_manager=self.mock_config, max_retries=2, backoff_factor=2.0)
        err_resp = mock.MagicMock()
        err_resp.status_code = 503
        err_resp.text = "Service Unavailable"

        success_resp = mock.MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        side_effects = [err_resp, success_resp]

        with mock.patch.object(client.session, "request", side_effect=side_effects) as mock_req:
            res = client.chat_completion([{"role": "user", "content": "hi"}], stream=False)
            self.assertEqual(res["choices"][0]["message"]["content"], "ok")
            self.assertEqual(mock_req.call_count, 2)
            self.assertEqual(mock_sleep.call_count, 1)
            d = mock_sleep.call_args[0][0]
            self.assertTrue(2.0 <= d <= 2.25, f"Expected d in [2.0, 2.25], got {d}")

    def test_calculate_backoff_includes_jitter(self):
        client = LocalAIClient(config_manager=self.mock_config, backoff_factor=1.0)
        delays = [client._calculate_backoff(1) for _ in range(50)]
        for d in delays:
            self.assertGreaterEqual(d, 2.0)
            self.assertLessEqual(d, 2.25)
        # Verify random variation
        self.assertGreater(len(set(delays)), 1)

    @mock.patch("time.sleep")
    def test_retry_exhausted_raises_connection_error(self, mock_sleep):
        client = LocalAIClient(config_manager=self.mock_config, max_retries=2, backoff_factor=1.0)
        with mock.patch.object(
            client.session,
            "request",
            side_effect=requests.exceptions.ConnectionError("Host unreachable"),
        ) as mock_req:
            with self.assertRaises(LocalAIConnectionError) as cm:
                client.list_models()
            self.assertEqual(mock_req.call_count, 3)  # Initial + 2 retries
            self.assertIn("local-ai config set-url", str(cm.exception))

    @mock.patch("time.sleep")
    def test_retry_exhausted_on_5xx_raises_http_error(self, mock_sleep):
        client = LocalAIClient(config_manager=self.mock_config, max_retries=2, backoff_factor=1.0)
        err_resp = mock.MagicMock()
        err_resp.status_code = 500
        err_resp.text = "Internal Server Error"

        with mock.patch.object(client.session, "request", return_value=err_resp) as mock_req:
            with self.assertRaises(LocalAIHTTPError) as cm:
                client.list_models()
            self.assertEqual(mock_req.call_count, 3)
            self.assertEqual(cm.exception.status_code, 500)

    def test_no_retry_on_4xx_client_error(self):
        client = LocalAIClient(config_manager=self.mock_config, max_retries=3, backoff_factor=1.0)
        err_resp = mock.MagicMock()
        err_resp.status_code = 404
        err_resp.text = "Model Not Found"

        with mock.patch.object(client.session, "request", return_value=err_resp) as mock_req:
            with self.assertRaises(LocalAIHTTPError) as cm:
                client.chat_completion([{"role": "user", "content": "hi"}], stream=False)
            self.assertEqual(mock_req.call_count, 1)  # No retries on 4xx
            self.assertEqual(cm.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
