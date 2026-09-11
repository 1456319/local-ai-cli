import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from localai_core.client import LocalAIClient
from localai_core.tools import (
    TOOLS,
    TOOL_DEFINITIONS,
    dispatch_tool,
    execute_command,
    read_file,
    write_file,
)
from localai_core.agent import (
    DEFAULT_SYSTEM_PROMPT,
    GoalRunner,
    extract_tool_calls,
)


class TestTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tool_definitions_schema(self):
        self.assertIs(TOOLS, TOOL_DEFINITIONS)
        self.assertEqual(len(TOOLS), 3)

        tool_names = [t["function"]["name"] for t in TOOLS]
        self.assertIn("execute_command", tool_names)
        self.assertIn("read_file", tool_names)
        self.assertIn("write_file", tool_names)

        for t in TOOLS:
            self.assertEqual(t["type"], "function")
            fn = t["function"]
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertEqual(fn["parameters"]["type"], "object")
            self.assertIn("properties", fn["parameters"])
            self.assertIn("required", fn["parameters"])

    def test_execute_command_success(self):
        result = execute_command("echo 'unit test output'")
        self.assertIn("Exit code: 0", result)
        self.assertIn("unit test output", result)

    def test_execute_command_failure_exit_code(self):
        result = execute_command("echo 'err' >&2 && exit 7")
        self.assertIn("Exit code: 7", result)
        self.assertIn("err", result)

    def test_execute_command_timeout(self):
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep 10", timeout=1)):
            result = execute_command("sleep 10", timeout=1)
            self.assertIn("timed out", result.lower())

    def test_execute_command_preserves_deck_context_when_root(self):
        mock_pw = mock.MagicMock()
        mock_pw.pw_uid = 1000
        mock_pw.pw_gid = 1000
        mock_pw.pw_dir = "/home/deck"

        mock_res = mock.MagicMock(returncode=0, stdout="deck\n", stderr="")

        with mock.patch("os.geteuid", return_value=0), \
             mock.patch("pwd.getpwnam", return_value=mock_pw), \
             mock.patch("subprocess.run", return_value=mock_res) as mock_sub:
            res = execute_command("whoami")
            self.assertIn("Exit code: 0", res)
            self.assertIn("deck", res)
            call_kwargs = mock_sub.call_args[1]
            self.assertEqual(call_kwargs.get("user"), 1000)
            self.assertEqual(call_kwargs.get("group"), 1000)
            self.assertEqual(call_kwargs.get("env", {}).get("USER"), "deck")

    def test_read_file_success(self):
        test_file = os.path.join(self.temp_dir, "sample.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Hello file content 🚀")

        content = read_file(test_file)
        self.assertEqual(content, "Hello file content 🚀")

    def test_read_file_not_found(self):
        missing_file = os.path.join(self.temp_dir, "nonexistent.txt")
        result = read_file(missing_file)
        self.assertIn("Error:", result)
        self.assertIn("not found", result.lower())

    def test_read_file_directory_error(self):
        result = read_file(self.temp_dir)
        self.assertIn("Error:", result)
        self.assertIn("directory", result.lower())

    def test_write_file_creates_file_and_parent_dirs(self):
        nested_file = os.path.join(self.temp_dir, "nested", "dir", "test.txt")
        result = write_file(nested_file, "Line 1\nLine 2")

        self.assertIn("Successfully wrote", result)
        self.assertTrue(os.path.exists(nested_file))
        with open(nested_file, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Line 1\nLine 2")

    def test_write_file_sets_deck_ownership_when_posix(self):
        test_file = os.path.join(self.temp_dir, "owned.txt")
        mock_pw = mock.MagicMock(pw_uid=1000)
        mock_grp = mock.MagicMock(gr_gid=1000)

        with mock.patch("pwd.getpwnam", return_value=mock_pw), \
             mock.patch("grp.getgrnam", return_value=mock_grp), \
             mock.patch("os.chown") as mock_chown:
            result = write_file(test_file, "owned content")
            self.assertIn("Successfully wrote", result)
            self.assertTrue(mock_chown.called)
            args, _ = mock_chown.call_args
            self.assertEqual(args[1], 1000)
            self.assertEqual(args[2], 1000)

    def test_write_file_recursive_parent_ownership(self):
        nested_file = os.path.join(self.temp_dir, "deep", "nested", "path", "file.txt")
        mock_pw = mock.MagicMock(pw_uid=1000)
        mock_grp = mock.MagicMock(gr_gid=1000)

        with mock.patch("pwd.getpwnam", return_value=mock_pw), \
             mock.patch("grp.getgrnam", return_value=mock_grp), \
             mock.patch("os.chown") as mock_chown:
            result = write_file(nested_file, "deep content")
            self.assertIn("Successfully wrote", result)
            # Expect chown called for file + deep/nested/path + deep/nested + deep (4 times)
            self.assertGreaterEqual(mock_chown.call_count, 4)

    def test_write_file_atomic_replacement(self):
        target = os.path.join(self.temp_dir, "atomic.txt")
        write_file(target, "initial content")
        with mock.patch("os.replace", wraps=os.replace) as mock_replace:
            result = write_file(target, "updated content")
            self.assertIn("Successfully wrote", result)
            mock_replace.assert_called_once()
            self.assertEqual(mock_replace.call_args[0][1], target)
        with open(target, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "updated content")

    def test_write_file_default_permissions_0644(self):
        target = os.path.join(self.temp_dir, "perm_default.txt")
        write_file(target, "content")
        st = os.stat(target)
        self.assertEqual(st.st_mode & 0o777, 0o644)

    def test_write_file_preserves_target_permissions(self):
        target = os.path.join(self.temp_dir, "perm_existing.txt")
        with open(target, "w") as f:
            f.write("orig")
        os.chmod(target, 0o755)

        write_file(target, "new content")
        st = os.stat(target)
        self.assertEqual(st.st_mode & 0o777, 0o755)

    def test_write_file_cleans_up_temp_file_on_error(self):
        target = os.path.join(self.temp_dir, "fail.txt")
        with mock.patch("os.replace", side_effect=OSError("Disk full")):
            result = write_file(target, "content")
            self.assertIn("Error writing to file", result)
            self.assertIn("Disk full", result)

        remaining = [f for f in os.listdir(self.temp_dir) if f.startswith(".tmp_write_")]
        self.assertEqual(remaining, [])

    def test_dispatch_tool_execute_command(self):
        result = dispatch_tool("execute_command", {"command": "echo 'dispatch_exec'"})
        self.assertIn("dispatch_exec", result)

    def test_dispatch_tool_write_and_read_file(self):
        target = os.path.join(self.temp_dir, "dispatch.txt")
        w_res = dispatch_tool("write_file", {"path": target, "content": "roundtrip data"})
        self.assertIn("Successfully wrote", w_res)

        r_res = dispatch_tool("read_file", {"path": target})
        self.assertEqual(r_res, "roundtrip data")

    def test_dispatch_tool_unknown(self):
        result = dispatch_tool("unknown_tool", {})
        self.assertIn("Error: Unknown tool 'unknown_tool'", result)

    def test_dispatch_tool_missing_args(self):
        result = dispatch_tool("execute_command", {})
        self.assertIn("Error: Missing required argument 'command'", result)

        result = dispatch_tool("read_file", {})
        self.assertIn("Error: Missing required argument 'path'", result)

        result = dispatch_tool("write_file", {"path": "foo"})
        self.assertIn("Error: Missing required argument 'content'", result)

    def test_dispatch_tool_non_dict_arguments(self):
        result = dispatch_tool("read_file", "invalid")
        self.assertIn("Error: Arguments must be a dictionary", result)


class TestExtractToolCalls(unittest.TestCase):
    def test_extract_openai_standard_tool_calls_json_string(self):
        msg = {
            "role": "assistant",
            "content": "Running command",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "execute_command",
                        "arguments": json.dumps({"command": "ls -l"}),
                    },
                }
            ],
        }
        calls = extract_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_123")
        self.assertEqual(calls[0]["name"], "execute_command")
        self.assertEqual(calls[0]["arguments"], {"command": "ls -l"})

    def test_extract_openai_standard_tool_calls_dict_args(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "/etc/hosts"},
                    },
                }
            ],
        }
        calls = extract_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_456")
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[0]["arguments"], {"path": "/etc/hosts"})

    def test_extract_fallback_tag_single(self):
        msg = {
            "role": "assistant",
            "content": 'I need to write a file.\n<tool_call>\n{"name": "write_file", "arguments": {"path": "foo.txt", "content": "hello"}}\n</tool_call>',
        }
        calls = extract_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertEqual(calls[0]["arguments"], {"path": "foo.txt", "content": "hello"})

    def test_extract_fallback_tag_multiple(self):
        msg = {
            "role": "assistant",
            "content": (
                '<tool_call>{"name": "execute_command", "arguments": {"command": "echo 1"}}</tool_call>\n'
                '<tool_call>{"name": "execute_command", "arguments": {"command": "echo 2"}}</tool_call>'
            ),
        }
        calls = extract_tool_calls(msg)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["arguments"]["command"], "echo 1")
        self.assertEqual(calls[1]["arguments"]["command"], "echo 2")

    def test_extract_fallback_markdown_block(self):
        msg = {
            "role": "assistant",
            "content": (
                "<tool_call>\n"
                "```json\n"
                '{"name": "read_file", "arguments": {"path": "/var/log"}}\n'
                "```\n"
                "</tool_call>"
            ),
        }
        calls = extract_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[0]["arguments"], {"path": "/var/log"})

    def test_extract_fallback_nested_function_dict(self):
        msg = {
            "role": "assistant",
            "content": '<tool_call>{"function": {"name": "execute_command", "arguments": {"command": "pwd"}}}</tool_call>',
        }
        calls = extract_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "execute_command")
        self.assertEqual(calls[0]["arguments"], {"command": "pwd"})

    def test_extract_empty_when_no_tool_calls(self):
        msg = {"role": "assistant", "content": "Just a normal answer."}
        calls = extract_tool_calls(msg)
        self.assertEqual(calls, [])


class TestGoalRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_client = mock.create_autospec(LocalAIClient, instance=True)
        self.mock_config = mock.MagicMock()
        self.mock_config.get_default_model.return_value = "default-llama"
        self.mock_client.config_manager = self.mock_config
        self.out_buf = io.StringIO()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_immediate_goal_complete(self):
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Goal already achieved! <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        }
        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal("Check something")

        self.assertTrue(success)
        self.assertIn("Goal Complete", self.out_buf.getvalue())
        self.assertEqual(self.mock_client.chat_completion.call_count, 1)

    def test_standard_tool_calling_loop_to_completion(self):
        target_path = os.path.join(self.temp_dir, "goal_output.txt")

        turn1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I will write the requested file.",
                        "tool_calls": [
                            {
                                "id": "call_w1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps({
                                        "path": target_path,
                                        "content": "Autonomous success",
                                    }),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        turn2_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Verified file created. <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        }
        self.mock_client.chat_completion.side_effect = [turn1_resp, turn2_resp]

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal(
            "Write Autonomous success to file",
            autonomous=True,
        )

        self.assertTrue(success)
        self.assertTrue(os.path.exists(target_path))
        with open(target_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Autonomous success")

        self.assertEqual(self.mock_client.chat_completion.call_count, 2)
        # Verify second call passed tool result in messages
        call2_messages = self.mock_client.chat_completion.call_args_list[1][1]["messages"]
        tool_msg = [m for m in call2_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msg), 1)
        self.assertEqual(tool_msg[0]["tool_call_id"], "call_w1")
        self.assertIn("Successfully wrote", tool_msg[0]["content"])

    def test_fallback_tag_tool_calling_loop(self):
        turn1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Executing echo:\n"
                            '<tool_call>{"name": "execute_command", "arguments": {"command": "echo fallback_ok"}}</tool_call>'
                        ),
                    }
                }
            ]
        }
        turn2_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Command executed. <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        }
        self.mock_client.chat_completion.side_effect = [turn1_resp, turn2_resp]

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal("Echo fallback", autonomous=True)

        self.assertTrue(success)
        self.assertEqual(self.mock_client.chat_completion.call_count, 2)
        self.assertIn("fallback_ok", self.out_buf.getvalue())

        # Verify that turn 2 received an assistant message with synthesized tool_calls
        call2_messages = self.mock_client.chat_completion.call_args_list[1][1]["messages"]
        assistant_msgs = [m for m in call2_messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertIn("tool_calls", assistant_msgs[0])
        syn_calls = assistant_msgs[0]["tool_calls"]
        self.assertEqual(len(syn_calls), 1)
        self.assertEqual(syn_calls[0]["function"]["name"], "execute_command")
        self.assertEqual(
            json.loads(syn_calls[0]["function"]["arguments"]),
            {"command": "echo fallback_ok"},
        )

        # Verify matching tool message was also appended
        tool_msgs = [m for m in call2_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], syn_calls[0]["id"])
        self.assertIn("fallback_ok", tool_msgs[0]["content"])

    def test_max_turns_exceeded_returns_false(self):
        turn_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Working...",
                        "tool_calls": [
                            {
                                "id": "call_loop",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": {"command": "echo looping"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        # Infinite loop without GOAL_COMPLETE
        self.mock_client.chat_completion.return_value = turn_resp

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal("Infinite task", max_turns=3, autonomous=True)

        self.assertFalse(success)
        self.assertEqual(self.mock_client.chat_completion.call_count, 3)
        self.assertIn("turn limit", self.out_buf.getvalue().lower())

    def test_max_turns_zero_allows_many_turns_until_completion(self):
        # 4 turns of tool calling, then 5th turn completion
        responses = []
        for i in range(4):
            responses.append({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": f"call_{i}",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_command",
                                        "arguments": {"command": f"echo step_{i}"},
                                    },
                                }
                            ],
                        }
                    }
                ]
            })
        responses.append({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done with all 4 steps! <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        })

        self.mock_client.chat_completion.side_effect = responses

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        # max_turns=0 represents unlimited turns
        success = runner.run_goal("Multi-step task", max_turns=0, autonomous=True)

        self.assertTrue(success)
        self.assertEqual(self.mock_client.chat_completion.call_count, 5)

    def test_autonomous_mode_false_confirmed_via_callback(self):
        confirm_cb = mock.MagicMock(return_value=True)
        turn1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_cmd",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": {"command": "echo step"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        turn2_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Finished! <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        }
        self.mock_client.chat_completion.side_effect = [turn1_resp, turn2_resp]

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal("Confirm me", autonomous=False, confirm_cb=confirm_cb)

        self.assertTrue(success)
        confirm_cb.assert_called_once_with("execute_command", {"command": "echo step"})

    def test_autonomous_mode_false_rejected_via_callback(self):
        confirm_cb = mock.MagicMock(return_value=False)
        turn1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_rejected",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": {"command": "rm -rf /"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        turn2_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Understood, skipping dangerous step. <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        }
        self.mock_client.chat_completion.side_effect = [turn1_resp, turn2_resp]

        with mock.patch("localai_core.agent.dispatch_tool") as mock_dispatch:
            runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
            success = runner.run_goal("Dangerous goal", autonomous=False, confirm_cb=confirm_cb)

            self.assertTrue(success)
            mock_dispatch.assert_not_called()

        # Check that rejection message was sent back as tool result
        call2_messages = self.mock_client.chat_completion.call_args_list[1][1]["messages"]
        tool_msg = [m for m in call2_messages if m.get("role") == "tool"][0]
        self.assertIn("rejected by user", tool_msg["content"].lower())

    def test_autonomous_mode_false_prompt_stream_interactive(self):
        in_buf = io.StringIO("y\n")
        turn1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_in",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": {"command": "echo interactive"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        turn2_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        }
        self.mock_client.chat_completion.side_effect = [turn1_resp, turn2_resp]

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf, in_stream=in_buf)
        success = runner.run_goal("Interactive goal", autonomous=False)

        self.assertTrue(success)
        self.assertIn("[y/N]", self.out_buf.getvalue())
        self.assertIn("interactive", self.out_buf.getvalue())

    def test_keyboard_interrupt_during_completion(self):
        self.mock_client.chat_completion.side_effect = KeyboardInterrupt()

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal("Interrupted goal")

        self.assertFalse(success)
        self.assertIn("interrupted", self.out_buf.getvalue().lower())

    def test_keyboard_interrupt_during_tool_confirmation(self):
        def raise_interrupt(name, args):
            raise KeyboardInterrupt()

        turn1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_int",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": {"command": "ls"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        self.mock_client.chat_completion.return_value = turn1_resp

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal("Interrupt in prompt", autonomous=False, confirm_cb=raise_interrupt)

        self.assertFalse(success)
        self.assertIn("interrupted", self.out_buf.getvalue().lower())

    def test_text_only_response_triggers_continuation_nudge(self):
        turn1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I am thinking about the best approach...",
                    }
                }
            ]
        }
        turn2_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "All done now! <!-- GOAL_COMPLETE -->",
                    }
                }
            ]
        }
        self.mock_client.chat_completion.side_effect = [turn1_resp, turn2_resp]

        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        success = runner.run_goal("Think and do", autonomous=True)

        self.assertTrue(success)
        self.assertEqual(self.mock_client.chat_completion.call_count, 2)
        # Check that a continuation user message was sent in turn 2
        call2_messages = self.mock_client.chat_completion.call_args_list[1][1]["messages"]
        self.assertEqual(call2_messages[-1]["role"], "user")
        self.assertIn("continue", call2_messages[-1]["content"].lower())

    def test_model_override_parameter_passed(self):
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": "Done <!-- GOAL_COMPLETE -->"}}
            ]
        }
        runner = GoalRunner(client=self.mock_client, out_stream=self.out_buf)
        runner.run_goal("Test model override", model="custom-super-model")

        self.mock_client.chat_completion.assert_called_once()
        self.assertEqual(self.mock_client.chat_completion.call_args[1]["model"], "custom-super-model")

    def test_extract_tool_calls_raw_json_tool_call(self):
        msg = {
            "role": "assistant",
            "content": '{\n  "tool_call": {\n    "name": "write_file",\n    "arguments": {"path": "/tmp/test.txt", "content": "hi"}\n  }\n}',
        }
        calls = extract_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertEqual(calls[0]["arguments"], {"path": "/tmp/test.txt", "content": "hi"})


if __name__ == "__main__":
    unittest.main()

