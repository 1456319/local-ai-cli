import unittest
import os
import json
import tempfile
from localai_core.config import ConfigManager, DEFAULT_URL, DEFAULT_MODEL

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
        config = self.manager.load_config()
        self.assertEqual(config.get("default_model"), "Hermes-3-Llama-3.2-3B-Q4_K_M.gguf")

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

    def test_openai_base_url_fallback(self):
        os.environ["OPENAI_BASE_URL"] = "http://openai-base:8080"
        try:
            url = self.manager.get_url()
            self.assertEqual(url, "http://openai-base:8080")
        finally:
            del os.environ["OPENAI_BASE_URL"]

    def test_set_url(self):
        self.manager.set_url("http://dynamic-ip:9000")
        self.assertEqual(self.manager.get_url(), "http://dynamic-ip:9000")
        with open(self.config_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data.get("url"), "http://dynamic-ip:9000")

    def test_get_and_set_default_model(self):
        self.assertEqual(self.manager.get_default_model(), "Hermes-3-Llama-3.2-3B-Q4_K_M.gguf")
        self.manager.set_default_model("qwen-2.5-7b")
        self.assertEqual(self.manager.get_default_model(), "qwen-2.5-7b")
        self.assertEqual(self.manager.get_default_model(cli_override="hermes-override"), "hermes-override")

    def test_model_env_var_override(self):
        os.environ["LOCALAI_MODEL"] = "env-model-llama"
        try:
            self.assertEqual(self.manager.get_default_model(), "env-model-llama")
        finally:
            del os.environ["LOCALAI_MODEL"]

    def test_save_and_load_config_preserves_custom_keys(self):
        self.manager.save_config({"url": "http://custom:1234", "custom_key": "custom_value"})
        config = self.manager.load_config()
        self.assertEqual(config["url"], "http://custom:1234")
        self.assertEqual(config["custom_key"], "custom_value")

    def test_config_path_expanduser(self):
        mgr = ConfigManager(config_path="~/test_config_dir/cfg.json")
        expected = os.path.abspath(os.path.expanduser("~/test_config_dir/cfg.json"))
        self.assertEqual(mgr.config_path, expected)
        self.assertFalse(mgr.config_path.startswith("~"))

    def test_env_var_expanduser(self):
        os.environ["LOCALAI_CONFIG_PATH"] = "~/test_env_cfg.json"
        try:
            mgr = ConfigManager()
            expected = os.path.abspath(os.path.expanduser("~/test_env_cfg.json"))
            self.assertEqual(mgr.config_path, expected)
            self.assertFalse(mgr.config_path.startswith("~"))
        finally:
            del os.environ["LOCALAI_CONFIG_PATH"]

    def test_save_config_file_permissions_0644(self):
        self.manager.save_config({"url": "http://permissions-check:8080"})
        st = os.stat(self.config_file)
        self.assertEqual(st.st_mode & 0o777, 0o644)

    def test_set_timeout_persists(self):
        self.manager.set_timeout(180)
        self.assertEqual(self.manager.get_timeout(), 180)
        reloaded = ConfigManager(config_path=self.config_file)
        self.assertEqual(reloaded.get_timeout(), 180)


if __name__ == '__main__':
    unittest.main()
