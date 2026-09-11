"""Configuration manager for local-ai remote client.

Handles resolution priority: CLI flags > Environment variables > Configuration file > Defaults.
Supports atomic persistence of settings to JSON configuration files.
"""

import os
import json
import pwd
import grp
import tempfile
from typing import Optional, Dict, Any

DEFAULT_URL: str = "http://94.130.18.206:8080"
DEFAULT_MODEL: str = "Hermes-3-Llama-3.2-3B-Q4_K_M.gguf"
DEFAULT_CONFIG_DIR: str = "/home/deck/.config/local-ai"
DEFAULT_CONFIG_PATH: str = os.path.join(DEFAULT_CONFIG_DIR, "config.json")
DEFAULT_TIMEOUT: int = 60
DEFAULT_MAX_RETRIES: int = 3

DEFAULT_CONFIG: Dict[str, Any] = {
    "url": DEFAULT_URL,
    "default_model": DEFAULT_MODEL,
    "timeout": DEFAULT_TIMEOUT,
    "max_retries": DEFAULT_MAX_RETRIES,
}


class ConfigManager:
    """Manages reading, writing, and hierarchical resolution of local-ai configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ConfigManager with an optional custom configuration file path."""
        if config_path:
            self.config_path = os.path.abspath(os.path.expanduser(config_path))
        else:
            env_path = os.environ.get("LOCALAI_CONFIG_PATH")
            if env_path:
                self.config_path = os.path.abspath(os.path.expanduser(env_path))
            else:
                self.config_path = os.path.abspath(os.path.expanduser(DEFAULT_CONFIG_PATH))

    @staticmethod
    def _ensure_deck_ownership(path: str) -> None:
        """Ensure file or directory is owned by user and group 'deck' if running as root."""
        try:
            if os.geteuid() == 0:
                try:
                    deck_pwd = pwd.getpwnam("deck")
                    uid = deck_pwd.pw_uid
                except KeyError:
                    uid = 1000

                try:
                    deck_grp = grp.getgrnam("deck")
                    gid = deck_grp.gr_gid
                except KeyError:
                    gid = 1000

                os.chown(path, uid, gid)
        except Exception:
            pass

    def load_config(self) -> Dict[str, Any]:
        """Load configuration dictionary from disk.

        If the configuration file does not exist, creates it with default settings atomically.
        """
        if not os.path.exists(self.config_path):
            config = dict(DEFAULT_CONFIG)
            self.save_config(config)
            return config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return dict(DEFAULT_CONFIG)
        except Exception:
            return dict(DEFAULT_CONFIG)

    def save_config(self, data: Dict[str, Any]) -> None:
        """Atomically persist configuration data to disk."""
        dir_name = os.path.dirname(os.path.abspath(self.config_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            self._ensure_deck_ownership(dir_name)

        # Write to a temporary file in the same directory, then rename atomically
        temp_fd, temp_path = tempfile.mkstemp(
            dir=dir_name,
            prefix="config_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

            try:
                os.chmod(temp_path, 0o644)
            except OSError:
                pass

            self._ensure_deck_ownership(temp_path)
            os.replace(temp_path, self.config_path)
            self._ensure_deck_ownership(self.config_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def get_url(self, cli_override: Optional[str] = None) -> str:
        """Resolve the remote LocalAI URL with priority resolution:

        1. CLI override argument (--url / -u)
        2. Environment variable: LOCALAI_URL or OPENAI_BASE_URL
        3. Config file setting 'url'
        4. Default fallback: DEFAULT_URL
        """
        if cli_override is not None and str(cli_override).strip():
            return str(cli_override).strip().rstrip("/")

        env_url = os.environ.get("LOCALAI_URL") or os.environ.get("OPENAI_BASE_URL")
        if env_url is not None and str(env_url).strip():
            return str(env_url).strip().rstrip("/")

        config = self.load_config()
        url = config.get("url")
        if url and str(url).strip():
            return str(url).strip().rstrip("/")

        return DEFAULT_URL

    def get_default_model(self, cli_override: Optional[str] = None) -> str:
        """Resolve the default model name with priority resolution:

        1. CLI override argument (--model / -m)
        2. Environment variable: LOCALAI_MODEL or OPENAI_MODEL
        3. Config file setting 'default_model'
        4. Default fallback: DEFAULT_MODEL
        """
        if cli_override is not None and str(cli_override).strip():
            return str(cli_override).strip()

        env_model = os.environ.get("LOCALAI_MODEL") or os.environ.get("OPENAI_MODEL")
        if env_model is not None and str(env_model).strip():
            return str(env_model).strip()

        config = self.load_config()
        model = config.get("default_model")
        if model and str(model).strip():
            return str(model).strip()

        return DEFAULT_MODEL

    def set_url(self, new_url: str) -> None:
        """Update and persist the target remote URL."""
        config = self.load_config()
        config["url"] = new_url.strip().rstrip("/")
        self.save_config(config)

    def set_default_model(self, new_model: str) -> None:
        """Update and persist the default model name."""
        config = self.load_config()
        config["default_model"] = new_model.strip()
        self.save_config(config)

    def get_timeout(self) -> int:
        """Get configured request timeout in seconds."""
        config = self.load_config()
        try:
            return int(config.get("timeout", DEFAULT_TIMEOUT))
        except (ValueError, TypeError):
            return DEFAULT_TIMEOUT

    def get_max_retries(self) -> int:
        """Get configured maximum retry attempts."""
        config = self.load_config()
        try:
            return int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        except (ValueError, TypeError):
            return DEFAULT_MAX_RETRIES
