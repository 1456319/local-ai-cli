"""localai_core package for local-ai remote client."""

from .config import (
    ConfigManager,
    DEFAULT_URL,
    DEFAULT_MODEL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)
from .client import (
    LocalAIClient,
    LocalAIError,
    LocalAIConnectionError,
    LocalAIHTTPError,
)

__all__ = [
    "ConfigManager",
    "DEFAULT_URL",
    "DEFAULT_MODEL",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "LocalAIClient",
    "LocalAIError",
    "LocalAIConnectionError",
    "LocalAIHTTPError",
]

