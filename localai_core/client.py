"""Resilient HTTP client for LocalAI remote endpoints.

Supports OpenAI-compatible /v1/models and /v1/chat/completions APIs,
SSE streaming token generator, exponential backoff retries on transient failures,
and dynamic host health checks.
"""

import json
import os
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import RequestException, Timeout

from .config import ConfigManager


class LocalAIError(Exception):
    """Base exception for all LocalAI client errors."""

    pass


class LocalAIConnectionError(LocalAIError):
    """Raised when connection to LocalAI server fails after retries."""

    pass


class LocalAIHTTPError(LocalAIError):
    """Raised when LocalAI server returns an HTTP error code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class LocalAIClient:
    """Client for interacting with a remote LocalAI server."""

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        backoff_factor: float = 1.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize LocalAIClient.

        Args:
            config_manager: ConfigManager instance. If None, instantiates a default ConfigManager.
            base_url: Optional base URL override (e.g. http://host:port).
            timeout: Optional request timeout in seconds. Defaults to config setting.
            max_retries: Optional max retry count. Defaults to config setting.
            backoff_factor: Factor for exponential backoff (delay = factor * 2^attempt).
            session: Optional custom requests.Session.
        """
        self.config_manager = config_manager or ConfigManager()
        self.base_url = (base_url or self.config_manager.get_url()).rstrip("/")
        self.timeout = (
            timeout if timeout is not None else self.config_manager.get_timeout()
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else self.config_manager.get_max_retries()
        )
        self.backoff_factor = backoff_factor
        self.session = session or requests.Session()

    def _get_endpoint(self, path: str) -> str:
        """Construct full URL for an API path, avoiding duplicate /v1 prefixes."""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        clean_path = "/" + path.lstrip("/")
        return f"{base}{clean_path}"

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay in seconds with random jitter."""
        return self.backoff_factor * (2**attempt) + random.uniform(0, 0.25)

    def _sleep(self, seconds: float) -> None:
        """Sleep for specified duration."""
        if seconds > 0:
            time.sleep(seconds)

    def _get_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Construct default HTTP headers including optional authorization token."""
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("LOCALAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if custom_headers:
            headers.update(custom_headers)
        return headers

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Execute an HTTP request with automatic retry and exponential backoff."""
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        headers = self._get_headers(kwargs.pop("headers", None))
        kwargs["headers"] = headers

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(method, url, **kwargs)

                # Retry on 5xx server errors
                if 500 <= resp.status_code <= 599:
                    if attempt < self.max_retries:
                        delay = self._calculate_backoff(attempt)
                        self._sleep(delay)
                        continue
                    else:
                        raise LocalAIHTTPError(
                            status_code=resp.status_code,
                            message=f"Server error {resp.status_code} after {self.max_retries + 1} attempts: {resp.text}",
                        )

                # Do NOT retry on 4xx client errors
                if 400 <= resp.status_code <= 499:
                    raise LocalAIHTTPError(
                        status_code=resp.status_code,
                        message=f"Client error {resp.status_code}: {resp.text}",
                    )

                return resp

            except (ReqConnectionError, Timeout, RequestException) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    self._sleep(delay)
                    continue
                else:
                    raise LocalAIConnectionError(
                        f"Failed to connect to LocalAI server at {self.base_url} after "
                        f"{self.max_retries + 1} attempts: {exc}.\n"
                        f"The dynamic IP might have changed. You can update it using:\n"
                        f"  local-ai config set-url http://<new-ip>:<new-port>"
                    ) from exc

        # Fallback if loop finishes unexpectedly
        raise LocalAIConnectionError(
            f"Failed to connect to LocalAI server at {self.base_url}: {last_error}"
        )

    def list_models(self) -> List[str]:
        """Query GET /v1/models and return list of available model IDs.

        Returns:
            List[str]: List of model ID strings.
        """
        url = self._get_endpoint("/v1/models")
        resp = self._request_with_retry("GET", url)
        try:
            data = resp.json()
        except Exception as e:
            raise LocalAIError(f"Failed to parse models JSON response: {e}") from e

        if isinstance(data, dict):
            raw_list = data.get("data") or data.get("models") or []
        elif isinstance(data, list):
            raw_list = data
        else:
            raw_list = []


        models: List[str] = []
        for item in raw_list:
            if isinstance(item, dict) and "id" in item:
                models.append(str(item["id"]))
            elif isinstance(item, str):
                models.append(item)

        return models

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[Iterator[str], Dict[str, Any]]:
        """Send chat completion request to /v1/chat/completions.

        Args:
            messages: List of message dictionaries (role, content, etc.).
            model: Model name to invoke. Defaults to configured default_model.
            stream: When True, yields string chunks as SSE data arrives.
                    When False, returns parsed response dictionary.
            tools: Optional OpenAI-compatible tool specifications.
            temperature: Sampling temperature (default 0.7).
            max_tokens: Optional token limit.
            **kwargs: Additional parameters for the API payload.

        Returns:
            Iterator[str] if stream=True, or Dict[str, Any] if stream=False.
        """
        target_model = model or self.config_manager.get_default_model()
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        if stream:
            return self._stream_chat_completion(payload)
        else:
            url = self._get_endpoint("/v1/chat/completions")
            resp = self._request_with_retry("POST", url, json=payload)
            try:
                return resp.json()
            except Exception as e:
                raise LocalAIError(f"Failed to parse chat completion JSON: {e}") from e

    def _stream_chat_completion(self, payload: Dict[str, Any]) -> Iterator[str]:
        """Stream SSE response chunks from /v1/chat/completions, yielding tokens."""
        url = self._get_endpoint("/v1/chat/completions")
        timeout_arg = (15.0, float(self.timeout))
        resp = self._request_with_retry("POST", url, json=payload, stream=True, timeout=timeout_arg)

        yielded_any = False
        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                else:
                    line = str(raw_line).strip()

                if not line or line.startswith(":"):
                    continue

                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str.strip().strip("\"'").upper() in ("[DONE]", "DONE"):
                        break
                    if not data_str:
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if choices and isinstance(choices, list):
                        choice = choices[0]
                        if isinstance(choice, dict):
                            delta = choice.get("delta", {})
                            content = None
                            if isinstance(delta, dict):
                                content = delta.get("content")
                            elif isinstance(choice.get("text"), str):
                                content = choice.get("text")

                            if content is not None and isinstance(content, str) and content:
                                yielded_any = True
                                yield content

                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                break
        except (RequestException, Timeout):
            if not yielded_any:
                raise
        finally:
            resp.close()

    def health_check(self, timeout: float = 5.0) -> Tuple[bool, str, float]:
        """Ping remote endpoint /v1/models and return health status, message, and latency.

        Args:
            timeout: Connection timeout in seconds for ping (default 5.0).

        Returns:
            Tuple[bool, str, float]: (is_healthy, status_message, latency_seconds)
        """
        start = time.perf_counter()
        url = self._get_endpoint("/v1/models")

        try:
            headers = self._get_headers()
            resp = self.session.get(url, headers=headers, timeout=timeout)
            latency = round(time.perf_counter() - start, 4)

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    raw_models = (
                        (data.get("data") or data.get("models") or [])
                        if isinstance(data, dict)
                        else (data if isinstance(data, list) else [])
                    )
                    count = len(raw_models)
                    return True, f"Healthy ({count} models available)", latency

                except Exception:
                    return True, "Healthy", latency
            else:
                reason = getattr(resp, "reason", "Error")
                return (
                    False,
                    f"HTTP Error {resp.status_code}: {reason}",
                    latency,
                )
        except Exception as exc:
            latency = round(time.perf_counter() - start, 4)
            return False, f"Connection failed: {exc}", latency
