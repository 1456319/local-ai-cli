"""Single-shot query runner for LocalAI remote client.

Sends a prompt to the remote LocalAI server and streams token deltas in real-time.
"""

import sys
from typing import Any, Optional

from .client import LocalAIClient


def run_prompt(
    client: LocalAIClient,
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    out_stream: Any = sys.stdout,
    err_stream: Any = sys.stderr,
) -> int:
    """Execute a single prompt against the remote client and stream response tokens.

    Args:
        client: LocalAIClient instance to communicate with.
        prompt: User prompt text.
        model: Optional model name override.
        system: Optional system prompt to prepend.
        temperature: Sampling temperature (default 0.7).
        out_stream: Stream to write streamed tokens and ending newline to (default stdout).
        err_stream: Stream to write error messages to on failure (default stderr).

    Returns:
        0 on success, 1 on error.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt if prompt is not None else ""})

    try:
        stream = client.chat_completion(
            messages=messages,
            model=model,
            stream=True,
            temperature=temperature,
        )
        for chunk in stream:
            out_stream.write(chunk)
            out_stream.flush()
        out_stream.write("\n")
        out_stream.flush()
        return 0
    except KeyboardInterrupt:
        out_stream.write("\n")
        out_stream.flush()
        if err_stream:
            err_stream.write("Interrupted.\n")
            err_stream.flush()
        return 1
    except Exception as exc:
        if err_stream:
            err_stream.write(f"Error: {exc}\n")
            err_stream.flush()
        return 1
