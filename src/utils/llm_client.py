"""Qwen (DashScope) API client with retry and exponential backoff."""
from __future__ import annotations

import time
from typing import Optional

import dashscope

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenClient:
    """Thin wrapper around `dashscope.Generation.call` with retries."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.8,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries

    def _extract_text(self, response) -> str:
        output = getattr(response, "output", None)
        if output is None:
            raise ValueError("Qwen response has no output field")
        choices = getattr(output, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if content is not None:
                    return content
        text = getattr(output, "text", None)
        if text is not None:
            return text
        raise ValueError("Qwen response has no recognizable text payload")

    def call(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                response = dashscope.Generation.call(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    result_format="message",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    timeout=self.timeout,
                )
                status_code = getattr(response, "status_code", None)
                if status_code == 200:
                    return self._extract_text(response)
                # non-200: treat as retryable failure
                err_msg = (
                    f"Qwen call failed: status={status_code} "
                    f"code={getattr(response, 'code', None)} "
                    f"message={getattr(response, 'message', None)}"
                )
                logger.warning("%s (attempt %d/%d)", err_msg, attempt + 1, self.max_retries)
                last_exc = RuntimeError(err_msg)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Qwen call raised %s (attempt %d/%d): %s",
                    type(exc).__name__,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                last_exc = exc

            if attempt < self.max_retries - 1:
                backoff = 1.5 ** attempt
                time.sleep(backoff)

        assert last_exc is not None
        raise last_exc
