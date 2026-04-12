from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from src.utils.llm_client import QwenClient


def _make_success_response(text: str = "hello"):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    output = SimpleNamespace(choices=[choice], text=text)
    return SimpleNamespace(status_code=200, output=output, request_id="req-1", code=None, message="ok")


def _make_failure_response():
    return SimpleNamespace(status_code=500, output=None, request_id="req-err", code="ServerError", message="boom")


@patch("src.utils.llm_client.time.sleep", return_value=None)
@patch("src.utils.llm_client.dashscope.Generation.call")
def test_qwen_client_call_returns_text(mock_call, mock_sleep):
    mock_call.return_value = _make_success_response("hello")

    client = QwenClient(api_key="fake", model="qwen-turbo")
    result = client.call("prompt")

    assert result == "hello"
    assert mock_call.call_count == 1


@patch("src.utils.llm_client.time.sleep", return_value=None)
@patch("src.utils.llm_client.dashscope.Generation.call")
def test_qwen_client_retries_on_failure(mock_call, mock_sleep):
    mock_call.side_effect = [
        Exception("network err"),
        _make_failure_response(),
        _make_success_response("hello"),
    ]

    client = QwenClient(api_key="fake", model="qwen-turbo", max_retries=3)
    result = client.call("prompt")

    assert result == "hello"
    assert mock_call.call_count == 3
    assert mock_sleep.call_count >= 2
