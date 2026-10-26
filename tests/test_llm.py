"""OpenAI 兼容模型请求参数的回归测试。

by AI.Coding
"""

from pathlib import Path
from typing import Any

from experience_agent.llm import LLMClient, TruncatedResponseError


class RecordingClient(LLMClient):
    """记录请求体并返回固定响应的测试客户端。"""

    def __init__(self, cache_path: Path) -> None:
        """用假密钥和非思考配置初始化测试客户端。"""
        self.payloads: list[dict[str, Any]] = []
        super().__init__(
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            cache_path=cache_path,
            reasoning_effort="none",
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """截获请求并返回最小合法回答。"""
        self.payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {"content": "Answer: A"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }


def test_reasoning_effort_is_sent_and_affects_cache_key(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """确认非思考设置进入请求体，避免长题只产生隐藏推理。"""
    monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")
    client = RecordingClient(tmp_path / "cache.jsonl")

    result = client.complete(
        system="Answer the question.",
        user="Choose A.",
        max_tokens=32,
        tag="test",
    )

    assert result.text == "Answer: A"
    assert client.payloads[0]["reasoning_effort"] == "none"


class TruncatingThenCompleteClient(LLMClient):
    """Returns one truncated reply before a complete reply."""

    def __init__(self, cache_path: Path) -> None:
        self.payloads: list[dict[str, Any]] = []
        super().__init__(
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            cache_path=cache_path,
            max_retries=2,
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        if len(self.payloads) == 1:
            return {
                "choices": [
                    {"message": {"content": "Answer: partial"}, "finish_reason": "length"}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            }
        return {
            "choices": [
                {"message": {"content": "Answer: A"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }


def test_truncated_reply_retries_with_larger_budget(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")
    client = TruncatingThenCompleteClient(tmp_path / "cache.jsonl")

    result = client.complete(
        system="Answer the question.",
        user="Choose A.",
        max_tokens=16,
        tag="test",
    )

    assert result.text == "Answer: A"
    assert result.attempts == 2
    assert [payload["max_tokens"] for payload in client.payloads] == [16, 32]


class AlwaysTruncatedClient(LLMClient):
    def __init__(self, cache_path: Path) -> None:
        self.calls = 0
        super().__init__(
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            cache_path=cache_path,
            max_retries=2,
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {
            "choices": [
                {"message": {"content": "Answer: partial"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }


def test_permanent_truncation_is_never_cached(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")
    cache_path = tmp_path / "cache.jsonl"
    client = AlwaysTruncatedClient(cache_path)

    try:
        client.complete(
            system="Answer the question.",
            user="Choose A.",
            max_tokens=16,
            tag="test",
        )
    except TruncatedResponseError:
        pass
    else:
        raise AssertionError("truncated reply should not be returned as complete")

    assert client.calls == 2
    assert not cache_path.exists()
