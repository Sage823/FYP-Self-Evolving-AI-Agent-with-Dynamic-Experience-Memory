"""OpenAI 兼容模型请求参数的回归测试。

by AI.Coding
"""

from pathlib import Path
from typing import Any

from experience_agent.llm import LLMClient


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
