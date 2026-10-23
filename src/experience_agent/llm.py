"""Small OpenAI-compatible client with caching and hard per-run budgets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Usage
    cached: bool
    latency_s: float


class BudgetExceeded(RuntimeError):
    """Raised before a run exceeds its explicit API-call or token budget."""


def _load_api_key() -> str:
    for name in ("EXPERIENCE_LLM_API_KEY", "TOKENDANCE_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    candidates = [
        Path(os.environ.get("DSH_HOME", "D:/dsh")) / ".credentials.yaml",
        Path.home() / ".dsh" / ".credentials.yaml",
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"^\s*TOKENDANCE_API_KEY:\s*(\S+)\s*$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    raise RuntimeError(
        "No API key found. Set EXPERIENCE_LLM_API_KEY or TOKENDANCE_API_KEY."
    )


def extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError(f"reply contains no JSON object: {stripped[:240]!r}")


class LLMClient:
    """Sequential client because the pilot is small and cost control is easier."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        cache_path: str | Path,
        temperature: float = 0.0,
        max_new_calls: int = 120,
        max_total_tokens: int = 300_000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.max_new_calls = max_new_calls
        self.max_total_tokens = max_total_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._key = _load_api_key()
        self._cache = self._load_cache()
        self.usage = Usage()
        self.billable_usage = Usage()
        self.new_calls = 0

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        cache: dict[str, dict[str, Any]] = {}
        if not self.cache_path.exists():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    cache[record["key"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue
        return cache

    def _cache_key(
        self, messages: Sequence[dict[str, str]], max_tokens: int, json_mode: bool
    ) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _append_cache(self, record: dict[str, Any]) -> None:
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.load(response)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        tag: str,
        json_mode: bool = False,
    ) -> LLMResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            # Some reasoning models spend the whole completion budget before
            # emitting visible content. Do not cache that result as an answer;
            # retry the same prompt with a larger visible-output allowance.
            attempt_tokens = min(max_tokens * (2 ** (attempt - 1)), 4096)
            key = self._cache_key(messages, attempt_tokens, json_mode)
            cached = self._cache.get(key)
            if cached is not None:
                usage = Usage(**cached["usage"])
                self.usage.add(usage)
                return LLMResult(
                    text=cached["text"],
                    model=cached.get("model", self.model),
                    usage=usage,
                    cached=True,
                    latency_s=float(cached.get("latency_s", 0.0)),
                )
            if self.new_calls >= self.max_new_calls:
                raise BudgetExceeded(
                    f"new-call budget reached ({self.max_new_calls}); "
                    "rerun with a higher cap if needed"
                )
            if self.billable_usage.total_tokens >= self.max_total_tokens:
                raise BudgetExceeded(
                    f"token budget reached ({self.max_total_tokens}); "
                    "inspect the cache before raising it"
                )
            self.new_calls += 1
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": attempt_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            started = time.perf_counter()
            try:
                data = self._post(payload)
                choice = (data.get("choices") or [{}])[0]
                text = str((choice.get("message") or {}).get("content") or "").strip()
                if not text:
                    raise RuntimeError(
                        f"empty model reply (finish_reason={choice.get('finish_reason')})"
                    )
                raw_usage = data.get("usage") or {}
                usage = Usage(
                    prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                    completion_tokens=int(raw_usage.get("completion_tokens") or 0),
                    calls=1,
                )
                latency = time.perf_counter() - started
                self.usage.add(usage)
                self.billable_usage.add(usage)
                if self.billable_usage.total_tokens > self.max_total_tokens:
                    raise BudgetExceeded(
                        f"token budget exceeded after call: {self.billable_usage.total_tokens} > "
                        f"{self.max_total_tokens}"
                    )
                record = {
                    "key": key,
                    "tag": tag,
                    "model": self.model,
                    "text": text,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "calls": 1,
                    },
                    "latency_s": round(latency, 4),
                    "finish_reason": choice.get("finish_reason"),
                }
                self._cache[key] = record
                self._append_cache(record)
                return LLMResult(
                    text=text,
                    model=self.model,
                    usage=usage,
                    cached=False,
                    latency_s=latency,
                )
            except BudgetExceeded:
                raise
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403, 404}:
                    raise RuntimeError(
                        f"provider rejected model or credentials with HTTP {exc.code}"
                    ) from exc
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(8.0, 1.5 ** attempt))
            except Exception as exc:  # Network and provider errors are retried.
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(8.0, 1.5 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")


class MockLLM:
    """Deterministic offline stand-in used only to verify the pipeline."""

    model = "mock-model"

    def __init__(self) -> None:
        self.usage = Usage()
        self.billable_usage = Usage()
        self.new_calls = 0

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        tag: str,
        json_mode: bool = False,
    ) -> LLMResult:
        self.new_calls += 1
        if "json" in system.lower() or json_mode:
            text = json.dumps(
                {
                    "rule": "Check whether the stated evidence is sufficient before answering.",
                    "trigger": "When the question asks what necessarily follows",
                    "error_type_guess": "insufficient_evidence",
                }
            )
        elif "Reply YES or NO" in user:
            text = "The answer format is a simple choice.\nAnswer: NO"
        elif "one letter" in user.lower():
            text = "The answer format is a simple choice.\nAnswer: A"
        else:
            text = "This is the offline test answer.\nAnswer: 7"
        usage = Usage(prompt_tokens=len(system) // 4 + len(user) // 4, completion_tokens=len(text) // 4, calls=1)
        self.usage.add(usage)
        self.billable_usage.add(usage)
        return LLMResult(
            text=text,
            model=self.model,
            usage=usage,
            cached=False,
            latency_s=0.0,
        )

