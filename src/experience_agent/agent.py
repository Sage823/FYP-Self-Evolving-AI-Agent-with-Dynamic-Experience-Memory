"""Answering and correction-to-lesson behaviour for the agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .benchmark import BenchmarkItem, normalize_answer
from .llm import LLMClient, MockLLM, extract_json


ANSWER_SYSTEM = """You answer one objective question.
Use only the supplied question and any explicitly supplied experience memory.
Do not invent information. Finish with exactly one final line in this format:
Answer: <your answer>

If experience memory is present, use an item only when it genuinely applies.
An irrelevant memory must not replace direct reasoning about the question."""

ANSWER_WITH_MEMORY = """Experience memory:
{memory_block}

Question:
{question}"""

ANSWER_WITHOUT_MEMORY = """Question:
{question}"""

LESSON_SYSTEM = """You convert a correct user correction into one reusable lesson.
Return exactly one JSON object with these keys:
{
  "rule": "a general check or rule for similar future questions",
  "trigger": "the condition under which the rule applies",
  "error_type_guess": "short snake_case label"
}
The lesson must not contain names, numbers, or wording that identifies only the
original question. It must be useful when applied to a differently worded item
from the same family."""

LESSON_USER = """Original question:
{question}

The agent answered:
{wrong_answer}

Correct answer:
{correct_answer}

User correction:
{correction}

Write the reusable lesson as JSON only."""


@dataclass
class AnswerResult:
    raw_text: str
    answer: str
    correct: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "answer": self.answer,
            "correct": self.correct,
        }


@dataclass
class Lesson:
    rule: str
    trigger: str
    error_type_guess: str

    def as_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "trigger": self.trigger,
            "error_type_guess": self.error_type_guess,
        }


class ExperienceAgent:
    def __init__(self, client: LLMClient | MockLLM, config: Any) -> None:
        self.client = client
        self.config = config

    def answer(self, item: BenchmarkItem, memories: list[dict[str, Any]]) -> AnswerResult:
        if memories:
            block = "\n\n".join(
                f"[Memory {index}] {record['memory_text']}"
                for index, record in enumerate(memories, start=1)
            )
            user = ANSWER_WITH_MEMORY.format(memory_block=block, question=item.question)
        else:
            user = ANSWER_WITHOUT_MEMORY.format(question=item.question)
        result = self.client.complete(
            system=ANSWER_SYSTEM,
            user=user,
            max_tokens=self.config.answer_max_tokens,
            tag=f"answer:{item.id}",
        )
        answer = normalize_answer(result.text)
        return AnswerResult(
            raw_text=result.text,
            answer=answer,
            correct=answer == item.answer,
        )

    def extract_lesson(
        self,
        item: BenchmarkItem,
        wrong_answer: str,
    ) -> Lesson:
        if not item.correction:
            raise ValueError(f"{item.id} has no correction")
        user = LESSON_USER.format(
            question=item.question,
            wrong_answer=wrong_answer,
            correct_answer=item.answer,
            correction=item.correction,
        )
        errors: list[str] = []
        for json_mode in (True, False):
            suffix = "" if json_mode else "\nReturn only the JSON object, with no prose or code fence."
            try:
                result = self.client.complete(
                    system=LESSON_SYSTEM,
                    user=user + suffix,
                    max_tokens=self.config.lesson_max_tokens,
                    tag=f"lesson:{item.id}",
                    json_mode=json_mode,
                )
                parsed = extract_json(result.text)
                rule = str(parsed.get("rule", "")).strip()
                trigger = str(parsed.get("trigger", "")).strip()
                error_type = str(
                    parsed.get("error_type_guess", item.family)
                ).strip() or item.family
                if not rule or not trigger:
                    raise ValueError("lesson JSON is missing rule or trigger")
                if len(rule.split()) > 45 or len(trigger.split()) > 20:
                    raise ValueError("lesson is longer than the v0 limits")
                return Lesson(rule=rule, trigger=trigger, error_type_guess=error_type)
            except Exception as exc:  # Try the next response-format strategy.
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError(f"could not extract a valid lesson for {item.id}: {errors}")


def records_from_lessons(lessons: list[dict[str, Any]]) -> str:
    return json.dumps(lessons, indent=2, ensure_ascii=False)

