from types import SimpleNamespace

from experience_agent.agent import AnswerStatus, ExperienceAgent
from experience_agent.benchmark import BenchmarkItem
from experience_agent.llm import LLMResult, TruncatedResponseError, Usage


ITEM = BenchmarkItem(
    id="test",
    family="test_family",
    split="feedback",
    domain="test",
    question="Choose A.",
    answer="A",
    correction="The answer is A.",
)
CONFIG = SimpleNamespace(answer_max_tokens=100, lesson_max_tokens=100)


class FixedClient:
    model = "fixed"
    usage = Usage()
    billable_usage = Usage()
    new_calls = 0

    def __init__(self, *, text: str = "", exc: Exception | None = None) -> None:
        self.text = text
        self.exc = exc

    def complete(self, **_: object) -> LLMResult:
        if self.exc is not None:
            raise self.exc
        return LLMResult(
            text=self.text,
            model="fixed",
            usage=Usage(prompt_tokens=1, completion_tokens=1, calls=1),
            cached=False,
            latency_s=0.0,
            finish_reason="stop",
        )


def test_complete_answer_is_classified_correct_or_wrong() -> None:
    correct = ExperienceAgent(FixedClient(text="Answer: A"), CONFIG).answer(ITEM, [])
    wrong = ExperienceAgent(FixedClient(text="Answer: B"), CONFIG).answer(ITEM, [])
    assert correct.status == AnswerStatus.COMPLETE_CORRECT.value
    assert wrong.status == AnswerStatus.COMPLETE_WRONG.value


def test_truncated_and_malformed_answers_are_not_counted_as_complete_errors() -> None:
    truncated = ExperienceAgent(
        FixedClient(
            exc=TruncatedResponseError(
                "cut off", text="The answer is", finish_reason="length"
            )
        ),
        CONFIG,
    ).answer(ITEM, [])
    malformed = ExperienceAgent(FixedClient(text="I think A"), CONFIG).answer(ITEM, [])
    assert truncated.status == AnswerStatus.TRUNCATED.value
    assert not truncated.is_complete
    assert malformed.status == AnswerStatus.FORMAT_ERROR.value
    assert not malformed.is_complete
