"""生成可复现、带 oracle 答案的 BBH 风格 v1 benchmark。

题目借鉴 BBH 的能力类型，但不复制公开题目或答案。
by AI.Coding
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from .benchmark import BenchmarkItem, validate_benchmark


DEFAULT_SEED = 20261026
FAMILY_LAYOUT = (("feedback", 2), ("followup", 4))


def _item_id(prefix: str, split: str, index: int) -> str:
    """按题族、数据分区和序号生成稳定 ID。"""
    short_split = "fb" if split == "feedback" else "fu"
    return f"{prefix}_{short_split}_{index:02d}"


def _state_tracking_items(
    rng: random.Random, difficulty: int = 1
) -> list[BenchmarkItem]:
    """生成连续交换任务，并用字典逐步更新得到标准答案。"""
    people_pool = [
        "Ava",
        "Ben",
        "Cora",
        "Dylan",
        "Eli",
        "Faye",
        "Gina",
        "Hugo",
        "Iris",
        "Jude",
        "Kira",
        "Liam",
    ]
    object_pool = [
        "amber",
        "bronze",
        "coral",
        "denim",
        "emerald",
        "flint",
        "gold",
        "hazel",
        "ivory",
        "jade",
        "khaki",
        "lilac",
    ]
    items: list[BenchmarkItem] = []
    family_index = 0
    for split, count in FAMILY_LAYOUT:
        for split_index in range(1, count + 1):
            family_index += 1
            participant_count = 9 if difficulty == 1 else 12
            people = rng.sample(people_pool, participant_count)
            objects = rng.sample(object_pool, participant_count)
            holdings = dict(zip(people, objects))
            initial = dict(holdings)
            swaps: list[tuple[str, str]] = []
            swap_count = 24 + family_index if difficulty == 1 else 52 + family_index * 2
            for _ in range(swap_count):
                left, right = rng.sample(people, 2)
                swaps.append((left, right))
                holdings[left], holdings[right] = holdings[right], holdings[left]
            target = people[(family_index * 3) % len(people)]
            answer = holdings[target]
            distractor = initial[target]
            if distractor == answer:
                distractor = next(value for value in objects if value != answer)
            assignments = "; ".join(
                f"{person} starts with {obj}" for person, obj in zip(people, objects)
            )
            operations = "; ".join(
                f"{number}. {left} swaps with {right}"
                for number, (left, right) in enumerate(swaps, start=1)
            )
            question = (
                "Track the objects exactly. "
                f"{assignments}. Perform these swaps in order: {operations}. "
                f"After every swap, each person keeps the newly received object. "
                f"What object does {target} hold at the end? Reply with the object word only."
            )
            correction = None
            if split == "feedback":
                correction = (
                    f"The correct final object is {answer}. General method: maintain a complete "
                    "person-to-object table and update both people immediately after every swap; "
                    "each later swap must use the updated table, never the initial assignments."
                )
            items.append(
                BenchmarkItem(
                    id=_item_id("b1", split, split_index),
                    family="B1_state_tracking",
                    split=split,
                    domain="algorithmic_reasoning",
                    question=question,
                    answer=answer,
                    distractor=distractor,
                    correction=correction,
                )
            )
    return items


def _boolean_expression(rng: random.Random, depth: int) -> tuple[str, bool]:
    """递归生成完全加括号的布尔表达式及其 oracle 值。"""
    if depth == 0:
        value = bool(rng.getrandbits(1))
        return ("True" if value else "False"), value
    if rng.random() < 0.28:
        child_text, child_value = _boolean_expression(rng, depth - 1)
        return f"(not {child_text})", not child_value
    left_text, left_value = _boolean_expression(rng, depth - 1)
    right_text, right_value = _boolean_expression(rng, depth - 1)
    operator = rng.choice(("and", "or"))
    value = left_value and right_value if operator == "and" else left_value or right_value
    return f"({left_text} {operator} {right_text})", value


def _boolean_items(rng: random.Random, difficulty: int = 1) -> list[BenchmarkItem]:
    """生成深层布尔表达式任务。"""
    items: list[BenchmarkItem] = []
    for split, count in FAMILY_LAYOUT:
        for split_index in range(1, count + 1):
            expression, value = _boolean_expression(
                rng, depth=6 if difficulty == 1 else 8
            )
            answer = "TRUE" if value else "FALSE"
            distractor = "FALSE" if value else "TRUE"
            correction = None
            if split == "feedback":
                correction = (
                    f"The expression evaluates to {answer}. General method: evaluate the "
                    "innermost parentheses first, apply each not to its complete child result, "
                    "and only then combine the two Boolean values with and/or while recording "
                    "every intermediate value."
                )
            items.append(
                BenchmarkItem(
                    id=_item_id("b2", split, split_index),
                    family="B2_boolean_logic",
                    split=split,
                    domain="formal_logic",
                    question=(
                        "Evaluate this fully parenthesized Boolean expression exactly: "
                        f"{expression}. Reply TRUE or FALSE only."
                    ),
                    answer=answer,
                    distractor=distractor,
                    correction=correction,
                )
            )
    return items


def _date_items(rng: random.Random, difficulty: int = 1) -> list[BenchmarkItem]:
    """生成跨月、跨年和闰年的顺序日期运算任务。"""
    items: list[BenchmarkItem] = []
    total_index = 0
    for split, count in FAMILY_LAYOUT:
        for split_index in range(1, count + 1):
            total_index += 1
            start = date(
                rng.randint(1996, 2028),
                rng.randint(1, 12),
                1,
            ) + timedelta(days=rng.randint(0, 24))
            offset_count = 6 + total_index % 2 if difficulty == 1 else 14 + total_index % 3
            offsets = [
                rng.choice((-1, 1)) * rng.randint(37, 620)
                for _ in range(offset_count)
            ]
            current = start
            for offset in offsets:
                current += timedelta(days=offset)
            answer = current.isoformat()
            distractor = (current + timedelta(days=rng.choice((-1, 1)))).isoformat()
            operations = ", then ".join(
                f"move {'forward' if offset >= 0 else 'backward'} {abs(offset)} days"
                for offset in offsets
            )
            correction = None
            if split == "feedback":
                correction = (
                    f"The correct final date is {answer}. General method: apply every signed "
                    "day offset sequentially to the date produced by the previous step, using "
                    "actual month lengths and leap-year rules; the starting date is day zero."
                )
            items.append(
                BenchmarkItem(
                    id=_item_id("b3", split, split_index),
                    family="B3_date_arithmetic",
                    split=split,
                    domain="date_reasoning",
                    question=(
                        f"Start on {start.isoformat()} and treat it as day zero. {operations}. "
                        "What is the final Gregorian date? Reply exactly as YYYY-MM-DD."
                    ),
                    answer=answer,
                    distractor=distractor,
                    correction=correction,
                )
            )
    return items


def _dyck_problem(
    rng: random.Random, index: int, difficulty: int = 1
) -> tuple[str, str, list[str]]:
    """生成合法括号前缀、唯一补全串和四个互异选项。"""
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    openings = list(pairs)
    stack: list[str] = []
    prefix: list[str] = []
    prefix_steps = 30 + index if difficulty == 1 else 68 + index * 2
    maximum_depth = 17 if difficulty == 1 else 31
    for step in range(prefix_steps):
        should_open = len(stack) < 5 or (
            len(stack) < maximum_depth and rng.random() < 0.72
        )
        if should_open:
            token = rng.choice(openings)
            stack.append(token)
            prefix.append(token)
        else:
            prefix.append(pairs[stack.pop()])
    minimum_remaining = 11 if difficulty == 1 else 24
    while len(stack) < minimum_remaining:
        token = rng.choice(openings)
        stack.append(token)
        prefix.append(token)
    correct_tokens = [pairs[token] for token in reversed(stack)]
    correct = " ".join(correct_tokens)
    candidates = {correct, " ".join(reversed(correct_tokens))}
    if len(correct_tokens) >= 2:
        swapped = list(correct_tokens)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        candidates.add(" ".join(swapped))
    replacements = list(pairs.values())
    attempt = 0
    while len(candidates) < 4:
        wrong = list(correct_tokens)
        position = attempt % len(wrong)
        wrong[position] = replacements[(replacements.index(wrong[position]) + 1) % 4]
        candidates.add(" ".join(wrong))
        attempt += 1
    options = list(candidates)
    rng.shuffle(options)
    return " ".join(prefix), chr(65 + options.index(correct)), options


def _dyck_items(rng: random.Random, difficulty: int = 1) -> list[BenchmarkItem]:
    """生成 BBH Dyck-language 风格的栈补全任务。"""
    items: list[BenchmarkItem] = []
    total_index = 0
    for split, count in FAMILY_LAYOUT:
        for split_index in range(1, count + 1):
            total_index += 1
            prefix, answer, options = _dyck_problem(rng, total_index, difficulty)
            option_text = " ".join(
                f"{chr(65 + index)}) {value}"
                for index, value in enumerate(options)
            )
            distractor = next(letter for letter in "ABCD" if letter != answer)
            correction = None
            if split == "feedback":
                correction = (
                    f"The correct option is {answer}. General method: scan left to right with "
                    "a stack, push every opening bracket, pop only when its matching closer "
                    "appears, then close all remaining openings in exact reverse stack order."
                )
            items.append(
                BenchmarkItem(
                    id=_item_id("b4", split, split_index),
                    family="B4_dyck_completion",
                    split=split,
                    domain="formal_language",
                    question=(
                        "The following is a valid prefix of a bracket sequence using (), [], {}, "
                        f"and <>: {prefix}. Which option closes every still-open bracket correctly? "
                        f"{option_text} Reply with one letter only."
                    ),
                    answer=answer,
                    distractor=distractor,
                    correction=correction,
                )
            )
    return items


def _unrelated_items() -> list[BenchmarkItem]:
    """生成无需经验记忆即可回答的基础安全检查题。"""
    cases = [
        ("What is 47 + 38? Reply with the number only.", "85"),
        ("What is 144 divided by 12? Reply with the number only.", "12"),
        ("What is 19 multiplied by 7? Reply with the number only.", "133"),
        ("What is 205 minus 87? Reply with the number only.", "118"),
        ("How many letters are in the English word 'planet'? Reply with the number only.", "6"),
        ("Which is larger, 0.72 or 0.702? Reply with the larger decimal only.", "0.72"),
        ("What is the next integer after 999? Reply with the number only.", "1000"),
        ("If a triangle has three sides, how many sides do two separate triangles have in total? Reply with the number only.", "6"),
    ]
    return [
        BenchmarkItem(
            id=f"bbh_u_{index:02d}",
            family="unrelated",
            split="unrelated",
            domain="basic_safety",
            question=question,
            answer=answer,
        )
        for index, (question, answer) in enumerate(cases, start=1)
    ]


def build_benchmark(
    seed: int = DEFAULT_SEED, difficulty: int = 1
) -> list[BenchmarkItem]:
    """按固定随机种子构建完整的 32 道 BBH 风格 benchmark。"""
    rng = random.Random(seed)
    if difficulty not in {1, 2}:
        raise ValueError("difficulty must be 1 or 2")
    items = [
        *_state_tracking_items(rng, difficulty),
        *_boolean_items(rng, difficulty),
        *_date_items(rng, difficulty),
        *_dyck_items(rng, difficulty),
        *_unrelated_items(),
    ]
    validate_benchmark(items)
    return items


def write_benchmark(
    path: str | Path, seed: int = DEFAULT_SEED, difficulty: int = 1
) -> Path:
    """把生成的 benchmark 以 UTF-8 JSONL 写入指定路径。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": item.id,
                "family": item.family,
                "split": item.split,
                "domain": item.domain,
                "question": item.question,
                "answer": item.answer,
                **({"distractor": item.distractor} if item.distractor else {}),
                **({"correction": item.correction} if item.correction else {}),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for item in build_benchmark(seed, difficulty)
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    """提供命令行入口，便于重新生成和审查版本化题库。"""
    parser = argparse.ArgumentParser(description="Generate the BBH-inspired v1 benchmark")
    parser.add_argument("--output", default="data/benchmark_bbh_v1.jsonl")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--difficulty", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    output = write_benchmark(args.output, args.seed, args.difficulty)
    print(
        f"Wrote {len(build_benchmark(args.seed, args.difficulty))} items to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
