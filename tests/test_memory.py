"""ChromaDB 经验记忆的回归测试。

by AI.Coding
"""

from pathlib import Path

from experience_agent.memory import ExperienceMemory


def test_memory_retrieves_same_error_type() -> None:
    memory = ExperienceMemory(dimension=256)
    memory.add(
        {
            "error_type": "quantifier",
            "rule": "Not every does not mean none.",
            "trigger": "Questions about not every or some",
        }
    )
    memory.add(
        {
            "error_type": "ordering",
            "rule": "Follow each ordering constraint transitively.",
            "trigger": "Questions with before and after constraints",
        }
    )
    hits = memory.search("Not every parcel was scanned. Does that mean none?", k=2)
    assert hits
    assert hits[0].record["error_type"] == "quantifier"


def test_duplicate_memory_is_rejected() -> None:
    """确认完全相同的经验不会重复写入。"""
    memory = ExperienceMemory(dimension=128)
    record = {"rule": "Check the implication direction.", "trigger": "Implication questions"}
    assert memory.add(record)
    assert not memory.add(dict(record))
    assert len(memory) == 1


def test_chroma_memory_can_be_reopened(tmp_path: Path) -> None:
    """确认落盘的 Chroma 集合可由新实例重新打开并检索。"""
    persist_directory = tmp_path / "chroma"
    record = {
        "error_type": "quantifier",
        "rule": "Not every does not mean none.",
        "trigger": "Questions about not every or some",
    }

    first = ExperienceMemory(
        dimension=128,
        persist_directory=persist_directory,
        collection_name="test_memory",
    )
    assert first.add(record)
    assert first.backend == "chromadb"

    # 新实例必须从磁盘集合恢复记录，而不是依赖第一个对象的内存状态。
    reopened = ExperienceMemory(
        dimension=128,
        persist_directory=persist_directory,
        collection_name="test_memory",
    )
    hits = reopened.search("Does not every mean none?", k=1)

    assert len(reopened) == 1
    assert hits
    assert hits[0].record["error_type"] == "quantifier"


def test_hashing_backend_reports_embedding_metadata() -> None:
    memory = ExperienceMemory(dimension=128, embedding_backend="hashing")
    assert memory.backend == "chromadb"
    assert memory.embedding_name.startswith("hashing:")
    assert memory.embedding_dimension == 128

