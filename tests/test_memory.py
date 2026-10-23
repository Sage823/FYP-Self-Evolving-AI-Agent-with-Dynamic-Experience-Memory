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
    memory = ExperienceMemory(dimension=128)
    record = {"rule": "Check the implication direction.", "trigger": "Implication questions"}
    assert memory.add(record)
    assert not memory.add(dict(record))
    assert len(memory) == 1

