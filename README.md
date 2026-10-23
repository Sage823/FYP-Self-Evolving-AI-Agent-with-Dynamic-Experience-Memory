# Experience Memory for a Self-Evolving LLM Agent

This repository is a fresh implementation of the FYP proposal **Self-Evolving AI
Agent with Dynamic Experience Memory** (COMP 4981, supervisor: Prof. LI Xiaomeng).
It does not import conclusions or result files from an earlier exploratory folder.

## Current status

Version `v0` implements the complete minimal learning loop:

1. ask an objective question without memory;
2. detect a wrong answer and read a recorded user correction;
3. store either the correction verbatim (`R`) or an abstracted lesson (`L`);
4. retrieve relevant memories for new, differently worded questions;
5. report repeated-error rate, retrieval hit rate, paired contrasts, and unrelated
   question safety.

The repository has a validated 32-item benchmark, 5 passing tests, an offline
end-to-end smoke mode, response caching, and hard per-run call/token budgets.

On 2026-10-23 the configured model (`deepseek-v4.1-flash`) answered all eight
feedback items correctly on the current hard benchmark. The runner therefore
stopped before inventing an effect or storing lessons from non-errors. This is a
benchmark-calibration result, not evidence for or against experience memory. The
next milestone is to find a task distribution on which the selected model makes
reproducible errors.

## System

```text
question
   |
   +-- no memory / raw correction memory / abstract lesson memory
   |          |
   |          +-- local hashed-vector retrieval (FAISS)
   |
   +-- answer
          |
          +-- wrong -> recorded correction -> lesson extraction -> memory store
```

The v0 experiment uses one model for both answering and lesson extraction. This
keeps the first milestone cheap and makes the workflow easy to debug.

## Quick start

Python 3.10+ is required.

```powershell
python -m pip install -e ".[dev]"
python -m experience_agent validate
python -m pytest
python -m experience_agent run --offline --output results/runs/offline_smoke
```

For a real model run, set the API key in the shell. The client also supports an
existing local DSH credential store, but never writes a key into the repository.

```powershell
$env:TOKENDANCE_API_KEY = "<key>"
python -m experience_agent run --config config/experiment_v0.json
```

Generated runs are written under `results/runs/` and are intentionally ignored by
Git. Commit only reviewed summaries or conclusions, not raw API caches or secrets.

## Design

The benchmark contains four error families:

| Family | Task |
|---|---|
| `E1_constraint_order` | six-person ordering with gap and adjacency constraints |
| `E2_status_precedence` | ordered decision rules where earlier rules override later ones |
| `E3_charge_rules` | conditional discounts and shipping calculations |
| `E4_protocol_triage` | synthetic educational protocol application |

Each family has 2 feedback and 4 follow-up items. Eight unrelated arithmetic or
language questions provide a basic safety check. Every mode uses the same model,
prompt, temperature, and retrieval settings; only memory contents differ.

The primary outcome is repeated-error rate on follow-up items from families where
at least one feedback error was observed. The runner also records exact McNemar
tests, retrieval hit rate, unrelated accuracy, API usage, and a complete JSONL
event trace.

See [the experiment manual](docs/EXPERIMENT_MANUAL.md) for design decisions,
trade-offs, troubleshooting, and the next experimental steps. A draft aligned to
the monthly report template is in
[the October progress notes](docs/PROGRESS_REPORT_NOTES_2026-10.md).

## Repository layout

```text
config/                  frozen v0 runtime settings
data/                    versioned benchmark items
src/experience_agent/    client, agent, memory, runner, and analysis
tests/                   offline unit and end-to-end tests
docs/                    experiment manual and progress-report notes
results/                 generated runs (Git-ignored)
```

## Cost and reproducibility

- Every model request is cached by content hash.
- Empty or length-capped replies are retried with a larger completion budget and
  are never cached as answers.
- A run stops at 120 new model calls and 300,000 billed tokens by default.
- Retrieval uses a 2,048-dimensional local hashing vectorizer and FAISS, avoiding
  embedding-model downloads and per-query embedding cost.
- The v0 run uses one repeat because it is a pipeline pilot, not a final
  statistical study.

