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

The repository has validated BBH-inspired benchmarks, 25 passing tests, an
offline end-to-end mode, response caching, and hard per-run call/token budgets.
The v3 protocol has separate baseline, feedback, retrieval-validation, follow-up,
and unrelated splits. Responses are classified as complete-correct,
complete-wrong, truncated, empty, malformed, or API error; only complete answers
enter repeated-error metrics.

On 2026-10-23 the previously configured model (`deepseek-v4.1-flash`) answered all eight
feedback items correctly on the current hard benchmark. The runner therefore
stopped before inventing an effect or storing lessons from non-errors. This is a
benchmark-calibration result, not evidence for or against experience memory. It
motivated the harder BBH-inspired task distribution described below.

The earlier `benchmark_bbh_v2.jsonl` run produced failures, but several were
output-length truncations rather than genuine reasoning errors. Those observations
motivated the stricter v3 workflow; they are not treated as a memory effect.

An expanded v3 run froze baseline, feedback, validation, model, embedding, and
retrieval settings while increasing the eligible family's follow-up set from 4 to
20 items. With two truncated answers excluded, repeated errors were 63.2% without
memory, 36.8% with raw corrections, and 35.0% with abstracted lessons. Raw and
abstract memory both improved on no memory; the difference between raw and
abstract memory remained small and non-significant. This is a one-run descriptive
result, not a final claim. See
[the reviewed result summary](docs/V3_EXPANDED_RESULTS_2026-10-27.md).

## System

```text
question
   |
   +-- no memory / raw correction memory / abstract lesson memory
   |          |
   |          +-- hashing or sentence-transformers embeddings in ChromaDB
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
# Create an isolated environment so project dependencies do not leak into the
# system Python installation.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"                 # hash baseline only
# or: python -m pip install -e ".[dev,semantic]"  # adds MiniLM/PyTorch
python -m experience_agent validate
python -m pytest
python -m experience_agent run --offline --output results/runs/offline_smoke
python -m experience_agent.bbh_benchmark --difficulty 2 `
  --with-development-splits --output data/benchmark_bbh_v3.jsonl
python -m experience_agent.calibration --backends hashing,sentence_transformers
python -m experience_agent run --config config/experiment_bbh_v3.json
python -m experience_agent run --config config/experiment_bbh_v3_semantic.json
```

For a real model run, set the API key in the shell. The client also supports an
existing local DSH credential store, but never writes a key into the repository.

```powershell
$env:DEEPSEEK_API_KEY = "<key>"
python -m experience_agent run --config config/experiment_bbh_v3_semantic.json
```

Generated runs are written under `results/runs/` and are intentionally ignored by
Git. Commit only reviewed summaries or conclusions, not raw API caches or secrets.

## Experimental protocol

The v3 benchmark contains four error families:

| Family | Task |
|---|---|
| `B1_state_tracking` | long state-update sequences |
| `B2_boolean_logic` | deeply nested Boolean expressions |
| `B3_date_arithmetic` | sequential date arithmetic |
| `B4_dyck_completion` | bracket-stack completion |

Each family has 2 baseline, 2 feedback, 2 validation, and 4 follow-up items.
Eight unrelated items provide a safety check. Baseline selects error families;
feedback creates memory; validation calibrates retrieval; follow-up evaluates
transfer. Follow-up items are never used to select models or retrieval thresholds.

The primary outcome is repeated-error rate on follow-up items whose status is
`COMPLETE_WRONG`. Truncated, empty, malformed, and API-error responses are reported
separately and excluded from the analysis denominator.

See [the experiment manual](docs/EXPERIMENT_MANUAL.md) for design decisions,
trade-offs, troubleshooting, and the next experimental steps. A draft aligned to
the monthly report template is in
[the October progress notes](docs/PROGRESS_REPORT_NOTES_2026-10.md). A step-by-step
introduction to the new status, baseline, calibration, and Git workflow is in
[the beginner workflow guide](docs/BEGINNER_WORKFLOW.md).

## Repository layout

```text
config/                  frozen v0 runtime settings
data/                    versioned benchmark items
src/experience_agent/    client, agent, memory, calibration, runner, analysis
tests/                   offline unit and end-to-end tests
docs/                    experiment manual and progress-report notes
results/                 generated runs (Git-ignored)
```

## Cost and reproducibility

- Every model request is cached by content hash.
- Empty or length-capped replies are retried with a larger completion budget and
  are never cached as answers.
- A run stops at 120 new model calls and 300,000 billed tokens by default.
- Retrieval supports a deterministic local hashing baseline and optional
  sentence-transformers embeddings with persistent ChromaDB collections.
- The v0 run uses one repeat because it is a pipeline pilot, not a final
  statistical study.

