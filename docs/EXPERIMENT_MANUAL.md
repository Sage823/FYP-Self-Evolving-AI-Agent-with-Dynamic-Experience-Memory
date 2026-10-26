# Experiment Manual

Last updated: 2026-10-23

## 1. Scope

This repository starts a new implementation from the FYP proposal rather than
reusing results, benchmarks, or conclusions from the earlier `fyp/` attempt. The
goal of v0 is to establish a correct and inspectable loop:

```text
answer -> correction -> lesson -> external vector memory -> retrieval -> new answer
```

The first milestone deliberately uses one model, one repeat, a small benchmark,
and local lexical embeddings. It is designed to expose engineering and measurement
problems before spending on a full evaluation.

## 2. Research questions

The proposal's two core hypotheses remain the reference:

- **H1:** A lesson obtained from a correction reduces repeated errors on
  differently worded questions compared with no memory.
- **H2:** An abstracted lesson reduces repeated errors more than a raw correction.

Operational requirement:

- consulting memory should not reduce accuracy on questions for which no lesson
  applies.

The v0 metrics are descriptive. They are not sufficient for a final statistical
claim.

## 3. Experimental units

### Data splits

| Split | Count | Purpose |
|---|---:|---|
| feedback | 8 | measure initial errors and create memories |
| follow-up | 16 | measure whether the same error family recurs |
| unrelated | 8 | basic safety check |

There are four error families with two feedback and four follow-up items each.
Feedback and follow-up text never overlap.

### Conditions

| Code | Stored memory | Retrieved | Role |
|---|---|---|---|
| `N` | none | no | baseline |
| `R` | user correction, unchanged | yes | raw-correction control |
| `L` | abstract rule plus trigger | yes | proposed lesson treatment |

Only feedback items answered incorrectly can create a memory. A family enters the
follow-up comparison only if at least 30% of its feedback items are wrong. This
prevents the experiment from claiming an effect on a task the model already solves.

## 4. Pipeline

1. **Validate data:** unique IDs, disjoint text, valid corrections, and matching
   feedback/follow-up families.
2. **Feedback baseline:** answer every feedback item with no memory.
3. **Select families:** retain families whose measured feedback error rate reaches
   the configured threshold.
4. **Create memory:** store wrong-item corrections under `R`; ask the model for a
   general rule and store it under `L`.
5. **Retrieve:** encode a query with local hashed features and query persistent
   ChromaDB collections with cosine distance.
6. **Follow-up run:** answer identical follow-up items under `N`, `R`, and `L`.
7. **Safety run:** answer unrelated items under all conditions.
8. **Analyse:** calculate repeated-error rate, retrieval hit rate, exact McNemar
   tests, and unrelated accuracy.
9. **Trace:** save the benchmark hash, selected config, memory records, per-item
   CSV, summary JSON, Markdown report, and every model/retrieval event.

## 5. Design decisions and reasons

### D-01: New implementation, no imported conclusions

The earlier exploratory folder was used only to understand the proposal and shared
WIP context. This repository does not copy its benchmark, scoring, result numbers,
or recommendations. This avoids anchoring the new experiment to an unverified
earlier run.

### D-02: One model family for v0

The same model answers questions and extracts lessons. This is not the final design
for controlling stylistic self-consistency, but it makes the first implementation
debuggable and stays within a low budget.

Configured model: `deepseek-flash` through the official DeepSeek API.

Reason: the official OpenAI-compatible endpoint supports the existing chat and
JSON-output paths. Candidate probes are not used as experimental evidence.

Observed probes:

- `deepseek-v3.2`: HTTP 403, unavailable to this gateway account.
- `spark-x2.5-1.7b`: returned empty content at the tested budgets.
- `qwen3.5-flash`: response latency was too high for a pilot and was stopped.
- `glm-4.5-air`: answered the current feedback set correctly.

### D-03: ChromaDB plus hashed lexical vectors

The proposal identifies ChromaDB as the primary experience store. v0 uses persistent
ChromaDB collections with a deterministic 2,048-dimensional hashing vectorizer,
word unigrams and bigrams, and cosine distance. Raw corrections and abstracted
lessons are kept in separate collections inside each run directory.

Reason: Chroma supplies inspectable local persistence while custom embeddings avoid
neural-weight downloads and API embedding calls. The vectors remain deterministic.
A sentence-transformer or hosted embedding backend is planned after the task
distribution is calibrated.

Trade-off: lexical overlap is weaker than semantic embedding. A retrieval miss may
therefore reflect the v0 backend rather than the experience-memory idea.

### D-04: Recorded correction text instead of an LLM user

Each feedback item contains a factually correct correction derived from its answer
and explanation. v0 uses that correction as the simulated user response.

Reason: an LLM user adds cost and can generate contradictory corrections. A fixed
correction keeps the learning signal constant while the memory format is tested.

Trade-off: the corrections are less varied and more polished than real human
feedback. A later robustness study should collect or generate independent user
corrections.

### D-05: One repeat in v0

The proposal asks for multiple repeats in the final evaluation. v0 uses one repeat
because it is an implementation pilot.

Consequence: McNemar values and percentage differences are diagnostic only. Do not
write final claims from a one-repeat run.

### D-06: Initial retrieval gate at similarity 0.08

The threshold is intentionally permissive so the minimal loop can be observed
before calibration. It is not a final parameter.

Next action: compute same-family hit rate and unrelated retrieval rate on feedback
items, then select `top_k` and threshold without using follow-up outcomes.

### D-07: No empty reply is treated as a wrong answer

During the real pilot, long reasoning prompts sometimes consumed the completion
budget before producing visible content (`finish_reason=length` and an empty
content string).

Resolution:

- empty replies are retried with 2x, 4x, ... completion allowances;
- they are never inserted into the answer cache;
- they never enter the accuracy calculation as an incorrect model answer;
- authentication/model-not-found errors (HTTP 401/403/404) fail fast.

This distinction matters: infrastructure failure is not evidence about model
reasoning.

### D-08: Honest stop when the benchmark has no errors

The first easy benchmark and the later hard benchmark were both solved perfectly by
the previously configured `deepseek-v4.1-flash` during feedback calibration. The runner records
`no_eligible_error_families` and stops.

Reason: without an observed error, there is no correction to learn from and no
repeat-error phenomenon to reduce. Creating lessons from correct answers would make
the outcome meaningless.

This is currently the main blocker. The next benchmark must be calibrated from
actual failures, not only made nominally more complex.

## 6. Files produced by a run

```text
results/runs/<run-id>/
├── config_used.json
├── benchmark.sha256
├── events.jsonl
├── summary.json              # written for completed or stopped calibration
├── per_item.csv              # completed runs only
├── report.md                 # completed runs only
└── memories.json             # completed runs only
```

`events.jsonl` is the source trace. The summary is derived from it.

## 7. Commands

```powershell
# Data and config only
python -m experience_agent validate

# No API calls; verifies every software component
python -m experience_agent run --offline --output results/runs/offline_smoke

# One real pilot
python -m experience_agent run --config config/experiment_v0.json

# Tests
python -m pytest
```

## 8. Troubleshooting

### `no_eligible_error_families`

The model is too strong for the current items. Inspect
`results/runs/<run-id>/events.jsonl`. Replace or extend a family only after
confirming that errors are stable enough to correct.

### Empty model reply

The client automatically increases the output budget. If all retries are empty,
lower prompt ambiguity, shorten the item, or choose a compatible model. Never mark
the empty reply as a wrong answer.

### HTTP 403

The model is not available to this account. Check the provider's `/models` list;
do not retry a permanent authorization failure repeatedly.

### Budget stop

All successful calls are cached in `results/cache/llm_cache.jsonl`. Rerunning the
same prompts reuses cached output. Raise the cap only after inspecting which new
calls are required.

### Retrieval hits unrelated questions

Inspect the score distribution before changing the cutoff. Calibrate on feedback
items and reserve follow-up items for evaluation. Consider metadata filtering only
after recording the unfiltered baseline.

## 9. Current limitations

- No eligible error family yet, so the real-model H1/H2 claims are not tested.
- Only one model, one repeat, and 32 items.
- Corrections are template-provided rather than independently collected.
- Embeddings are lexical, not semantic.
- Triage items are synthetic protocol exercises and are not medical advice.
- The safety split is small and currently measures retrieval presence rather than
  a full over-application judgement.
- Statistical analysis is descriptive; no confidence intervals or power analysis
  are included in v0.

## 10. Next milestones

1. Find a benchmark family with reproducible model errors; use feedback items only
   for construction and selection.
2. Freeze 6-10 feedback and 20+ follow-up items for each retained family.
3. Calibrate retrieval on feedback data and report hit rate before testing transfer.
4. Add at least three repeats and paired confidence intervals.
5. Compare a local sentence embedding with the hashed baseline.
6. Add an explicit applicability judgement and measure over-application.
7. Test a second model family only after the single-model pipeline is stable.

## 11. Security and Git hygiene

- Secrets belong in environment variables or the local credential store.
- `.env`, caches, raw generated runs, and editor files are Git-ignored.
- Commit benchmark definitions, configuration, code, tests, and reviewed summaries.
- Never commit a response cache that may contain sensitive prompts or completions.

