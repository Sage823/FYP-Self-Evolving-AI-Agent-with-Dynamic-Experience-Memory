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
| baseline | 8 | select error families without using feedback or follow-up |
| feedback | 8 | measure initial errors and create memories |
| validation | 8 | compare embeddings and calibrate retrieval only |
| follow-up | 16 | measure whether the same error family recurs |
| unrelated | 8 | basic safety check |

The v3 protocol has four error families with two baseline, two feedback, two
validation, and four follow-up items each. All question text is unique across
splits. Follow-up items are reserved for evaluation and are never used to choose
error families, embedding backends, thresholds, or `top_k`.

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
2. **Baseline:** answer the independent baseline split with no memory.
3. **Select families:** retain families whose complete-wrong rate reaches the
   configured threshold. Truncated, empty, malformed, and API-error responses are
   excluded from the denominator.
4. **Feedback:** answer the feedback split with no memory.
5. **Create memory:** store complete-wrong item corrections under `R`; ask the model for a
   general rule and store it under `L`.
6. **Calibrate retrieval:** on the validation split, compare hashing and
   sentence-transformers embeddings across thresholds and `top_k`. Use unrelated
   items as negatives. Never use follow-up outcomes.
7. **Retrieve:** encode a query with the selected backend and query persistent
   ChromaDB collections with cosine distance.
8. **Follow-up run:** answer identical follow-up items under `N`, `R`, and `L`.
9. **Safety run:** answer unrelated items under all conditions.
10. **Analyse:** calculate repeated-error rate, retrieval hit rate, exact McNemar
   tests, and unrelated accuracy.
11. **Trace:** save the benchmark hash, selected config, memory records, status counts, per-item
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

### D-09: Complete answers are separated from infrastructure failures

Every answer now receives one of:

```text
COMPLETE_CORRECT
COMPLETE_WRONG
TRUNCATED
EMPTY
FORMAT_ERROR
API_ERROR
```

Only the first two enter accuracy and repeated-error rates. A `finish_reason=length`
reply is retried with a larger budget and is never cached. If it remains
truncated, it is excluded from the analysis and counted separately.

Reason: the v2 pilot mixed genuine wrong answers with output cutoffs. That creates
a false apparent effect and can make memory look helpful merely because one
condition produces shorter responses.

### D-10: Baseline, feedback, validation, and follow-up are independent

Family selection uses the baseline split. Corrections and memories come from the
feedback split. Retrieval calibration uses feedback memories plus validation
queries and unrelated negatives. The follow-up split is used only after all
choices are frozen.

Reason: selecting families and retrieval thresholds on the same items used to
evaluate transfer leaks test information into the design.

### D-11: Hashing and semantic embeddings share one interface

`embedding.py` provides both:

```text
hashing:
sentence_transformers:
```

Both return L2-normalised vectors to ChromaDB. The semantic backend is optional
because PyTorch and model weights are much larger than the hash baseline.

`calibration.py` compares thresholds and `top_k` values for raw corrections and
abstract lessons separately. This is necessary because their texts have different
similarity distributions.

The reviewed October calibration is in
`docs/RETRIEVAL_CALIBRATION_2026-10-26.md`. On that development split, hashing
achieved 25% family-level retrieval hit rate, while MiniLM achieved 100%, with
zero unrelated false positives. This is an engineering result, not yet an
experimental memory effect.

### D-12: Follow-up expansion freezes all development splits

Increasing the follow-up count with the original generator would consume
additional random numbers before later families and change their baseline,
feedback, and validation items. The expanded generator therefore reads the frozen
v3 file, keeps every non-follow-up item unchanged, and draws fixed-seed follow-up
items from independent generator runs.

The resulting benchmark has 112 items: 8 baseline, 8 feedback, 8 validation,
80 follow-up, and 8 unrelated. Only the eligible B1 family ran during the
2026-10-27 experiment.

The reviewed result is
`docs/V3_EXPANDED_RESULTS_2026-10-27.md`. With two truncated answers excluded,
the N, R, and L repeated-error rates were 63.2%, 36.8%, and 35.0%. The result
shows a descriptive improvement after expansion, but one run and one family
remain insufficient for a final claim.

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

