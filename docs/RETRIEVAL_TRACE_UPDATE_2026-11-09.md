# Retrieval trace update (2026-11-09)

This change answers Dr. Chen's request to separate retrieval failures from answer failures. The previous runner recorded only memories that passed the similarity threshold. `ExperimentRunner._answer_and_record()` now keeps the ranked top-k candidates **before** the threshold and logs the final selected memories separately in both `per_item.csv` and `events.jsonl`.

| Field | Meaning |
|---|---|
| `candidate_ids`, `candidate_scores` | Top-k results before the configured cutoff. |
| `selected_ids`, `selected_scores` | Results that passed the cutoff and entered the answer prompt. |
| `top_k`, `similarity_threshold`, `retrieval_status` | Parameters and observable reason for selecting no memory or a same/other-family memory. |
| `retrieval_hit_basis` | Explicitly says the existing `retrieval_hit` metric is a **same-family proxy**, not an independently labelled item-memory applicability judgment. |

The change does not alter prompts, thresholds, or selected memories: the same top-k search runs first, and the configured threshold filters it afterward. Existing `retrieved_*` fields remain for older analysis code. A new regression test checks that a below-threshold candidate is logged but not injected. The complete suite passed: 26 tests. An offline Mock-model run at `results/runs/offline_retrieval_trace_20261109/` wrote 80 result rows and confirmed that candidate, score, threshold, selection, and status fields appear in the CSV and event log. Mock answers are pipeline checks, not model evidence.

## Why the monthly-report cache was not replayed

The monthly-report replay configuration points to `data/benchmark_bbh_v2.jsonl` and the shared `results/cache/llm_cache.jsonl`. A read-only preflight checked exact no-memory prompt cache keys for the configured model, temperature, and all three retry output limits. Coverage was **2/8 feedback, 2/16 follow-up, and 6/8 unrelated items**. These are necessary but insufficient conditions for a complete N/R/L replay. A complete cache-only replay would therefore require new provider requests for most items; it cannot show a clean “same answers, new trace” comparison. No cache replay or live API call was made for this change. Even a complete cache replay would be a logging check, not a new independent experiment.

## Original version and review

The original file is preserved in Git history at commit `f7ed599` (for example, `git show f7ed599:src/experience_agent/experiment.py`). This feature branch keeps the change reviewable without copying a second, potentially stale Python module into the package. Review the branch diff and merge it through the team's normal process.
