# Expanded v3 Follow-up Result

Date: 2026-10-27

## Purpose

The first clean v3 run had only four follow-up items. The diagnostic implementation
increased the eligible family's follow-up set without changing the frozen
baseline, feedback, validation, unrelated items, model, embedding, thresholds, or
top-k.

## Data

- Benchmark: `data/benchmark_bbh_v3_expanded.jsonl`
- SHA-256: `bf6cd0ed82553a5222a8cc8ad347d276542f720e136d13508ee3684aae73a128`
- Total items: 112
- Baseline: 8
- Feedback: 8
- Validation: 8
- Follow-up: 80 (20 per family)
- Unrelated: 8

Only `B1_state_tracking` passed the frozen baseline threshold. The follow-up
evaluation therefore used the 20 B1 questions; the other 60 follow-up items exist
for later studies but were not run in this experiment.

## Run

Local generated run directory:

```text
results/runs/v3_expanded_semantic_20261027
```

- Model: `deepseek-flash`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Raw correction threshold / top-k: 0.45 / 1
- Abstract lesson threshold / top-k: 0.35 / 1
- Repeats: 1

## Result validity

The 100 recorded answer attempts produced:

| Status | Count |
|---|---:|
| `COMPLETE_CORRECT` | 70 |
| `COMPLETE_WRONG` | 28 |
| `TRUNCATED` | 2 |
| `EMPTY` | 0 |
| `FORMAT_ERROR` | 0 |
| `API_ERROR` | 0 |

The two truncated answers were one no-memory answer and one raw-correction answer.
They were excluded from the analysis denominator. The abstract-memory condition
had 20 valid answers. No-memory and raw-correction each had 19 valid answers.

## Main result

| Condition | Valid items | Repeated errors | Repeated-error rate | Retrieval hit rate |
|---|---:|---:|---:|---:|
| N: no memory | 19 | 12 | 63.2% | 0% |
| R: raw correction | 19 | 7 | 36.8% | 100% |
| L: abstract lesson | 20 | 7 | 35.0% | 100% |

Paired contrasts:

| Contrast | Risk difference | Relative reduction | Exact McNemar p |
|---|---:|---:|---:|
| R vs N | -33.3 pp | 50.0% | 0.070 |
| L vs N | -26.3 pp | 41.7% | 0.180 |
| L vs R | -5.3 pp | 14.3% | 1.000 |

Eight unrelated questions were answered correctly under all three conditions,
and no memory was retrieved for them. No memory-safety degradation was observed.

## Interpretation

Before expansion, all three conditions had a 50% repeated-error rate on four
items. After expansion, both memory conditions had substantially fewer repeated
errors than no memory. This suggests the earlier null aggregate was strongly
affected by the four-item sample.

The result does not establish that abstraction is better than the raw correction.
The raw and abstract conditions differed by only one error count in the paired
sample and the McNemar comparison was not significant.

The current evidence remains descriptive because:

- only one error family passed baseline;
- only one model was evaluated;
- the experiment was run once;
- two truncated responses changed the valid denominator;
- the benchmark family is long and attention-heavy.

## Traceability

The local run directory contains:

```text
config_used.json
benchmark.sha256
events.jsonl
per_item.csv
summary.json
report.md
memories.json
chroma/
```

The source benchmark, generator, configuration, tests, and review summary are
versioned in Git. Raw API caches and Chroma databases remain local because they
contain untracked prompts and binary state.

