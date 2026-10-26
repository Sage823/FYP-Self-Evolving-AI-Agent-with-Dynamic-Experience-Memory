# Retrieval Calibration — 2026-10-26

## Protocol

- Benchmark: `data/benchmark_bbh_v3.jsonl`
- Benchmark SHA-256: `b59c0345592e59ce2118dd1a7d14c14d128a7b5cb6670f6c750375e17fd68293`
- Memory source: 8 feedback items
- Positive queries: 8 independent validation items
- Negative queries: 8 unrelated items
- Follow-up used during calibration: no
- Candidate top-k: 1, 2, 3, 5
- Candidate thresholds: 0.00 through 1.00 in 0.05 steps
- Maximum accepted false-positive rate: 0.20

The memory-side lessons are deterministic proxies extracted from the feedback
corrections' “General method” text. This permits a cheap backend comparison
without making LLM calls. It must be repeated with the actual extracted lessons
before freezing the final experiment.

## Result

| Backend | Memory | Hit rate | P@k | Top-1 family | FPR | top-k | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| hashing | raw correction | 25.0% | 100.0% | 25.0% | 0.0% | 1 | 0.25 |
| hashing | abstract lesson | 25.0% | 100.0% | 25.0% | 0.0% | 1 | 0.20 |
| MiniLM | raw correction | 100.0% | 100.0% | 100.0% | 0.0% | 1 | 0.45 |
| MiniLM | abstract lesson | 100.0% | 100.0% | 100.0% | 0.0% | 1 | 0.35 |

## Interpretation

The hashed lexical backend retrieves the correct family for only two of eight
validation questions. MiniLM retrieves the correct family for all eight and does
not return any above-threshold memory for the eight unrelated questions.

This establishes a clear retrieval-layer difference, but it is not yet evidence
that semantic memory improves repeated-error rates. That question still requires
a fresh model run with the response-status fixes and the calibrated semantic
configuration.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m experience_agent.calibration `
  --benchmark data\benchmark_bbh_v3.jsonl `
  --output results\calibration\retrieval_calibration.json `
  --backends hashing,sentence_transformers `
  --top-k 1,2,3,5 --threshold-step 0.05
```

