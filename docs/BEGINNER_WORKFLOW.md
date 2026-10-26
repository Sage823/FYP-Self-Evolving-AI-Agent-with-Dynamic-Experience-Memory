# Beginner Workflow: Result Status, Baseline, and Retrieval Calibration

This guide explains the three changes made on 2026-10-26 and how to inspect,
test, and publish them from VS Code. Commands assume Windows PowerShell and the
repository root:

```text
D:\00-Ruth\00a-comp\000fyp-proposal\FYP-Self-Evolving-AI-Agent-with-Dynamic-Experience-Memory
```

## 1. What problem did these changes solve?

The earlier workflow had three weaknesses:

1. Any answer that differed from the gold answer was counted as wrong, even if
   the model had been cut off by the output-token limit.
2. Error families were selected on the same feedback items later used to create
   memory.
3. Retrieval used one permissive hash similarity setting, with no independent
   comparison against semantic embeddings.

The new workflow separates these concerns:

```text
baseline split    -> choose families that the model actually fails
feedback split    -> collect errors and create memory
validation split  -> compare embeddings and tune threshold / top-k
follow-up split   -> evaluate transfer AFTER calibration is frozen
unrelated split   -> measure false retrieval on questions with no applicable memory
```

The current v3 benchmark has 48 items:

| Split | Items | Purpose |
|---|---:|---|
| baseline | 8 | select eligible error families |
| feedback | 8 | obtain corrections and create memories |
| validation | 8 | calibrate retrieval only |
| follow-up | 16 | evaluate memory transfer |
| unrelated | 8 | negative retrieval queries |

Each of the four error families contributes:

```text
2 baseline + 2 feedback + 2 validation + 4 follow-up = 10 items
```

The generated file is `data/benchmark_bbh_v3.jsonl`. Its current SHA-256 is:

```text
b59c0345592e59ce2118dd1a7d14c14d128a7b5cb6670f6c750375e17fd68293
```

## 2. Generate or inspect the benchmark

The generator controls all question text and oracle answers. Re-running with the
same seed reproduces the file:

```powershell
python -m experience_agent.bbh_benchmark --difficulty 2 `
  --with-development-splits --output data/benchmark_bbh_v3.jsonl
```

Look at the first lines:

```powershell
Get-Content data/benchmark_bbh_v3.jsonl -TotalCount 5
```

Each line is one JSON item. Important fields:

- `id`: unique item ID.
- `family`: one of the four error families.
- `split`: baseline, feedback, validation, follow-up, or unrelated.
- `question`: text sent to the model.
- `answer`: oracle answer.
- `correction`: present only on feedback items.

Count the splits:

```powershell
python -m experience_agent validate --config config/experiment_bbh_v3.json
```

## 3. Where response classification lives

The classification is not one function. It is a chain of responsibilities:

| File | Responsibility |
|---|---|
| `src/experience_agent/llm.py` | Inspects `finish_reason`, retries incomplete replies, defines API/empty/truncated exceptions |
| `src/experience_agent/agent.py` | Converts a provider response into an `AnswerResult` with status |
| `src/experience_agent/experiment.py` | Records status and only learns from `COMPLETE_WRONG` |
| `src/experience_agent/analysis.py` | Excludes incomplete responses from rates |

The status values are:

| Status | Meaning | Enters error rate? |
|---|---|---|
| `COMPLETE_CORRECT` | Complete response, correct answer | Yes |
| `COMPLETE_WRONG` | Complete response, wrong answer | Yes |
| `TRUNCATED` | `finish_reason=length` after all larger-budget retries | No |
| `EMPTY` | No visible answer after retries | No |
| `FORMAT_ERROR` | Complete text but no final `Answer:` line | No |
| `API_ERROR` | Authentication, transport, or provider failure | No |

How the flow works:

1. `LLMClient.complete()` receives `finish_reason`.
2. A `length` response is not cached and is retried with twice the token budget.
3. Empty content is also retried and never cached.
4. If retries are exhausted, the client raises a typed exception.
5. `ExperienceAgent.answer()` catches that exception and returns an invalid
   `AnswerResult` rather than crashing the whole experiment.
6. The experiment records the status in `events.jsonl` and `per_item.csv`.
7. `analysis.py` calculates repeated-error rates only over complete answers.

The reason for separating these states is simple: a transport failure or token
cutoff is not evidence that the model reasoned incorrectly.

Inspect the exact implementation:

```powershell
Select-String -Path src/experience_agent/llm.py -Pattern "TruncatedResponseError|finish_reason"
Select-String -Path src/experience_agent/agent.py -Pattern "AnswerStatus|FORMAT_ERROR"
Select-String -Path src/experience_agent/analysis.py -Pattern "excluded_items|answer_valid"
```

## 4. Baseline selection

The experiment no longer chooses error families from the feedback items used to
create memory. It first answers the separate baseline split.

The relevant code is in `ExperimentRunner.run()`:

```text
baseline rows -> count COMPLETE_WRONG -> family error rate
             -> keep families with rate >= feedback_error_threshold
feedback rows -> only COMPLETE_WRONG rows can create memory
```

Invalid outputs are removed from the denominator. For example, if one baseline
item is truncated, it does not count as either correct or wrong.

Inspect the run summary:

```powershell
Get-Content results/runs/<run-id>/summary.json
```

Look for:

```json
{
  "selection_source": "independent_baseline_split",
  "selection_error_rate": {},
  "selection_status_counts": {},
  "feedback_status_counts": {}
}
```

If a family does not qualify on baseline, it never enters the follow-up run.

## 5. Why there is a validation split

Threshold tuning is a form of learning. If we tuned the threshold on the
follow-up questions, the reported follow-up result would no longer be independent.

The calibration therefore uses:

```text
memory source      = feedback
positive queries   = validation
negative queries   = unrelated
never used         = follow-up
```

For each backend, memory kind, threshold, and `top_k`, it reports:

- **Hit rate**: fraction of validation queries for which any retrieved item has
  the expected error family.
- **P@k**: proportion of retrieved memories that have the expected family.
- **Top-1 family**: whether the first result has the expected family.
- **FPR**: fraction of unrelated questions receiving at least one memory above
  threshold.

The selector first keeps options with false-positive rate at or below 0.20. It
then maximises hit rate, precision, and top-1 family accuracy.

## 6. Compare hashing and semantic embeddings

Install the optional semantic dependencies in the project environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,semantic]"
```

The main model is `sentence-transformers/all-MiniLM-L6-v2`. The backend is loaded
lazily, so ordinary tests and the hash workflow do not need PyTorch.

Run calibration:

```powershell
python -m experience_agent.calibration `
  --backends hashing,sentence_transformers `
  --top-k 1,2,3,5 `
  --threshold-step 0.05
```

Outputs:

```text
results/calibration/retrieval_calibration.json  # all candidate settings
results/calibration/retrieval_calibration.md    # short recommendation table
```

The complete JSON is generated data and is Git-ignored. A reviewed summary is in
`docs/RETRIEVAL_CALIBRATION_2026-10-26.md`.

The current validation comparison is striking:

| Backend | Memory | Hit rate | P@k | Top-1 family | FPR | top-k | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| hashing | raw correction | 25.0% | 100.0% | 25.0% | 0.0% | 1 | 0.25 |
| hashing | abstract lesson | 25.0% | 100.0% | 25.0% | 0.0% | 1 | 0.20 |
| MiniLM | raw correction | 100.0% | 100.0% | 100.0% | 0.0% | 1 | 0.45 |
| MiniLM | abstract lesson | 100.0% | 100.0% | 100.0% | 0.0% | 1 | 0.35 |

Interpretation: hash retrieval barely exceeds chance for the four families on
this validation set, while MiniLM retrieves the right family for all eight
positive queries without retrieving any memory on the eight negative queries.

This does not yet prove MiniLM will improve the final task result. It proves that
the retrieval layer is no longer an obvious confound.

The calibrated semantic settings are recorded in:

```text
config/experiment_bbh_v3_semantic.json
```

They are deliberately different for raw and abstract memory:

```json
{
  "raw_top_k": 1,
  "raw_min_similarity": 0.45,
  "lesson_top_k": 1,
  "lesson_min_similarity": 0.35
}
```

## 7. Run and inspect the experiment

Offline checks consume no API calls:

```powershell
python -m experience_agent run --offline `
  --config config/experiment_bbh_v3_semantic.json `
  --output results/runs/offline_v3_semantic
```

A real run uses the same command without `--offline`:

```powershell
$env:DEEPSEEK_API_KEY = "<your key>"
python -m experience_agent run `
  --config config/experiment_bbh_v3_semantic.json
```

For each run directory:

```text
events.jsonl       one machine-readable event per answer / memory operation
per_item.csv       one table row per recorded answer
summary.json       aggregates and status counts
report.md          human-readable report
memories.json      stored raw and abstract memory records
chroma/            persistent vector database
```

Check status counts first:

```powershell
python -c "import json; s=json.load(open('results/runs/<run-id>/summary.json')); print(s['all_result_status_counts'])"
```

Check excluded answers before looking at effects:

```powershell
python -c "import json; s=json.load(open('results/runs/<run-id>/summary.json')); print('excluded', s['excluded_answers'])"
```

If many answers are truncated, the experiment needs prompt/task repair before
interpreting any memory effect.

## 8. Run the test suite

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

The tests check:

- complete vs wrong vs truncated vs format status;
- no cache entry for permanent truncation;
- only complete answers enter repeated-error rates;
- independent baseline selection;
- Chroma persistence and embedding metadata;
- calibration excludes follow-up data;
- the v3 dataset has the expected split sizes.

## 9. Git version management with VS Code

Open the repository folder in VS Code, then open **Terminal → New Terminal**.
Make sure the prompt's current directory is the repository root.

### Pull the latest version before working

```powershell
git status
git pull --ff-only origin main
```

`git status` should ideally show a clean working tree. Untracked personal files
are allowed, but inspect them before adding them.

### Inspect changes before committing

```powershell
git status
git diff --stat
git diff
```

Files under `.venv/`, `results/runs/`, and `results/calibration/` are intentionally
ignored. They must not be committed.

### Stage and commit

Review the changed filenames and add only project files:

```powershell
git add README.md pyproject.toml .gitignore config data docs src tests
git status
git commit -m "Add result status, baseline, and retrieval calibration"
```

`git add .` also works in this repository because generated outputs and secrets
are ignored, but checking the staged list remains important:

```powershell
git diff --cached --stat
git diff --cached
```

### Push from the VS Code terminal

Yes, push from the integrated terminal by running:

```powershell
git push origin main
```

If the remote has new commits:

```powershell
git pull --rebase origin main
python -m pytest -q
git push origin main
```

After pushing, verify:

```powershell
git status
git log -3 --oneline
git ls-remote --heads origin main
```

Expected result:

- `git status` says `Your branch is up to date with 'origin/main'`.
- The latest local commit appears in `git log`.
- `git ls-remote` shows the same commit hash as local `HEAD`.

Also open the GitHub repository in a browser and check:

1. The newest commit is visible.
2. The **Actions** tab shows the test workflow passing.
3. `data/benchmark_bbh_v3.jsonl` and the new source/config/docs files are present.
4. No `.env`, API key, `.venv`, Chroma database, or generated result file is shown.

## 10. Important reporting caveat

The current calibration compares retrieval on a small eight-item validation
split and uses deterministic lesson proxies derived from feedback corrections.
After a real model run extracts its actual lessons, rerun calibration with those
lessons or regenerate this comparison before freezing the final threshold. The
current result is adequate for an engineering milestone, not a final paper claim.

