# October 2026 Monthly Progress Report Notes

This is a draft aligned to `monthly_report/Monthly_Report_Form.docx`. Fill the
student IDs and meeting details from official records before submission.

## Header

- Project code: LXM1
- Supervisor: Prof. LI Xiaomeng
- Project title: Self-Evolving AI Agent with Dynamic Experience Memory
- Group members: WANG Zijing; GUO Ruxin
- Student IDs: to be filled from the submitted project form
- Reporting period: October 2026
- Report number: #1 (October)

## Progress

This month we turned the proposal into a clean, independently implemented
experimental workspace and completed the first end-to-end version of the learning
loop. The current system can answer a question with no memory, receive a correction
after an error, store either the raw correction or an extracted rule in separate
persistent ChromaDB collections, retrieve relevant records using deterministic local
hashed embeddings, and answer new questions with the retrieved experience.

We created a 32-item pilot benchmark covering constraint ordering, ordered decision
rules, conditional pricing, and synthetic protocol application. The benchmark has
separate feedback, follow-up, and unrelated-question splits. We also implemented
automatic scoring and a paired analysis of repeated-error rates, retrieval hit
rate, and unrelated-question accuracy. Fourteen automated tests now pass, including a
complete no-API run of the answer-correct-learn-retrieve loop.

The implementation includes response caching, a per-run API budget, benchmark
integrity checks, and detailed event logs. Real-model testing on the official
DeepSeek API with `deepseek-flash` revealed that the model solved all current feedback items,
so the workflow correctly stopped before creating lessons from correct answers.

We then generated a harder BBH-inspired benchmark with programmatic answer oracles.
After an initial calibration remained too easy, the second version triggered six
real feedback errors and completed the full correction-memory-transfer loop. Raw
correction memory reduced the observed repeated-error rate from 56.25% to 37.5%;
abstracted lessons did not improve the aggregate result, and the small paired test
was not statistically significant. This is pilot evidence for a functioning
workflow, not a final research conclusion.

## Major difficulties

1. The initial benchmark was too easy for the available model. Making the puzzles
   longer improved implementation realism but did not produce a stable error
   distribution.
2. Some reasoning-model responses consumed the completion budget without producing
   visible text. We now retry these calls with a larger budget and never count or
   cache an empty response as a model error.
3. The current lexical embedding baseline is inexpensive but may miss semantic
   matches. Retrieval calibration is therefore the next necessary step once an
   error-bearing task family is available.
4. Several candidate models were unavailable, returned empty content, or were too
   slow for a low-cost pilot. We retained one reachable model for this milestone.

## Overall progress

The software pipeline and evaluation harness are operational. The main research
risk is now benchmark validity, not missing infrastructure: we need tasks on which
the configured model makes reproducible errors before H1 and H2 can be tested.

## Future plan

1. Construct and calibrate a harder task set from observed model failures.
2. Freeze the selected feedback and follow-up sets before the main run.
3. Calibrate retrieval threshold and `top_k` on feedback items.
4. Run the three conditions with at least three repeats.
5. Add paired confidence intervals and a stronger unrelated-question safety test.
6. Discuss the revised task distribution and model choice with the supervisor.

## Meeting fields

- Meeting date and time: to be confirmed
- Group representative signature: to be signed
- Supervisor comments and grade: reserved for supervisor

