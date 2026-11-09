# E4 Difficulty Router Spec — Dory on 2×T4

**Version 2** (repairs `results/e4` v0). Model: Dory1.0-4B. Hardware: Kaggle 2×T4.
Data: `results/e5/checkpoints.jsonl` — 200 prompts (GSM8K + MATH), loops 2 and 4, per-prompt `correct`.

## 0. Why v0 failed

v0 trained a binary "escalate to loop4" classifier on 9 features with a fixed 0.5 threshold.
Out-of-fold: TP 14, FP 86, FN 8, TN 92. It reported **oracle capture 63.6%** — that is recall on
the 22 improvement prompts and ignores the 86 false escalations. Net accuracy was **0.220**, below
static loop2 (0.225). Under the E5 asymmetry (0.110 upside vs 0.155 downside) the router must be
judged on net accuracy, not recall. v0 is killed by the criteria in §8.

## 1. Objective and break-even

Per-prompt cells on the 200: both 14 (0.070), neither 133 (0.665), only-loop2 31 (0.155),
only-loop4 22 (0.110). Let `r` = recall on only-loop4 and `f` = false-escalation rate on
only-loop2:

```
acc  = 0.225 + 0.110*r - 0.155*f
break-even vs loop2: f = 0.71*r
net oracle capture = (acc - 0.225) / 0.110 = r - 1.409*f
```

The router may false-escalate on only-loop2 at 71% of its true-escalation rate and still break even,
but no higher. Decision rule: escalate iff `0.110*P(only4|x) - 0.155*P(only2|x) > 0`, i.e.
`P(only4|x) / P(only2|x) > 1.41`. Routing 4 on both/neither cells changes cost, not accuracy.

## 2. Data and labels

Label = 4-cell class from (`loop2_correct`, `loop4_correct`). Split: stratified 160/40 train/test
by cell; the test set stays untouched until final evaluation. Thresholds and hyperparameters are
tuned on train via inner 5-fold CV, and per-fold spread is reported, not just the mean. With 22
positives the test set holds only ~4–5 only-loop4 prompts, so every test claim carries a bootstrap
CI. `max_tokens` stays fixed per family (math 512, bbh 256) in v1; a token-budget head is deferred
because it interacts with loop count and the cost model.

## 3. Features (frozen, 5)

Prompt-derived only, computed in <1 ms, no model call. Frozen before training — no embeddings,
no TF-IDF, no perplexity, no open-ended feature hunting against 200 labels.

1. `log_len` = log1p(n_chars)
2. `digit_density` = n_digits / n_chars
3. `math_symbol` = 1 if any of `= + - × ÷ $ \` or `\frac` is present
4. `question_marks` = count of `?`
5. `task_type` = 1 if word-problem pattern (digit present AND one of how many / how much / total /
   each / sells / buys), else 0

Dataset family (gsm8k/math) is recorded as a diagnostic only. If family is the strongest signal,
the router does not generalize and is not shipped.

## 4. Model and decision rule

Primary: multinomial logistic regression over the 4 cells, L2, inverse-frequency class weights,
standardized features. It produces `P(only4|x)` and `P(only2|x)` directly for the §1 rule.
Secondary: binary LR on the 53 mixed-cell prompts (22 only4 / 31 only2) — the smallest problem that
contains the whole decision. Optional capacity-capped MLP (one hidden layer, 8 units, dropout 0.2)
only if it beats LR on nested CV; default is LR. Model size < 10 KB, inference < 1 ms CPU.

Threshold: tuned on train to maximize expected net accuracy under the §1 rule, confirmed once on
test. Platt calibration; report ECE.

## 5. Baselines to beat

1. Static loop2 (0.225) — the default and the kill target.
2. Static loop4 (0.180).
3. Oracle (0.335).
4. **Heuristic (default-to-beat):** escalate iff `digit_density > t1` and `log_len < t2`, tuned on
   train, evaluated untouched on test. If the learned router cannot beat the heuristic head-to-head
   (McNemar on per-prompt correctness deltas), ship the heuristic.
5. v0 router (0.220) — regression check.

## 6. Evaluation

Primary: net accuracy on the untouched test; oracle capture = (acc - 0.225) / 0.110.
Secondary: misroute rate (accuracy-misroutes only: only4→2 and only2→4), 4-cell confusion matrix,
calibration ECE/Brier, per-cell recall. Paired bootstrap (1,000 resamples over prompts) for 95% CIs;
McNemar vs loop2 and vs heuristic. Also report the accuracy/latency Pareto point: P(route4),
avg loops = 2 + 2*P(route4), avg latency vs static loop2.

## 7. Latency and cost

Router budget: < 50 ms p99 added latency, no GPU forward pass, no LLM call (v0's pure-Python LR
already satisfies this). Loop economics on 2×T4: loop2 = 577.3 s / 200 = 2.89 s per prompt;
loop4 = 985.7 s / 200 = 4.93 s per prompt; each avoided escalation saves ≈ 2.04 s. A router that
escalates on half the prompts (like v0) costs ≈ 2× loop2 latency and must be accuracy-dominated to
survive. Report cost per correct answer and correct answers per 1k seconds for loop2, loop4,
heuristic, and router.

## 8. Kill criteria (pre-registered)

Evaluate on the untouched test plus a 5% canary:

1. Net capture < 0.50 (net acc < 0.280) → revert to static loop2.
2. 95% CI of (acc_router - 0.225) includes 0 or is entirely negative → revert.
3. Net acc < 0.215 → immediate revert, no canary.
4. Heuristic beats the learned router head-to-head → ship heuristic, not model.
5. Router p99 added latency > 5% of loop2 latency → revert.
6. ECE > 0.10 → no ship without recalibration.

If 0.50 ≤ capture < 0.70: ship behind a flag at 10% traffic, keep collecting labels, retrain.
v0 fails criteria 1 and 3.

## 9. Required pre-flight: loop1 and the escalation arm

E5 never ran loop1 on the hard set. Before any 2-vs-4 claim: rerun loop1 on the same 200 prompts
(~300 s on 2×T4) and report loop1 accuracy and the 3-way oracle. If loop1 ≥ loop2 - 0.01, looping
itself is suspect on hard prompts and the action space must expand to {1, 2, 4} (difficulty gate to
loop1); otherwise the router optimizes the wrong margin.

Arm B (comparison, same criteria): gated escalation — always run loop2, then escalate to 4 only when
a cheap confidence signal (self-consistency between the two loop2 samples, or the E9 verifier score)
is below a calibrated threshold. Its worst case is loop2; the pre-router's worst case is worse than
loop2 on every prompt. Run both arms only if the loop1 gate passes.

## 10. Milestones

- **E4a:** freeze features, build the 4-cell dataset, tune heuristic, rerun loop1.
- **E4b:** train LR/MLP, calibrate, tune threshold on train, report inner-CV spread.
- **E4c:** untouched test eval, bootstrap CIs, cost/latency, baseline comparison.
- **E4d:** ship/kill decision per §8; write `results/e4` v2.
