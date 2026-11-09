# E4b — Router Redesign on Re-Extracted Labels

**Date:** 2026-11-09 · **Author:** LongCat · **Status:** design / pre-registration
**Inputs:** `results/e5/checkpoints.jsonl` (400 outputs), `results/e11` (re-extraction), `results/e12` (naive rerouter), `experiments/04_router.py`, `experiments/12_router_rerun.py`

## 0. The label change voids E4's cost model

E11 re-grades the same 400 outputs with answer-position extraction: loop2 **0.460**, loop4 **0.465** (was 0.225/0.180). The old router's arithmetic — upside 0.110, downside 0.155, break-even `f = 0.71·r` — was derived from the broken labels and no longer applies. The re-extracted 4-cell table:

| cell | count |
|---|---:|
| both correct | 76 |
| neither | 91 |
| only loop2 | 16 |
| only loop4 | 17 |

New economics: upside = 17/200 = **0.085**, downside = 16/200 = **0.080** — symmetric. Break-even `f = 1.06·r`; escalate iff `P(only4|x)/P(only2|x) > 0.94`. Each decision is easier; total headroom is smaller.

**Does the oracle gain survive?** Yes, shrunk. Oracle = 109/200 = **0.545**, gain **+0.085** vs loop2 (was +0.110). At content level (target present anywhere) the cells are both 138 / only2 6 / only4 10, so content oracle = 0.770 and the content gain is only **+0.020**. The graded gain is 4× the content gain; 55/200 loop2 and 57/200 loop4 outputs contain the target yet are still graded wrong. Residual extraction failure, not reasoning, dominates.

**Are the loops distinguishable?** No. loop4 − loop2 = +0.005, McNemar p = 1.000, paired bootstrap 95% CI [−0.05, +0.06]. The loops are statistically tied; loop2 is 1.7× cheaper (2.89 vs 4.93 s/prompt). The default is loop2, and a router must beat it on accuracy *and* cost.

## 1. The first rerouter (E12) and a pilot

E12 (`results/e12`) reruns the E4 binary LR on the 5 frozen features with a fixed 0.5 threshold, 5-fold OOF: net **0.475**, escalates **86/200 (43%)**, FP 78. Its headline "oracle capture 47.06%" is **recall** (8/17), not net capture — net capture is (0.475−0.460)/0.085 = **17.6%**. This repeats the original E4's metric error. E12 has no held-out test, cost analysis, or family baseline.

Pilot on the same labels (numpy LR, OOF):

- 5-feature multinomial LR over 4 cells: net **0.465** (ties static loop4); best optimistic threshold 0.470.
- Family-conditional LR: combined **0.455** — worse than the family-only rule (GSM8K 0.62 vs static 0.66; MATH 0.29 vs 0.31). n=200 cannot support within-family learning.
- Binary mixed-cell LR: best **0.475** (r=0.471, f=0.312).
- Length × digit-density heuristic: best **0.475**.
- **Family-only rule** (escalate MATH, loop2 GSM8K): net **0.485**, +0.025 vs loop2, r=0.588, f=0.312, escalation 50%; McNemar p=0.302, CI [−0.01, +0.065].

No learned router beats the family rule at n=200. The signal is the family split: GSM8K loop2 0.660 > loop4 0.620; MATH loop2 0.260 < loop4 0.310; `task_type` proxies it (GSM8K 0.83 vs MATH 0.15).

## 2. The economic verdict: loop2 still wins

| strategy | acc | esc | s/prompt | s/correct | correct/1k s |
|---|---:|---:|---:|---:|---:|
| loop2 | 0.460 | 0 | 2.89 | **6.28** | **159** |
| loop4 | 0.465 | 1.0 | 4.93 | 10.60 | 94 |
| family rule | 0.485 | 0.50 | 3.91 | 8.06 | 124 |

Cost-parity with loop2 requires `acc ≥ 0.460 + 0.325·e`. At e=0.15 the router needs 0.509; at e=0.30 it needs 0.557 — above the oracle (0.545). **A router escalating ≥30% cannot beat loop2 on cost-per-correct even at the oracle.** The family rule, the strongest candidate, fails this bar. Unless a low-escalation rule clears both accuracy and cost-parity, the correct decision is static loop2.

## 3. Design

**Objective and action space.** Default **loop2**; action {loop2, loop4} with a family-conditional prior. E9 (loop1) is unrun and gates any expansion to {1,2,4}; E4b does not wait on it.

**Features (frozen before training, ≤5 prompt + family).** Keep the E4 five (`log_len`, `digit_density`, `math_symbol`, `question_marks`, `task_type`) and add **family** (or `task_type` as a proxy) as a first-class feature and baseline. The old spec's "family is diagnostic only" stance is falsified: family dominates. Within-family prompt features are underpowered at n=200 and must not be selected on the test set.

**Training.** Multinomial LR over the 4 cells (not a binary escalation label), class weights, standardized; stratified 160/40 split; inner 5-fold CV for the threshold, tuned on train under `0.085·P(only4|x) − 0.080·P(only2|x) > 0`; Platt calibration; report ECE and per-fold spread. Model < 10 KB, CPU, < 1 ms.

**Baselines to beat (head-to-head, McNemar).** loop2 (0.460), loop4 (0.465), family-conditional (0.485), length × digit heuristic (0.475), E12 (0.475).

**Evaluation.** Untouched 40-prompt test, touched once. Paired bootstrap (1,000) 95% CI on `acc − 0.460`; McNemar vs each baseline. **Primary metric: net capture = (acc − 0.460)/0.085**, never recall. Report graded and content-level oracle side by side; cost per correct and correct per 1k s; escalation rate, r, f, 4-cell confusion, ECE.

**Kill criteria (pre-registered).**

1. Net capture < 0.50 (net < 0.5025) → revert to loop2.
2. 95% CI of `acc − 0.460` includes 0 or is negative → revert.
3. Net < 0.460 → immediate revert, no canary.
4. Family-conditional baseline (0.485) beats the learned router head-to-head → ship the family rule (or loop2 if family is unavailable at inference).
5. Cost per correct worse than loop2 (6.28 s) → revert regardless of accuracy.
6. Escalation rate > 30% with no accuracy-dominated Pareto point → revert (cost-parity is impossible past e≈0.26).
7. ECE > 0.10 → no ship without recalibration.
8. 0.50 ≤ capture < 0.70 → ship behind a 10% flag, keep collecting labels.

## 4. Sequencing and deliverables

E4b is **CPU-only, 0 GPU-h**: the labels already exist. Pre-flight: freeze the grading protocol. E11's lenient answer-position grader is a diagnostic, not a correction (`research/e11_critique.md`); run the router on E11 labels as primary and on a strict grader (require `\boxed{}` / final-line `Answer:`, count truncation as failure) as sensitivity. If the strict grader collapses the oracle below +0.03, kill E4b and spend the effort on decoding fixes (stop sequences, repetition penalty, larger `max_tokens`) — the 55–57/200 residual extraction gap dwarfs the router's +0.085.

Deliverables: `results/e4b/checkpoints.jsonl`, `results/e4b/results.json`, `results/e4b/summary.md`; update `REPORT.md`.
