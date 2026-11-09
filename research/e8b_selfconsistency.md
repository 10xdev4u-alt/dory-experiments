# E8b: self-consistency-gated escalation

**Version 1.** Model: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`. Hardware: Kaggle 1xT4 (TP=1). Data: the verbatim 200 E5 prompts (`results/e5/checkpoints.jsonl`), paired by `idx`. Baselines: loop2 0.225 (2.89 s/prompt), loop4 0.180 (4.93 s/prompt), oracle 0.335, E8 v0 prompt-length gate 0.230 (+0.005). Blocked on E9's default choice.

## 0. Why E8 v0 failed

E8 v0 gated on prompt length (median split), escalated 99/200 (49.5%), and netted +0.005 over loop2, inside the noise at n=200. Prompt length is a difficulty proxy, not a confidence signal: it cannot see whether the model is uncertain about a specific prompt. E8b replaces it with a behavioral signal, 3 loop2 samples at T=0.7, escalate iff they disagree.

## 1. Hypothesis

H1: 3-sample agreement at T=0.7 separates the 22 loop4-winners from the 31 loop2-winners better than prompt length does. Escalating only on disagreement captures at least 50% of the +0.110 oracle gain at a 10% or lower false-escalation rate on loop2-winners, netting at least +0.05 (0.275) over loop2 and beating E8 v0 head-to-head.

H0 (rival, from the E5 taxonomy): the oracle gain is extraction variance, not reasoning instability. 19/22 improved cases contain the target in both loops (extraction lottery), 3/22 are truncation recovery, and 30/31 degraded cases still contain the target. If so, agreement will not separate winners from losers, and E8b lands at loop2. The test distinguishes them: r < 0.50 or f > 0.10 means H0 wins.

## 2. Method

Samples. 3 independent loop2 runs per prompt, temperature 0.7, top-p 0.9, fixed seeds, max_tokens 512 (GSM8K/MATH), TP=1, max_model_len 8192. The E5 greedy loop2 answer is the default; the 3 stochastic samples are the gate. Prompt identity and env manifest follow E9's pre-flight.

Agreement. Extract the final answer from each sample with the E5 grader unchanged (GSM8K last number; MATH `\boxed{}` then last number), normalize (strip units, commas, trailing zeros), and score agreement as the fraction matching the modal answer: 3/3 = 1.0, 2/3 = 0.667, all-different = 0.333. Escalate iff agreement < τ.

Threshold. τ is tuned on the 160-prompt train split (the E4 v2 stratified split) to maximize expected net accuracy under the E4 rule `0.110*r - 0.155*f`, with escalation capped at 40%. Candidates are τ = 0.667 (majority-tolerant) and τ = 1.0 (literal "disagree"). The 40-prompt test is touched once, after τ is frozen.

Two readouts, no extra GPU.

- E8b-gate: keep the E5 greedy loop2 answer; escalate iff agreement < τ. This isolates the gate under the E5 grading protocol.
- E8b-consensus: the non-escalated answer is the modal extracted answer; the escalated answer is the loop4 answer. This tests whether self-consistency alone recovers extraction-lottery cases.

Decisive diagnostic. Report the agreement distribution for only-loop4 vs only-loop2 prompts, r, f, escalation rate, and the taxonomy of escalated wins. If the gains are extraction lottery, fix the extractor, not the gate.

## 3. Metrics

Primary: net accuracy on the untouched test, bootstrap 95% CI (1,000 resamples) vs loop2 (0.225), McNemar vs E8 v0 (0.230). Secondary: net oracle capture `(acc - 0.225)/0.110`; r and f; escalation rate; agreement-AUC for only-loop4 vs only-loop2; cost per correct answer; correct answers per 1k s.

## 4. Cost: the 3x problem

Three loop2 samples cost 3 x 2.89 = 8.66 s/prompt, which is 3.0x loop2 and 1.76x static loop4 (4.93 s), before any escalation. Escalation adds 4.93 s, so the worst case is 13.59 s = 2.76x loop4 = 4.7x loop2. E8b cannot win on cost, only on accuracy, and it must clear the loop2 bar (0.280) by a margin large enough to justify 1.76x loop4. At equal accuracy, static loop4 is the better buy.

Is it worth it vs static loop4? Only if the accuracy margin is large; the E5 prior says it will not be. The upside is extraction variance, and the downside (loop4 on a loop2-winner) is also extraction variance, so r and f move together and the E4 break-even `f = 0.71*r` is hard to clear. Expect E8b to land between loop2 and the oracle with a cost floor above loop4, a bad Pareto point.

Cheaper variant, recommended if the 3-sample arm shows any signal. Adaptive 2-to-3: run 2 samples; if they agree, stop (5.77 s = 2x loop2 = 1.17x loop4); if they disagree, run a 3rd to break the tie and escalate iff there is no majority. This keeps the majority signal where it matters and cuts the floor to 1.17x loop4. If the 3rd sample adds under 2 pt over 2 samples, ship the adaptive variant or kill. The experiment reuses E5's greedy loop2, so the marginal GPU cost is about 0.3 h; production cost is 3 samples plus escalations and must be judged at that number.

## 5. Kill criteria (pre-registered)

1. Net capture < 0.50 (net < 0.280): revert to loop2 (or loop1 per E9).
2. 95% CI of `(acc - 0.225)` includes 0 or is negative: revert.
3. Net < 0.215: immediate revert, no canary.
4. E8 v0 heuristic (0.230) beats E8b head-to-head: ship the heuristic.
5. Test r < 0.50 or f > 0.10: the confidence signal is not real; kill the gate.
6. E8b p99 latency > 2x loop4 with no accuracy-dominated Pareto point: revert.
7. ECE > 0.10: no ship without recalibration.
8. 0.50 <= capture < 0.70: ship behind a 10% flag, keep collecting labels.

## 6. Deliverables

`results/e8b/checkpoints.jsonl`, `results/e8b/results.json`, `results/e8b/summary.md`, `results/e8b/env.json`; update `REPORT.md` and `research/next_wave_plan.md`. Non-goals: no loop2/loop4 re-run, no extractor change, no router retrain.
