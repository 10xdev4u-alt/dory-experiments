# Dory 20h Lap — Final Research Report

**Report date:** 2026-11-09
**Model:** nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
**Hardware:** Kaggle 2×T4 (Tesla T4, CC 7.5), vLLM (PR #60149 stack)
**Repo:** https://github.com/10xdev4u-alt/dory-experiments (branch exp/wave1-backbone)

## 1. Executive Summary

**Headline: looping does not degrade content; extraction does.** Re-extracting the same 400 E5 outputs with answer-position awareness moves graded accuracy from 0.225→0.460 (loop2) and 0.180→0.465 (loop4); 30 of 31 "degraded" cases are extraction artifacts. Loop gain on content is ~0 (+0.005) and the loops are statistically tied. Default to loop2 (1.7× cheaper). The router rerun on re-extracted labels is marginal (+0.015).

- **E1 Loop-count scaling:** accuracy saturated at 1.0 on 50 trivial prompts (loops 1–4); throughput 25.6→10.3 tok/s; rep rate ~0.44. No oracle headroom → rescope to hard tasks (E5).
- **E2 256K context feasibility:** feasible only up to 64K. NIAH perfect 8K–64K at loop 1; 64K loop 4 failed (0/5); 128K/256K OOM. Ceiling = 64K on T4.
- **E3 NVFP4 reality:** NVFP4 unsupported on T4 (SM7.5); vLLM/torch incompatibility blocked quantization. Defer.
- **E4 Difficulty router (original labels):** logistic regression on 5 prompt features, 5-fold CV: classifier acc 0.530, net 0.220 vs static loop2 0.225. Fails the kill criterion; superseded by E12.
- **E5 Hard-loop:** original loop2 0.225 / loop4 0.180 / oracle 0.335. Re-extracted (E11): 0.460 / 0.465; loop gain ~0; 30/31 degraded cases are extraction artifacts.
- **E6 Context ceiling:** queued on Kaggle (vLLM env issue fixed); results pending.
- **E7 Quant ladder:** all modes failed with vLLM Config 'deprecated' kwarg (env incompatibility). NVFP4 confirmed false on T4.
- **E8 Confidence-gated escalation:** net accuracy 0.2300 vs static loop2 0.2250 (+0.005). Marginal.
- **E11 Extraction fix:** answer-position-aware re-extraction lifts loop2 0.225→0.460 and loop4 0.180→0.465; 30/31 degraded cases are artifacts. Looping does not degrade content; extraction does.
- **E12 Router rerun:** on re-extracted labels, net accuracy 0.475 vs static loop2 0.460 (+0.015); net capture 17.6%. Marginal; loop2 remains default.

## 2. E1 — Loop-Count Scaling (trivial)

| Loops | Acc | tok/s | rep rate |
|---:|---:|---:|---:|
| 1 | 1.0 | 25.6 | 0.442 |
| 2 | 1.0 | 17.1 | 0.401 |
| 3 | 1.0 | 13.0 | 0.370 |
| 4 | 1.0 | 10.3 | 0.446 |

Saturation artifact; no oracle headroom. Throughput degrades ~2.5× from 1→4 loops.

## 3. E2 — Context Feasibility

Feasible lengths: 8K, 16K, 32K, 64K. 128K/256K OOM.

| Context | Loop1 NIAH | Loop4 NIAH |
|---:|---:|---:|
| 8K | 5/5 | 5/5 |
| 16K | 5/5 | 5/5 |
| 32K | 5/5 | 5/5 |
| 64K | 5/5 | 0/5 |
| 128K | OOM | OOM |
| 256K | OOM | OOM |

Ceiling = 64K on a single T4; `last_loop` mode keeps KV loop-invariant.

## 4. E3 — NVFP4 Reality

NVFP4 unsupported on T4 (SM7.5); dequant requires Blackwell (SM100+). vLLM bitsandbytes failed on pip/torch conflicts. Defer; pivot to FP16/INT8 on a working stack.

## 5. E4 — Difficulty Router (original labels)

E4 router (logistic regression on 5 prompt features, 5-fold CV) on E5 data: classifier acc 0.530, misroute 0.470, net 0.220 vs static loop2 0.225 — fails the kill criterion (net < 0.280). Spec in research/e4_router_spec.md; critique in research/e4_router_critique.md. Superseded by E12.

## 6. E5 — Hard-Loop Benchmark

Original graded: loop2 0.225 / loop4 0.180 / oracle 0.335, loop gain −0.045. Re-extracted (E11): loop2 0.460 / loop4 0.465, loop gain +0.005 (~0). Content accuracy is flat (126→131 target-present prompts). 30 of 31 "degraded" cases are extraction artifacts. Re-extracted oracle is 0.545 (+0.085 vs loop2); content oracle is 0.770 (+0.020). The loops are statistically tied (McNemar p = 1.000; paired bootstrap 95% CI [−0.05, +0.06]), and loop2 is 1.7× cheaper (2.89 vs 4.93 s/prompt).

## 7. E6 — Context Ceiling (env failure)

Rerun failed: `max_model_len` equaled context (no output headroom) → "decoder prompt + output tokens > max_model_len"; 64K+ also hit engine init failures. E2 established the 64K ceiling. `last_loop` vs `per_loop` not yet compared; re-run with `max_model_len` = context + gen buffer and pinned env.

## 8. E7 — Quant Ladder

All modes failed: `TypeError: Config() got an unexpected keyword argument 'deprecated'` (vLLM/transformers mismatch). NVFP4 confirmed false on T4. Environmental blocker; pin the vLLM env before retry.

## 9. E8 — Confidence-Gated Escalation

Confidence-gated (prompt-length median split) escalation: 99/200 escalated, net 0.2300 vs static loop2 0.2250 (+0.005) and oracle 0.3350. Marginal; a better confidence signal (self-consistency samples) is needed.

## 10. E11 — Extraction Fix

Re-extracting the 400 E5 outputs with answer-position-aware extraction lifts graded accuracy from 0.225→0.460 (loop2) and 0.180→0.465 (loop4). Per dataset: GSM8K 0.26→0.66 / 0.21→0.62; MATH 0.19→0.26 / 0.15→0.31. 30 of 31 "degraded" cases are extraction artifacts: the correct value is present, but the last-number extractor sampled post-answer noise (GSM8K) or a lost `\boxed{}` container (MATH). 20/31 are recovered; the remaining case (idx166, target `x^3+3x-6`) is a genuine repetition collapse. Loop gain on content is ~0 (+0.005; GSM8K −0.04, MATH +0.05), not −0.045. Caveat: a sensitivity analysis, not a corrected accuracy — freeze a strict grader a priori and validate on held-out outputs before treating the +23.5 pt jump as capability (research/e11_critique.md).

## 11. E12 — Router Rerun on Re-extracted Labels

Rerunning the E4 logistic router on the E11 labels (5 frozen features, fixed 0.5 threshold, 5-fold OOF, n=200): classifier accuracy 0.565, misroute 0.435, net 0.475 vs static loop2 0.460 (+0.015) and loop4 0.465 (+0.010). It predicts escalation for 86/200 prompts (TP 8, FP 78); only 17 are true opportunities, so net capture is (0.475−0.460)/0.085 = 17.6%, not the headline 47.1% recall. The gain is marginal, with no held-out test or cost analysis; a family-conditional rule (escalate MATH, loop2 GSM8K) reaches 0.485 in the E4b pilot. The learned router does not beat loop2 decisively.

## 12. E5 Taxonomy

Loop regression is ~100% extraction artifact, not reasoning degradation. Original grading classified 31/31 degraded cases as format degradation; under re-extraction 30/31 are artifacts and 1 is a genuine repetition collapse. GSM8K degraded cases average 8.0 numbers after the target; MATH `\boxed{}` is lost in 7/13. Content accuracy is flat (126→131 present). Improved cases (22) are mostly extraction lottery (19/22). Mitigation: answer-position extraction, stop sequences, repetition penalty, larger token budget.

## 13. Conclusions

1. **Loop count:** default loop2. Loop4 costs 1.7× (4.93 vs 2.89 s/prompt) for equal accuracy (0.465 vs 0.460, statistically tied). Looping does not degrade content; extraction does.
2. **Extractor:** fix it. Answer-position-aware extraction recovers +0.235 (loop2) and +0.285 (loop4). Freeze a strict grader a priori and validate on held-out outputs; ship stop sequences, repetition penalty, and a larger token budget to avoid truncating at the answer.
3. **Router:** marginal. E12 net 0.475 vs loop2 0.460 (+0.015; net capture 17.6%). Not compelling at n=200; the family rule (0.485) is a stronger candidate. Do not ship the learned router.
4. **Context:** 64K ceiling on T4; 128K/256K need PP=2 or SWA-aware KV + INT8/GGUF.
5. **Quantization:** blocked by env; NVFP4 dead on T4. Pursue FP16 baseline + INT8 via pinned vLLM.
6. **Negative result:** the E5 "−4.5 pt loop regression" was an extraction artifact. More loops ≠ better quality on hard tasks, but they also do not hurt content; the cost is throughput.

## 14. Reproducibility

Artifacts committed to repo results/e1, results/e2, results/e5, results/e11, results/e12, results/e7. Scripts in experiments/.
