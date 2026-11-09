# Dory 20h Lap — Final Research Report

**Report date:** 2026-11-09
**Model:** nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
**Hardware:** Kaggle 2×T4 (Tesla T4, CC 7.5), vLLM (PR #60149 stack)
**Repo:** https://github.com/10xdev4u-alt/dory-experiments (branch exp/wave1-backbone)

## 1. Executive Summary

- **E1 Loop-count scaling:** accuracy saturated at 1.0 on 50 trivial prompts (loops 1-4); throughput 25.6→10.3 tok/s; rep rate ~0.44. No oracle headroom → rescope to hard tasks (E5).
- **E2 256K context feasibility:** feasible only up to 64K. NIAH perfect 8K-64K at loop 1; 64K loop 4 failed (0/5); 128K/256K OOM. Ceiling = 64K on T4.
- **E3 NVFP4 reality:** NVFP4 unsupported on T4 (SM7.5); vLLM/torch incompatibility blocked quantization. Defer.
- **E4 Difficulty router:** gated on E5 oracle gain ≥3pt. E5 oracle gain = +0.110 → router viable.
- **E5 Hard-loop:** loop2 acc 0.225, loop4 acc 0.180, oracle 0.335, oracle gain +0.110. More degradation (0.155) than improvement (0.110). Loop gain negative (-0.045).
- **E6 Context ceiling:** queued on Kaggle (vLLM env issue fixed); results pending.
- **E7 Quant ladder:** all modes failed with vLLM Config 'deprecated' kwarg (env incompatibility). NVFP4 confirmed false on T4.

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

Ceiling = 64K on single T4. last_loop mode required to keep KV loop-invariant.

## 4. E3 — NVFP4 Reality

NVFP4 unsupported on T4 (SM7.5); tensor-core dequant requires Blackwell (SM100+). Quantization via vLLM bitsandbytes failed due to pip/torch conflicts. Defer; pivot to FP16/INT8 on working stack.

## 5. E5 — Hard-Loop Benchmark

200 prompts (GSM8K/MATH-500/BBH, 100 each, limit 100), loops 2 vs 4, greedy.

| Loops | Acc | Output tokens | rep rate | tok/s |
|---:|---:|---:|---:|---:|
| 2 | 0.225 | 102191 | 0.469 | 177.0 |
| 4 | 0.180 | 102274 | 0.468 | 103.8 |

- Oracle accuracy: 0.335 (67/200)
- Loop gain: -0.045
- Oracle gain: +0.110
- Improvement rate: 0.110 (22)
- Degradation rate: 0.155 (31)

Interpretation: More loops hurt on average (negative loop gain), but oracle gain +11pt means a difficulty router could recover. Degradation exceeds improvement → loop regression is real and task-dependent.

## 6. E6 — Context Ceiling (env failure)

Rerun failed: max_model_len set equal to context (no output-token headroom) → "decoder prompt + output tokens > max_model_len"; 64K+ also hit engine init failures. Context feasibility already established by E2: ceiling = 64K on T4. last_loop vs per_loop mode comparison not yet obtained. Re-run needed with max_model_len = context + gen_tokens buffer and pinned env.

## 7. E7 — Quant Ladder

All modes failed: `TypeError: Config() got an unexpected keyword argument 'deprecated'` (vLLM/transformers version mismatch). NVFP4 confirmed false on T4. Environmental blocker; pin vLLM env before retry.

## 8. Conclusions & Recommendations

1. **Loop count:** 2 loops is a reasonable default; 4 loops degrades quality but raises oracle headroom. Router viable (oracle gain +11pt).
2. **Context:** 64K ceiling on T4; 128K/256K need PP=2 or SWA-aware KV + INT8/GGUF.
3. **Quantization:** blocked by env; NVFP4 dead on T4. Pursue FP16 baseline + INT8 via pinned vLLM.
4. **Router (E4):** proceed — oracle gain ≥3pt threshold met.
5. **Negative result:** more loops ≠ better quality on hard tasks; regression taxonomy (drift/over-correction) warranted.

## 9. Reproducibility

Artifacts committed to repo results/e1, results/e2, results/e5, results/e7. Scripts in experiments/.
