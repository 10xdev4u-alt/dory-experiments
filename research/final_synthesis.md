# Final Lap Synthesis — Dory 20h Experiment Lap

**Date:** 2026-11-09 · **Model:** `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` · **Hardware:** Kaggle 2×T4 (SM 7.5)

## Executive summary

**Headline: looping does not degrade content; extraction does.** Re-extracting the same 400 E5 outputs with answer-position awareness moves graded accuracy from 0.225 → 0.460 (loop2) and 0.180 → 0.465 (loop4); 30 of 31 "degraded" cases are extraction artifacts. Loop gain on content is ~0 (+0.005, McNemar p = 1.0, 95% CI [−0.05, +0.06]). Default to **loop2** (1.7× cheaper: 2.89 vs 4.93 s/prompt). The router rerun is marginal (+0.015, net capture 17.6%). Context is capped at **64K**; quantization is blocked by the environment and must be deferred.

## All 12 experiments

| # | Experiment | Result |
|---|---|---|
| E1 | Loop ladder (50 trivial, loops 1–4) | Acc 1.0 all loops; tok/s 25.6 → 10.3; rep ~0.44. Saturated — no oracle headroom, rescope to hard tasks |
| E2 | 256K feasibility (NIAH) | 8K–64K perfect at loop1 (5/5); 64K loop4 0/5; 128K/256K OOM. **Ceiling = 64K** |
| E3 | NVFP4 reality | NVFP4 needs Blackwell (SM100+); unsupported on T4. Negative result |
| E4 | Router (original labels) | LR, 5 features, CV acc 0.530; net 0.220 vs static 0.225. Fails kill (<0.280) |
| E5 | Hard loop (200 = 100 GSM8K + 100 MATH-500) | Original: loop2 0.225, loop4 0.180, oracle 0.335; loop gain −0.045; improved 22 vs degraded 31. Note: 0 BBH (loader empty; 200 not 300) |
| E6 | Context ceiling (`last_loop` vs `per_loop`) | Void: `max_model_len` = context, no output headroom. Harness bug, not a result |
| E7 | Quant ladder (FP16/8/4-bit) | All modes failed: `Config() 'deprecated'` kwarg — vLLM/transformers skew. INT8 unmeasured |
| E8 | Confidence gate (length split) | 99/200 escalated; net 0.230 (+0.005). Marginal; length is not confidence |
| E9 | Loop1 vs loop2 (+ verifier) | Unrun at lap end; spec frozen. **The gate for all defaults** (~0.2 GPU-h next cycle) |
| E10 | Pinned-env quant retry | Pinned stack spec'd: vLLM 0.33.0, torch 2.14.0, transformers 5.16–5.21, bnb ≥0.48.1, plugin 0.0.3; TinyLlama gate; GGUF Q5_K_M fallback |
| E11 | Extraction fix | Re-extracted loop2 0.460, loop4 0.465 (GSM8K 0.26→0.66; MATH 0.19→0.26); 30/31 degraded = artifacts; 1 genuine collapse. **Breakthrough** |
| E12 | Router rerun (E11 labels) | CV acc 0.565; net 0.475 vs 0.460 (+0.015); 86/200 esc. (TP 8, FP 78); net capture 17.6% |

E8b (self-consistency gate) and E4b (family rule: escalate MATH, net 0.485) are pre-registered designs, not results.

## Key findings

**1. Loops are flat, then wasteful.** E1 saturated because trivial prompts have no headroom. E5's −4.5 pt loop gain vanishes under E11: re-extracted gain is +0.5 pt, content flat (126 → 131 target-present). 19/22 improvements are extraction lottery; GSM8K degrades with 8.0 numbers after the target, MATH loses `\boxed{}`. Cost is the difference: 577s vs 986s/200 prompts.

**2. Extraction dominates measurement.** The old grader took the last number / `\boxed{}`, so 199/200 outputs hitting the 512-token cap plus doubt markers ("let me double-check…") corrupted labels. E11 (first answer / `\boxed{}` priority, stop at doubt) recovers +23.5 pt (loop2) and +28.5 pt (loop4). Caveat: this is sensitivity analysis, not corrected capability — freeze a strict grader a priori and validate held-out before claiming the jump.

**3. Routers lose to static loop2.** Re-extracted cells are symmetric (both 76, neither 91, only-loop2 16, only-loop4 17; upside 0.085, downside 0.080). E12 captures 17.6% of oracle headroom. The family rule (0.485, p=0.30) also fails cost-parity: at escalation `e`, parity needs `acc ≥ 0.460 + 0.325·e`, so ≥30% escalation loses even at oracle. Loop2 wins on s/correct (6.28 vs 10.60) and correct/1k s (159 vs 94).

**4. Context 64K; quant deferred.** E2's failure is activation/workspace, not KV arithmetic (`last_loop` keeps KV loop-invariant). 128K/256K need PP=2 or SWA-aware KV + INT8/GGUF. INT8 was never measured; NVFP4 is Blackwell-only.

## Negative results

- **E5 "loop regression" is an artifact.** Fewer loops fixes cost, not quality.
- **E4/E12 routers fail the bar** (original net 0.220; rerun +0.015 at 43% escalation). Prompt features can't separate the 17 opportunities.
- **E8 length gating is noise** (+0.005, CI includes 0); E8b predicts the same wall at 1.76× loop4 cost.
- **NVFP4 on T4 and the repo's `requirements.txt` stack are dead.** Retrying INT8 on torch <2.7 / transformers <4.50 will not load.
- **E6 as run is void.** Always set `max_model_len` = context + generation buffer.

## Top 3 recommendations for the next dev cycle

**1. Ship the extraction fix (Beam + Iceman, CPU-first, 1–2 days).** Replace the last-number grader with E11 logic: GSM8K = first `Answer:`-position number before doubt markers; MATH = `\boxed{}` priority. Add stop sequences at the answer, `no_repeat_ngram_size=3`, repetition penalty, and stream-stop / larger token budget so the answer is not truncated. Freeze the grader before re-grading; report graded + content oracle. Acceptance: <5% extraction failures on held-out 200; loop2 graded within 2 pt of content. This closes the ~55/200 present-but-graded-wrong gap — the cheapest accuracy left.

**2. Default loop2; fund E9 before any gate (Crouton Alpha, ~0.2 GPU-h).** Production: `hf_overrides = {n_recurrent_loops: 2, recurrent_kv_cache_mode: last_loop}`; loop4 flagged off. Do not ship E12 or the family rule. First GPU job: E9 loop1-vs-loop2 on the same 200 (bootstrap + McNemar). If loop1 ≥ loop2 − 0.01, revert to loop1 and expand any gate to {1,2,4}; if loop1 ≤ loop2 − 0.05, keep loop2 and run the E5c decoding sweep. No E8 threshold freezes before E9 scores. Any future gate must clear net capture ≥0.50 on cost/correct, not accuracy alone (E8b adaptive 2-to-3 is the only candidate).

**3. Publish 64K; defer quant behind E10 (LongCat, ~1.2 GPU-h).** Remove 128K/256K claims. Fix E6 harness (context + gen buffer, chunked prefill, `enforce_eager`, `last_loop`), then `last_loop` vs `per_loop` at 32K/64K. Build E10's pinned venv in a fresh kernel, pass the TinyLlama FP16 gate, then FP16 vs INT8 at loop2 (50 hard prompts + 64K/128K NIAH probes). Kill INT8 if drop >5 pt or OOM where FP16 fits; else ship FP16 and treat GGUF Q5_K_M as the memory path. NVFP4 stays deferred on T4.

**Next:** E9 loop1 run → extractor + stop ship → E6 fix + E10 env (parallel) → E8b only if a real content-level gap survives. No distillation until a ≥3 pt content gap is proven.
