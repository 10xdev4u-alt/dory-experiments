# E9 Loop1 vs Loop2 Spec — Dory on the Hard 200

**Version 1.** Model: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`. Hardware: Kaggle 1×T4 (Dory is TP=1 only). Data: the verbatim 200 prompts from `results/e5/checkpoints.jsonl` (100 GSM8K + 100 MATH). Baselines: E5 loop2 = 0.225, loop4 = 0.180, oracle = 0.335.

## 0. Question and hypothesis

E5 measured loops 2 and 4 only. It never measured loop1, so the sign of the loop effect on hard prompts is unidentified: loop2 → loop4 loses 4.5 pt, but loop1 → loop2 could be flat or positive. The E5 taxonomy says the 2→4 regression is 100% extraction artifact — 199/200 outputs truncated at 512 tokens, content accuracy 63.0% → 65.5% while graded accuracy falls 22.5% → 18.0%. If that mechanism is monotone in loop count, loop1 (fewer recurrent passes, shorter and less degenerate output) may match or beat loop2. **Hypothesis:** loop1 ≥ loop2 − 0.01 on graded accuracy. If true, looping itself — not loop4 specifically — is the degradation, and the hard-task default should be loop1.

## 1. Data and pairing

The 200 prompts are the exact E5 set: 100 GSM8K (test) + 100 MATH-500. **BBH is absent from E5** despite the "GSM8K/MATH/BBH" label in `PLAN.md` and `REPORT.md`; `load_bbh` returned nothing in the E5 session. E9 uses the frozen E5 list, paired by `idx`, so the comparison is exact and no prompt is re-sampled. Adding BBH now would require fresh loop2 labels and break the pairing; it is out of scope unless separately funded. Prompt identity is verified by hashing the ordered (idx, question, target) list against the E5 checkpoints before the run.

## 2. Arms

| Arm | Source | Config |
|---|---|---|
| loop1 | fresh run | `hf_overrides={"n_recurrent_loops":1,"recurrent_kv_cache_mode":"last_loop"}` |
| loop2 | reuse E5 | 0.225 (45/200), 577.3 s, 177.0 tok/s |
| loop4 | reuse E5 | 0.180, 985.7 s, for the 3-way oracle only |

Loop2/loop4 are not re-measured. Reusing them keeps the comparison paired and saves ~0.2 GPU-h; the risk is environment drift between the E5 session and E9, so the manifest (§3) must be recorded and any vLLM/model-revision mismatch flagged before scoring. A fresh loop2 arm is optional insurance against drift (~0.1 GPU-h) and should run only if the manifest check fails. Both arms use: greedy (`temperature=0`), `max_tokens=512` for GSM8K/MATH (256 for BBH if ever added), `max_model_len=8192`, `gpu_memory_utilization=0.90`, `trust_remote_code=True`, TP=1, prompt `Question: {q}\nAnswer:`, checkpoint every 25 prompts, raw outputs saved. The grader is the E5 grader unchanged (GSM8K last number; MATH `\boxed{}` then last number) — changing it mid-comparison would break pairing.

## 3. Pre-flight gates (stop if any fails)

1. **Prompt identity:** SHA-256 of the ordered (idx, question, target) list equals the E5 checkpoint hash.
2. **Resolved config:** log `n_recurrent_loops` and `recurrent_kv_cache_mode` from the loaded model; assert 1 and `last_loop`. A silent fallback to HF defaults (2, `per_loop`) invalidates the run — this is the E6 failure mode.
3. **Behavioral canary:** on a 10-prompt pilot, loop1 must differ from E5 loop2 in output length or throughput. Identical outputs mean the override did not take effect.
4. **Env manifest:** `pip freeze`, vLLM version/commit, model revision, `nvidia-smi`, and exact flags committed to `results/e9/env.json`.

## 4. Metrics and analysis

**Primary:** paired accuracy difference `acc(loop1) − acc(loop2)` with a 95% CI from 1,000 paired bootstrap resamples over prompts, and McNemar's exact test on discordant pairs. Report the point estimate and CI, not just significance.

**Secondary, per arm:**
- graded accuracy; content accuracy (target value present anywhere); **extraction-failure rate** = target present but graded wrong / total; truncation rate (`output_tokens == max_tokens`).
- 3-way oracle: `max(loop1, loop2, loop4)` per prompt; oracle gain vs loop2 and vs loop1; 3-way cell table (1/2/4 correctness, 8 cells).
- improvement/degradation counts loop1→loop2 and loop2→loop1.
- rep rate (E5 bigram metric), output tokens, tok/s, s/prompt.
- splits: GSM8K vs MATH; prompt-length terciles as a difficulty proxy.

**Taxonomy of loop1→loop2 flips** (E5 method): extraction lottery (target present in both), truncation recovery (target genuinely gained), MATH box loss, post-target noise count, drift (coherent alternative answer). This separates a reasoning change from an extraction change and is the decisive diagnostic given §0.

## 5. Decision rule (pre-registered)

| Observation | Decision |
|---|---|
| loop1 ≥ loop2 − 0.01 | Default to loop1. Looping is suspect; E5's negative loop gain becomes the headline. Expand E8's action space to {1,2,4} and re-frame its default. |
| loop1 ≤ loop2 − 0.05 and degradation < 0.05 | Looping helps. Keep loop2 default; run E8 as specified. |
| Between, or CI straddles 0 | Underpowered at n=200. E8 must include loop1 as a baseline and may claim no win without it. |

**Guards:** (a) if loop1's advantage is entirely extraction (content accuracy equal, extraction-failure rate lower), report it as extraction-driven — the default still flips to loop1, but the extractor fix is a separate, cheaper mitigation and the "looping hurts reasoning" claim is weakened; (b) if loop1 wins by fewer than 5 prompts, the bootstrap CI must exclude 0 before any ship decision; (c) throughput and rep rate are Pareto context, not vetoes.

## 6. Cost and deliverables

~0.1–0.2 GPU-h on 1×T4: loop1 should be faster than loop2 (577 s / 200 prompts in E5; E1 showed 25.6 vs 17.1 tok/s on trivial prompts), so ~300–450 s plus model load. Deliverables: `results/e9/checkpoints.jsonl`, `results/e9/results.json`, `results/e9/summary.md`, `results/e9/env.json`; update `REPORT.md` and `research/next_wave_plan.md`. Non-goals: no loop2/loop4 re-run, no BBH, no extractor change, no router retrain.
