# Next Wave Plan — E8–E10 After E1–E7

**2026-11-09 · Kaggle 2×T4 (sm_75), vLLM stack · Dory1.0-4B-BF16 · Crouton Alpha**
Prioritization: high information per GPU-hour; all prompts reuse E5's 200 so comparisons stay paired.

## What E1–E7 established

- **E1 (trivial loop ladder):** accuracy 1.00 at loops 1–4 (50 prompts); throughput 25.6→10.3 tok/s; repetition ~0.44. Trivial prompts have no oracle signal — do not reuse.
- **E2 (context):** NIAH perfect 8K–64K at loop1; 0/5 at 64K with loop4; OOM at 128K/256K (BF16). Ceiling is 64K on one T4; the 128K failure is activation/workspace, not KV arithmetic.
- **E3/E7 (quant):** NVFP4 confirmed false on sm_75 (Blackwell-only). All E7 modes failed with `Config() got an unexpected keyword argument 'deprecated'` — an env mismatch, not a result. Quantization is still unmeasured.
- **E5 (hard loop, 200 prompts):** loop2 = 0.225, loop4 = 0.180, oracle = 0.335. Oracle gain +0.110, loop gain −0.045. Improvement (22 prompts, 0.110) < degradation (31, 0.155).
- **E4 (2-vs-4 router):** CV accuracy 0.530, misroute 0.470, net 0.220 vs static loop2 0.225. "63.6% oracle capture" hides 86 false escalations. Fails the kill (net < 0.280). Arithmetic is settled: `acc = 0.225 + 0.110·r − 0.155·f`, break-even `f = 0.71·r`.
- **E6:** void (harness loop bug + `max_model_len` with no output headroom). Do not re-run until E10's env lands.

Three gaps remain: is looping itself harmful (E9, gates everything), can we rescue only the prompts looping helps (E8, replaces E4), and can we get any quant number on a pinned stack (E10, unblocks context)? GPU order: E9 → E8 → E10, with E10's env build in parallel.

## E9 — loop1 vs loop2 on the hard subset

**Hypothesis:** loop1 is within 1 pt of loop2 (or higher) on the same 200 prompts, and degradation concentrates on high-difficulty prompts. If true, looping — not loop4 specifically — is the problem, and the hard-task default should be loop1/no-escalation.

**Method:** one new arm on the verbatim E5 list (GSM8K/MATH-500/BBH, math `max_tokens=512`, BBH 256, greedy, TP=1): loop1 with `hf_overrides={"n_recurrent_loops":1,"recurrent_kv_cache_mode":"last_loop"}`, `max_model_len` = context + generation buffer. Loop2/loop4 are not re-measured. Report per-prompt 3-way cells (1/2/4), per-family/difficulty splits, a drift/restatement/over-correction taxonomy on loop1→loop2 flips, and paired bootstrap (1,000 resamples) + McNemar.

**Metric:** loop1 accuracy and 3-way oracle gain (max of 1/2/4 − loop2). Secondary: loop1→loop2 improvement/degradation rates, repetition, tok/s. Baseline: loop2 cost 577 s/200 prompts (2.89 s/prompt).

**Kill / decision criteria:**
- Loop1 ≥ loop2 − 0.01: looping is suspect. Kill loop2-default as stated; reframe E8's default to loop1 and expand all routing to {1,2,4}. E5's negative loop gain becomes the headline.
- Loop1 ≤ loop2 − 0.05 and degradation < 0.05: looping helps; proceed with E8 as loop2-default.
- Between (CI straddles 0): n=200 is underpowered; E8 must still include loop1 as a baseline and claim no win without it.

**GPU-hours:** ~0.2 (0.08 h run + load). Owner: any free T4.

## E8 — confidence-gated escalation (loop2 default)

**Hypothesis:** low self-consistency after loop2 predicts loop4-wins better than prompt features predict them. Escalating only the lowest-consistency prompts captures ≥50% of the +0.110 oracle gain at ≤10% false-escalation rate on loop2-winners — net ≥ +0.05 over static loop2. This replaces the E4 pre-router, whose worst case is worse than loop2 on every prompt; gated escalation's worst case is loop2 itself.

**Method (Arm B of `e4_router_spec.md` §9, frozen before running):** each prompt runs greedy loop2 (already have it) plus two loop2 samples at temperature 0.7, top-p 0.9, fixed seed. Self-consistency = normalized exact-match on the final answer plus token-overlap on reasoning. Escalate to loop4 iff score < τ, where τ is tuned on 160 train prompts only (same stratified split as E4 v2; test touched once), escalation capped at 40% of prompts, gate budget <5% of loop2 latency, CPU-only, no LLM call. Report end-to-end latency including the extra sample — a gate costing more than the two saved loops erases itself.

**Metric:** primary = net accuracy on the untouched test (40 prompts) with bootstrap 95% CI vs static loop2 (0.225) and vs the E4 digit-density × log-length heuristic (the default-to-beat; if the learned gate cannot beat it head-to-head, ship the heuristic). Secondary: `f`, `r`, net oracle capture `(acc−0.225)/0.110`, escalation rate, cost per correct answer, correct per 1k s.

**Kill criteria (same bar as E4 §8):** net capture < 0.50 (net < 0.280) or CI of (acc − 0.225) includes 0 → revert to loop2 (or loop1 per E9). Net < 0.215 → immediate revert, no canary. Gate + extra sample >5% p99 latency with no accuracy-dominated Pareto point → revert regardless. ECE > 0.10 → no ship without recalibration. 0.50 ≤ capture < 0.70 → ship behind a 10% flag, keep collecting labels.

**GPU-hours:** ~0.6 on 1×T4 (second sample ~0.15 h + escalated loop4 on ≤40% ~0.15 h + load). No training. Blocked on E9's default choice.

## E10 — pinned-env quant retry + GGUF/llama.cpp fallback

**Hypothesis:** on a correctly pinned stack, INT8 via `vllm-bnb-plugin` 0.0.3 loads Dory on T4, holds accuracy within 2 pt of FP16 on the 50-prompt hard subset at loop1, and cuts weights 7.40 → 3.70 GiB — making 128K (and per the KV model, 256K with `last_loop`) feasible. GGUF Q5_K_M (3.42 GiB) is the fallback if the plugin path fails.

**Method:** two phases, TinyLlama gate first.
1. **Fresh pinned venv:** `vllm==0.33.0`, `torch==2.14.0`, `transformers>=5.16.1,<5.21.0`, `bitsandbytes>=0.48.1`, `vllm-bnb-plugin==0.0.3` (drop the repo's stale `torch<2.7` / `transformers<4.50` / `bitsandbytes<0.46` pins for this env only). Log `nvidia-smi` driver; if < R580 set `VLLM_ENABLE_CUDA_COMPATIBILITY=1`. TinyLlama-1.1B FP16 must generate before any Dory run; its bnb modes are reported but not blocking.
2. **Dory ladder:** FP16 vs INT8 (`quantization="bitsandbytes"`, `load_format="bitsandbytes"`) at loops 1/2 on 50 hard prompts. Log weights, tok/s, accuracy, repetition, plus a 64K NIAH probe (loop1/loop4, `last_loop`, `enforce_eager=True`, `max_model_len` = context + gen buffer) and one 128K NIAH probe under INT8/`last_loop`/chunked prefill. **Fallback only on plugin failure:** build llama.cpp PR #30306, load Dory Q5_K_M GGUF, `--ctx-size` 64K→256K, `last_loop`, `--no-context-shift`, reporting prefill chunk size with every number (it changes logits in that PR).

**Metric:** accuracy delta vs FP16 (pt), tok/s, weights GiB, max feasible NIAH context, load success/failure with pinned versions quoted.

**Kill criteria:** TinyLlama FP16 gate fails → stop, report driver/CUDA blocker, burn no Dory hours. INT8 drop >5 pt at loop1 or OOM at 64K where FP16 fits → kill INT8; FP16 is the T4 default, GGUF the memory path. 128K still OOM under INT8/`last_loop` → 64K ceiling is real; re-scope long-context claims to 64K. GGUF build fails → documented negative result (PR is Draft, K>4 unsupported).

**GPU-hours:** ~1.2 on the second T4 (0.5 env/build + 0.4 ladder + 0.3 NIAH; +0.5 only if GGUF fallback triggers). E6's re-run, when funded, reuses this env and its `hf_overrides` fix.

## Sequencing and total

**Total: ~2.0 GPU-h** (E9 0.2 + E8 0.6 + E10 1.2; +0.5 GGUF fallback), ~1.5 h wall-clock with 2×T4 parallelism. E9 must score before E8 thresholds are frozen — its default sets E8's action space — while E10's env builds in parallel. Non-goals: no router retrain (E4 arithmetic is settled), no 256K claims without INT8/GGUF, no speculative decoding while loop1 is unknown, no new datasets. The wave turns E5's negative loop gain into a decision: either looping is harmful and we stop, or gated escalation earns its latency.
