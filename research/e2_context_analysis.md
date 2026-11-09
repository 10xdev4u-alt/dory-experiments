# E2 Context Feasibility and E6 Failure — Analysis

## Headline

E2's "64K ceiling" is a KV-pool allocation boundary, not a retrieval limit, and the 64K loop4 failures are engine-init memory-budget errors, not NIAH accuracy failures. E6's failures are a separate harness bug (loop overrides silently dropped) plus a `max_model_len` config error. The path to 128K/256K is `last_loop` + INT8 KV + INT8/GGUF weights on one T4; PP=2 is unverified for Dory.

## 1. What E2 actually ran

`experiments/02_reality_check.py` calls `load_llm(loops, ctx)`, so `max_model_len` equals the target context, not 262144. The `meta.max_len=262144` in `results/e2/results.json` is only the ladder filter. The Kaggle log confirms per-load `max_model_len` values of 8192, 16384, 32768, 65536, 131072, 262144. E2 and E6 both used target-sized pools; pool sizing is not the E2/E6 difference.

Second, E2's prompts are character-based: `FILLER = "the forest river stone cloud light path "` (38 chars, roughly 7 tokens) repeated to `ctx` characters. The "64K" rung is therefore about 12K tokens, "32K" about 6K, and "128K" about 24K. E2 never actually retrieved from 64K tokens. The OOM boundary is real — vLLM must be able to allocate the pool at `max_model_len` — but the NIAH recall numbers describe roughly 12K-token contexts, not 64K.

## 2. Why 64K loop4 fails while loop1 passes

All five 64K loop4 cases in `results/e2/results.json` are engine-init errors (`ok: false`), not wrong answers. The log gives the exact cause:

```
ValueError: To serve at least one request with the model's max seq len (65536),
(4.04 GiB KV cache is needed, which is larger than the available KV cache memory (3.71 GiB).
Based on the available memory, the estimated maximum model length is 55808.
```

Under `last_loop`, the KV cache is loop-invariant, so the 4.04 GiB requirement is identical for loop1 and loop4. The difference is the *available* budget:

| Condition | CUDA graph capture | Available KV | Need at 64K | Result |
|---|---:|---:|---:|---|
| E2 loop1 | 0.41 GiB | 4.28 GiB | 4.04 GiB | fits (0.24 GiB headroom) |
| E2 loop4 | 0.94 GiB | 3.71 GiB | 4.04 GiB | fails (0.33 GiB short) |
| E6 (eager) | 0 GiB | 4.74 GiB | 4.98 GiB | fails (per_loop x2) |

The loop4 CUDA graph is 0.53 GiB larger because the captured forward pass includes four recurrent loops' intermediate buffers. That tax pushes the KV budget below the 64K requirement. This is not a recurrent-mode KV multiplication, and it is not allocator fragmentation: the same loop4 config succeeded at 8K/16K/32K earlier in the same process, and the available-KV values are stable across repeated loads.

## 3. E2 vs E6: the real difference

E6's `make_llm` passes `num_loops`/`loop_output_mode` directly to `LLM`; every candidate raises `TypeError` and the code falls back to `LLM(**llm_kwargs)` with no override. The Kaggle log's `non-default args` confirm there is no `hf_overrides`: E6 ran at Hugging Face defaults (`n_recurrent_loops=2`, `per_loop`). At 64K that needs 4.98 GiB of KV versus last_loop's 4.04 GiB, and E6's eager-mode budget is 4.74 GiB — short by 0.24 GiB. At 128K it needs 8.48 GiB. E6's 64K/128K engine-init failures are therefore the loop-override bug, not a genuine ceiling.

E6's 32K failures are a different config bug: `max_model_len=32768` with `gen_tokens=64`, so `prompt + output > max_model_len` is rejected at request time. The engine initialized fine; generation did not.

Two other differences matter. E6 used `enforce_eager=True`, removing the CUDA-graph tax and producing the higher 4.74 GiB budget. E6 also used tokenizer-trimmed prompts of exactly `ctx` tokens, so its 64K rung is a real 64K-token prefill, unlike E2's roughly 12K.

## 4. last_loop vs per_loop

`last_loop` keeps one KV cache per recurrent layer; each loop overwrites the previous loop's entries, so KV is loop-invariant. `per_loop` keeps N independent caches, multiplying recurrent-layer KV by N. The data show the cost: at 64K, per_loop x2 needs 4.98 GiB versus last_loop's 4.04 GiB; at 128K, 8.48 versus 6.29 GiB.

For this hardware, `last_loop` is the better default: it is the only mode that fits long context, and E2's NIAH recall was 5/5 at loop1 for every feasible rung. The open scientific question — whether the overwrite "chimera" (historical tokens keep final-loop KV while current tokens attend to current-loop KV) costs accuracy — is exactly what E6 was supposed to measure, and the loop-override bug invalidated it. The correct ablation is at 32K or below, where per_loop fits, with `hf_overrides` and a resolved-config assertion.

## 5. Path to 128K/256K

The empirical KV model from E2 is `KV_last_loop(L) = 36 KiB/token x L + 1.79 GiB` (4.04 GiB at 64K, 6.29 at 128K, 10.79 at 256K). The 36 KiB/token term is exactly the 9 full-attention layers; the 1.79 GiB fixed term is the sliding-layer allocation. Against a roughly 4.7 GiB eager budget, BF16 KV cannot reach 128K.

The path:

1. `kv_cache_dtype="int8"` halves the KV: 128K becomes 3.15 GiB (fits), 256K becomes 5.40 GiB.
2. Shrink weights: INT8 (3.70 GiB) or GGUF Q5_K_M (3.42 GiB). With INT8 weights + INT8 KV + eager, 256K totals about 11.1 GiB against the 14.4 GiB budget — it fits with roughly 3 GiB headroom.
3. `enforce_eager=True` to remove the 0.4-0.9 GiB CUDA-graph tax, or capture only small batch sizes.
4. Fix the harness: `hf_overrides={"n_recurrent_loops": N, "recurrent_kv_cache_mode": "last_loop"}`; `max_model_len = context + gen_tokens`; tokenizer-based prompts with actual token counts logged; reuse one engine across depths; `torch.cuda.empty_cache()` if reloading.
5. PP=2 is not the primary path: `dory.py` rejects TP greater than 1 and PP is unverified. The deep dive's claim that 256K fits with INT8/GGUF weights and BF16 KV used a KV model that underestimates vLLM's actual allocation by about 1.7x; with the empirical model, 256K also needs INT8 KV.

## 6. Corrections to prior artifacts

- `REPORT.md` sections 3 and 6: "ceiling = 64K" is a pool-allocation artifact, and the loop4 failures are engine-init, not recall.
- `research/dory_vllm_deep_dive.md` section 4: the KV table underestimates vLLM's Dory KV spec (2.36 vs 4.04 GiB at 64K); the "256K fits with INT8/GGUF + BF16 KV" verdict is wrong.
- E6's last_loop vs per_loop comparison is invalid; rerun with `hf_overrides` and the config assertions from `research/e9_loop1_spec.md` section 3.
