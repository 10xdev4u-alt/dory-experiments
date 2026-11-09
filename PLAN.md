# Experiment Plan — Dory on 2×T4 (20h lap)

## Gates (kill/pivot triggers)
- **G1 (E1):** If oracle loop-gain < 1 pt on hard tasks → kill E4 (router) + E6 (distill); pivot headline to E5 (loop regression) + E2 reality check.
- **G2 (E2):** If needle recall collapses past 64K → stop all long-context work; re-scope to 64K.
- **G3 (E3):** If NVFP4 upcasts on T4 → INT8 becomes default inference path for everything downstream.
- **G4 (E5):** If loop-1 vs loop-4 KV cosine sim is low → kill speculative decoding (E7b); reallocate to E6.

## Wave 1 (hours 0–6): Science backbone
| # | Exp | Owner | GPU | Time | Gate |
|---|-----|-------|-----|------|------|
| E1 | Loop-count scaling law + exchange rate (loops 1–4 on GSM8K/MATH/ARC/MMLU/TruthfulQA; oracle labels) | Beam | 1×T4 | 3h | G1 |
| E2-probe | 256K feasibility probe (length ladder 8K→256K prefill-only OOM) | LongCat | 1×T4 | 1h | G2 |
| E3-probe | NVFP4 dtype check + INT8 AWQ baseline | LongCat | 1×T4 | 1h | G3 |

## Wave 2 (hours 6–12): Shippable wins + reality checks
| # | Exp | Owner | GPU | Time | Gate |
|---|-----|-------|-----|------|------|
| E4 | Difficulty router (logistic/MLP on loop-1 features; Pareto frontier) | Crouton | 1×T4 + CPU | 3h | needs E1 labels |
| E5 | Loop regression hunt (adversarial/OOD; non-monotonicity; taxonomy) | Iceman | 1×T4 | 2h | — |
| E2-full | Full NIAH ladder with last-loop KV caching | LongCat | 2×T4 | 2h | G2 |
| E3-full | INT4 quant quality + KV headroom | LongCat | 1×T4 | 1h | — |

## Wave 3 (hours 12–16): Systems + attacks
| # | Exp | Owner | GPU | Time |
|---|-----|-------|-----|------|
| E7a | Adversarial loop fuzzing (repeat/oscillation/DoS) | Iceman | 1×T4 | 2h |
| E7b | KV-compat probe (30min) → speculative decoding accept-rate | Iceman | 1×T4 | 3h (gated on G4) |
| E8 | INT4 keep-rate + cross-model soup (kill fast) | Crouton | 1×T4 | 1h |

## Wave 4 (hours 16–20): Distillation (gated) + write-up
| # | Exp | Owner | GPU | Time |
|---|-----|-------|-----|------|
| E6 | Distill 4-loop teacher → 1-loop student (LoRA only) | Crouton | 1×T4 | 4h (only if E1 oracle gap ≥ 3pt) |
| E9 | Dory-as-verifier self-consistency | Iceman | 1×T4 | 2h |
| — | Write-up, plots, negative results, final report | Beam | — | 2h |

## 20h skeleton
- 0–3h: E1 + E2/E3 probes (interleaved on GPU-B)
- 3–6h: E4 router + E5 loop-breaker in parallel (one per GPU)
- 6–10h: E2 full ladder + E3 full quant
- 10–16h: E6 distill (only if E1 shows real oracle gap) + E7
- 16–20h: write-up with negative results

## Success criteria
- A defensible loop-scaling curve (accuracy vs loops) with oracle gap.
- A measured max context on 2×T4 with last-loop KV caching.
- A measured NVFP4 reality (dtype + speed + quality) on T4.
- A shippable difficulty router (if oracle gap ≥ 3pt) with Pareto frontier.
- Honest negative results documented (non-monotonicity, overthinking, OOM boundaries).
