# E9 Expectations — What Loop1 Will Tell Us

**Date:** 2026-11-09 · **Author:** Crouton Alpha
**Context:** E5 loop2 = 0.225 (45/200), loop4 = 0.180 (36/200), oracle = 0.335, improvement = 0.110 (22), degradation = 0.155 (31). E9 spec: `research/e9_loop1_spec.md` (loop1 fresh on the same 200, paired bootstrap + McNemar).

## Why E9 is the gate

E5 never measured loop1, so the sign of looping on hard prompts is unidentified: loop2→loop4 loses 4.5 pt, but loop1→loop2 could be flat or positive. Every downstream bet — E4's router, E8's escalation, distillation — assumes loop2 is the default. E9 tests that assumption for ~0.1–0.2 GPU-h and sets E8's action space.

The E5 taxonomy sharpens the prior. The 2→4 regression is 100% extraction artifact: content accuracy rises 63.0% → 65.5% while graded accuracy falls; 199/200 outputs hit the 512-token cap; degraded GSM8K cases average 8.0 numbers after the target. If truncation is monotone in loop count, loop1 should suffer less extraction loss. Prior: loop1 within a few points of loop2, plausibly above it.

## Scenario A: loop1 ≥ loop2 — looping is harmful

**Observation.** Loop1 accuracy ≥ loop2 − 0.01 on graded accuracy, with degradation loop1→loop2 > improvement. The 3-way oracle barely beats loop1 alone.

**Product meaning.** The headline negative result. Recurrence does not help reasoning on hard math; it buys repetition and truncation risk. Default becomes loop1; loop2-default dies. E8 becomes a three-way gate over {1, 2, 4} with loop1 as floor. If loop1's win is entirely extraction-driven (equal content accuracy, lower extraction-failure rate), the default still flips but the "hurts reasoning" claim weakens — the fix is a cheaper extractor.

**Next experiments.**
1. **E9b extractor + stop fix (no new loops).** Re-grade the saved loop1/loop2/loop4 outputs with answer-position extraction (last `Answer:` / `\boxed{}`), stop sequences, and `no_repeat_ngram`. If graded accuracy equalizes, E5's loop gain was measurement, not model. CPU-only, hours.
2. **E8-three-way gate.** Rebuild E8 around loop1 default: learn when any looping helps versus loop1-only, same kill bar as E4 (net capture ≥ 0.50, CI excludes 0). Do not run E8-two-way as written — it optimizes the wrong margin.

## Scenario B: loop1 ≪ loop2 — looping helps

**Observation.** Loop1 ≤ loop2 − 0.05 with degradation (loop1→loop2) below 0.05 and content accuracy also rising loop1→loop2.

**Product meaning.** Recurrence earns its cost; the 2→4 falloff is a fixable artifact. Keep loop2 default. The taxonomy (truncation, lost `\boxed{}`, post-answer noise) becomes the roadmap: better decoding recovers the +0.110 oracle headroom without changing the model.

**Next experiments.**
1. **E5c decoding sweep on fixed prompts.** Hold loop2, vary `max_tokens`, repetition penalty, and stop-at-answer. Predict: closing the 84-prompt loop2 extraction gap (target present but graded wrong) beats any router. This is the cheapest accuracy win if B holds.
2. **E8 as specified + E10 quant at loop1.** Run confidence-gated escalation (loop2 → loop4 only on low self-consistency), and re-run E10's INT8 ladder at loop1/loop2 so quant claims use the validated default. Do not expand E8 to {1,2,4} — loop1 is dominated and adds label cost without product value.

## Scenario C: loop1 ≈ loop2 − small — underpowered

**Observation.** Difference small (−0.01 to −0.05), bootstrap 95% CI straddles 0, discordant pairs too few (n=200 gives ≈5–10 mixed cells) for McNemar.

**Product meaning.** Likely if both loops truncate at 199/200. Neither claim is shippable. Throughput is the tiebreaker: loop1 was ~1.5× faster than loop2 in E1 (25.6 vs 17.1 tok/s), so it wins on cost per correct answer even at −2 pt. The accuracy claim stays open.

**Next experiments.**
1. **E9-extended: n = 1000 paired hard prompts (same families).** Powered to detect 2 pt. Freeze the E5 grader; do not change extraction mid-comparison. ~1 GPU-h at loop2 rates. Pre-register a ±0.02 equivalence margin, not just p < 0.05.
2. **E8 must carry loop1 baseline.** E8 runs with three static arms (loop1 / loop2 / gated) on the same split; no escalation claim without beating the best static, including loop1. A router that only beats loop2 while loop1 ties it is not a win.

## BBH mislabel — correction required

`PLAN.md` line 53 and `REPORT.md` §5 describe E5 as "GSM8K/MATH/BBH, 100 each" (300 prompts). The run never loaded BBH: `load_bbh` returned nothing; `results/e5/results.json` has `total_prompts: 200` = 100 GSM8K + 100 MATH-500. So E5 had **0 BBH**. Fix `PLAN.md`, `REPORT.md`, and `results/e5/summary.md` to read 200 = 100 + 100; note `max_tokens=256` (labeled BBH) was never exercised. E9 correctly excludes BBH to preserve pairing. No BBH loop-scaling conclusion is licensed.

## Bottom line

E9 is a 0.2 GPU-h experiment that decides the product default. Prior favors A or C (artifact is monotone in loops), but B is the only outcome that keeps the current loop2-default roadmap intact. Fund E9 first, freeze the corrected n=200, and let its CI — not E5's 2→4 slope — set E8's action space.
