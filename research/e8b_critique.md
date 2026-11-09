# E8b Self-Consistency Critique — Cost Without a Mechanism

**Date:** 2026-11-09 · **Author:** Crouton Alpha
**Context:** E8b v1 (`research/e8b_selfconsistency.md`) proposes 3× loop2 at T=0.7 as a confidence gate for loop4 escalation. E5: loop2 0.225, loop4 0.180, oracle 0.335 (+0.110), degradation 0.155. Taxonomy: content accuracy rises 63.0% → 65.5% while graded accuracy falls; 199/200 outputs hit the 512-token cap. 19/22 improvements are extraction lottery.

E8b's §1 names H0 — the oracle gain is extraction variance, not reasoning instability. H0 is not a rival to hedge against. It is the E5 base case, and E8b pays a 3× tax to test a mechanism E5 already falsified. Fix the extractor first.

## 1. The 3× tax: 8.66 s/prompt for the gate alone

Three loop2 samples cost 3 × 2.89 = **8.66 s/prompt — 3.0× loop2 and 1.76× static loop4 (4.93 s)** before the gate decides anything. With escalation, worst case is 13.59 s (4.7× loop2, 2.76× loop4). E8b can never win on cost, only on accuracy, and must clear loop2 (0.225) by enough to justify being permanently slower than the thing it escalates to.

The adaptive 2-to-3 variant (§4) does not rescue this. Two samples is still 5.77 s — 2× loop2, 1.17× loop4 — as a floor. A floor above static loop4 means E8b is never Pareto-dominant; at equal accuracy static loop4 is the better buy, and loop1 (per E9 scenarios A/C) likely wins on cost-per-correct-answer. Gated escalation's whole premise was "worst case is loop2 itself." E8b's worst case is 2–3× loop2 on every prompt.

The "~0.3 GPU-h" framing is misleading. It reuses E5's already-paid loop2; production has no reuse. Report cost-per-correct-answer as primary, or a passing test ships a product loss.

## 2. Agreement measures the wrong variance

Self-consistency assumes answer variance correlates with difficulty — the model is uncertain *about the answer*. E5 shows content is stable and measurement is unstable: 19/22 improved cases contain the target in *both* loops; loop2 graded wrong because the last-number extractor sampled post-answer noise, loop4 graded right because truncation landed cleanly (post-target numbers: 0 in 21/22 improved vs 8.0 mean in degraded). The 3 genuine gains are truncation recovery. 30/31 degraded cases still contain the target.

That breaks E8b's mechanism three ways:

**A lottery doesn't separate on agreement.** If loop4's win is "the extractor happened to hit at loop4," loop4-winners are not uncertain prompts — they are lucky-truncation prompts. Agreement among resampled lottery draws measures extractor stability, not difficulty. There is no path from "3 samples disagree" to "loop4's truncation lands better." H1 needs variance-to-difficulty correlation; E5 gives variance-to-truncation covariance.

**The gate is measured through the indicted instrument.** Agreement flags come from the same broken `extract_last_number` grader. A prompt with answer-present-but-noise-dominated looks high-variance (three different last numbers) and gets escalated to loop4 — which E5 shows is *more* noise-dominated (96 vs 84 false negatives). E8b systematically escalates where loop4 most likely degrades. On the lottery subset (86% of improvements) the gate is not noisy; it is anti-correlated.

**Consensus inherits the bias.** E8b-consensus takes the modal *extracted* answer. If the extractor favors intermediates — post-answer noise recurs ~10× per output — the mode is confidently wrong. Majority vote over biased measurements converges to the bias, because the lottery is in the grader, not the samples.

## 3. Cheaper alternatives dominate

E8b's own diagnostic concedes the answer: *"If the gains are extraction lottery, fix the extractor, not the gate."* So why pay GPU to re-derive the taxonomy?

- **Re-grade saved outputs (0 GPU).** Apply answer-position extraction (last `Answer:` / `The answer is` / `\boxed{}`, stop-at-answer, `no_repeat_ngram`) to the 400 saved E5 outputs. This recovers most of the 84 → 96 false negatives and tells us the *post-fix* oracle gain. Strictly more informative than 600 stochastic generations through a known-broken grader, at CPU-only cost.

- **2-sample majority, or a 1-sample noise gate.** E5's cleanest discriminator was post-target noise (0 vs 8.0) — computable from a single loop2 output at zero extra cost. Truncation-at-cap, lost `\boxed{}`, and repetition rate are all single-sample signals. If any behavioral gate deserves a test, it is "escalate iff loop2 is truncated/noisy," not "generate two more noisy outputs." Even granting self-consistency a fair test, the 3rd sample's marginal value over 2-sample agreement is minimal at n=200 — the spec's own rule ("if the 3rd adds <2 pt, kill") admits it.

Order matters: E8b at 3× cost is last resort, after the 0-GPU re-grade and the 1-sample gate, not first resort.

## 4. When it's worth it vs. when to abandon

**Worth it when all three hold:** (a) after extractor fixes, a genuine reasoning oracle ≥ +0.05 survives with drift — not truncation — as the dominant degraded class; (b) agreement on *content/boxed* answers predicts loop2-wrong better than prompt length; (c) the gated system is cheaper than static loop4 at matched accuracy, with worst case = default, not 2× it.

**Abandon when:** the oracle is ≥86% extraction lottery, the gate shares the extractor's bias, and the cost floor exceeds static loop4. Today all three hold.

**Recommendation: do not run E8b.** Run the 0-GPU re-grade first. If a real oracle survives, run a 2-sample gate against a 1-sample noise baseline on the fixed extractor. If none survives, kill escalation and spend GPUs on decoding fixes (stop sequences, repetition penalty, larger `max_tokens`) that close the 84-prompt extraction gap. That ~40 pt headroom dwarfs the +11 pt oracle E8b pays 3× to chase.
