# E4 Router Spec — Pre-Final Critique (Crouton Alpha)

## E5 Asymmetry Changes the Bar

E5 on 200 prompts: loop2 **0.225**, loop4 **0.180**, oracle **0.335**. Improvement from the oracle over loop2 is only **+0.11**, while degradation from wrongly picking loop4 is **0.155**. The downside is larger than the upside.

That means the router must be more than "better than chance." At 60% accuracy it can net negative, because each wrong flip costs more than each correct rescue gains. The spec must state its break-even accuracy vs. always-loop2. Oracle 0.335 alone is not a justification.

## 1. Learned Router vs. Difficulty Heuristic on 200 Labels

A learned router will struggle to beat prompt length + math markers in any convincing way at n=200.

The dominant signal is plausibly difficulty: longer prompts, more digits, equations, code fences. A two-feature heuristic will capture most of what 200 samples can support. Any learned edge (+0.02–0.03) will drown in noise.

The spec should therefore:

- **Make the heuristic the default, not the fallback.** Specify it up front and evaluate it untouched on a held-out split.
- **Require the learned router to beat the heuristic head-to-head** on held-out data with a paired test (McNemar or bootstrap on per-prompt deltas), not just to beat loop2.
- **Freeze features in advance.** If the spec permits open-ended feature hunting — embeddings, TF-IDF, perplexity, topic features — against 200 labels, it is not a spec, it is permission to overfit.

If the learned router cannot clear a length-plus-markers baseline, ship the baseline.

## 2. Overfitting Risk Is the Central Risk

200 samples with high-dimensional features is a textbook overfitting trap, and the asymmetry punishes it extra hard. A router that memorizes training loop4-winners will steer loop2-winners into loop4 in production — exactly the expensive error.

Cross-validation alone does not fix this:

- With 5-fold CV on 200 points, each test fold is 40 points. Variance is huge. Report per-fold spread and a final untouched test number, not just mean CV accuracy.
- Feature selection on all 200 followed by CV on the classifier leaks. Selection must be nested or done on training data only.
- The spec needs a strong prior toward loop2 (the better default), calibration, and a cap on model capacity.

Require the spec to pre-register: maximum feature count (I would say ≤5), the evaluation split, and minimum effect size worth acting on. Without that, E4 will arrive with an illusory +0.04 win that does not replicate.

## 3. Maybe the Right Action Is loop1, Not loop2 vs. loop4

E5 compares only loop2 vs. loop4. It never reports loop1. Given degradation (0.155) exceeds improvement (0.11), the natural hypothesis is that looping *hurts* on hard prompts — long chains drift, restate, and compound errors. If so, "don't loop on hard tasks" matters more than "pick 2 vs. 4."

Before finalizing a 2-vs-4 router, the spec must answer:

1. **What is loop1 on the same 200?** If loop1 is close to loop2 or beats loop4, the router optimizes the wrong margin.
2. **Where is degradation concentrated?** Break the 0.155 down by difficulty. I predict it lives on hard tasks, while the 0.11 improvement lives in the middle.
3. **What does a difficulty gate buy?** A simple "difficulty above T → loop1 (or strict loop2 cap, no escalation)" may capture more value than any 2-vs-4 chooser.

Either add loop1 to the action space or justify exclusion with numbers. A router choosing only between two loop counts while the data hints looping hurts hard prompts is mis-specified.

## 4. Latency: Router Cost vs. Loop Savings

The spec needs a latency and cost budget for the router itself. Picking loop2 over loop4 saves two iterations, but only if the router is cheap.

- A text-statistic heuristic costs ~0. Fine.
- An embedding model or LLM classifier adds a forward pass that may exceed the two saved loops. A router costing one loop to save two halves its benefit; costing more, it erases it.

Require the spec to compare end-to-end latency: `router + chosen loop` vs. `always loop2`, and cap router cost (e.g., <5% of loop2 latency, no LLM call). A more expensive router should be disqualified by default. Accuracy on paper that regresses in production is not a win.

## 5. Crouton Proposal: Confidence-Gated Escalation

I would replace the 2-vs-4 pre-router with this:

**Run loop2 always. Escalate only on low confidence.** After loop2, score confidence (verifier, self-consistency, reward model). Only below a calibrated threshold, run two more loops. This is strictly easier than 2-vs-4 classification: it finds the small subset where extra loops help, defaults to the better baseline, and its worst case is loop2 — never worse than doing nothing. The pre-router's worst case is actively worse, paid on every prompt.

If escalation rarely helps, fall back to **loop2 default + a different hard-task path** (retrieval, calculator, code execution, or loop1 with a fresh prompt). That attacks the dominant degradation term directly.

**Bottom line:** Do not final E4 as a 2-vs-4 classifier. Rewrite it as gated escalation with loop2 as default, a heuristic baseline to beat, caps on features and router latency, and a loop1 comparison on the same 200 before any improvement is claimed.
