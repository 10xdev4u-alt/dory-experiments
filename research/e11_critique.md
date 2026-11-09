# E11 Critique — Extraction Fix as Diagnostic, Not Correction

**Author:** Crouton Alpha
**Context:** E5 graded 200 hard prompts (GSM8K/MATH) with last-number extraction: loop2 0.225, loop4 0.180, oracle 0.335, loop gain −0.045. E11 re-grades the same 400 outputs with answer-position-aware extraction and reports loop2 0.460, loop4 0.465, with 30/31 degraded cases as artifacts. GSM8K 0.66/0.62, MATH 0.26/0.31. E11 is a useful diagnostic. The `REPORT.md` §10 claim that it "invalidates" E5 is not.

## 1. Post-hoc grader swap is metric gaming, even if the old grader was broken

E11's premise is correct. 199/200 outputs hit the 512-token cap, repetition rate ~0.47, and GSM8K degraded cases average 8.0 numbers after the last target occurrence. Last-number extraction then samples noise (E5 idx28, target 25, ends on 45). 19/22 improvements are extraction lottery. E5 taxonomy already proved this, and E9b pre-registered the re-grade as a CPU-only follow-up.

But changing the grader after seeing 0.225 and calling 0.460 the accuracy breaks comparability. E9 spec §2 froze the grader for exactly this reason: mid-comparison changes break pairing and let the metric do the work. A +23.5 pt jump from re-grading alone shows E5 measured a broken pipeline, not a stable model property — you cannot publish −0.045 as a loop regression and then retroactively declare it +0.005 because you changed the ruler, without a fresh run with the new ruler frozen *a priori* and a paired test.

Implementation detail worsens the concern. `_trim_doubt` cuts text after doubt markers, `_first_block` keeps only pre-`</think>` text, and `extract_gsm` picks the *last* `Answer:`/`=`/`is` candidate among many — the lenient choice that maximizes recovery of the buried value. No held-out extractor validation, no false-positive audit, no human review. E11 defines artifact as "target present in output," then reports 30/31 artifacts. Agreement with content accuracy (126→131 present) is therefore circular.

Keep E11 as sensitivity analysis — "how much could a lenient grader recover?" — not as corrected accuracy.

## 2. Buried answer ≠ answered question

E11 measures (i) generating the correct value somewhere in 512 tokens. Product correctness is (ii) presenting it as the final answer and (iii) the user finding it. Content accuracy and E11 both measure (i).

A model that writes "45 − 20 = 25" then repeats the solution five times past truncation, or loses `\boxed{}` in 7/13 degraded MATH cases and ends mid-sentence, did not answer. The target recurs ~10 times per output; for small targets (1, 2, 5, 12) coincidental appearance as an intermediate is expected even in wrong work. E5 §7 warned the content proxy "can credit coincidental intermediate values." E11 inherits this and degrades it: `target_present` is a raw substring check, firing on "104" inside "1040" or "2" inside "12."

GSM8K 0.26→0.66 is not a capability gain. It is the gap between "said it once" and "ended with it." Users rate (iii). Truncated noise they must mine is a failure, and crediting it trains us to ship an unusable model behind a heroic extractor.

## 3. Flat content collapses the loop, router, and escalation stories

Under E11's numbers loop gain is +0.005 overall (0.460→0.465), with opposite signs by dataset: GSM8K −0.04, MATH +0.05 at n=100 each. That is noise, not a loop effect. There never was a −4.5 pt reasoning regression; there was flat content and truncation-position variance.

Consequences:

- **Oracle gain likely collapses.** E5 oracle 0.335 (+0.110) was mostly lottery flips where the target was present in both loops. E11 never reports the re-extracted oracle. If it falls below +0.03, E4's kill bar (net ≥0.280) is moot — two equally-accurate loops leave nothing for a router to route, only its cost.
- **E9's hypothesis shifts.** If loop2→loop4 content is flat despite more degeneration, loop1 under re-extraction may also be ~0.45. Then the only loop differentiator is throughput (E1: 25.6→10.3 tok/s). Recurrence costs 2.5× for zero content gain at any count on hard tasks.
- **The real story is decoding.** 199/200 truncated, no stop-at-answer, no repetition penalty, 512-token budget — the benchmark measures where the cut lands. The +40 pt false-negative headroom (84 at loop2, 96 at loop4) dwarfs any router's +11 pt oracle. E11 proving looping doesn't hurt content does not prove looping helps or that loop4 is "fine."

## 4. Ship the stricter extractor

Grade like users: require `\boxed{}` for MATH, `Answer:`/`####` on the final line for GSM8K, count truncation-at-cap and buried answers as failures. E5's strictness, crudely implemented, punished the right failure mode. E11's leniency forgives it offline; shipping it hides a generation bug behind parsing.

Next: freeze the strict grader, re-run a decoding sweep (stop sequences, `max_tokens`, `no_repeat_ngram`) across loops 1/2/4, report strict/lenient/content side-by-side with human audit, compute the re-extracted oracle and re-test E4/E8 kills, and optimize cost-per-correct-answer. If strict accuracy stays low after decoding fixes, the model — not the extractor — is the blocker.
