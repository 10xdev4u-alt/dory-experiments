# E5 Hard-Loop Regression Taxonomy

**Date:** 2026-11-09
**Inputs:** `results/e5/checkpoints.jsonl` (400 rows = 200 prompts × loops 2/4), `results/e5/results.json`
**Model:** nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, greedy decoding, max_tokens 512 (GSM8K/MATH), 256 (BBH)

## 1. Headline

The 31 degraded and 22 improved cases are not primarily a reasoning phenomenon. They are a
**measurement artifact**: the model states the correct answer, then degenerates into repetition and
is cut off at the 512-token cap; the grader's last-number extractor samples whatever number happens
to be last before the cut. Content accuracy — the correct value appearing anywhere in the output —
is flat-to-up (63.0% → 65.5%), while graded accuracy falls (22.5% → 18.0%). Loop scaling from 2→4
does not measurably change reasoning content; it changes where the truncation lands.

## 2. Global evidence

| Metric | loop2 | loop4 |
|---|---:|---:|
| Graded accuracy | 45/200 (22.5%) | 36/200 (18.0%) |
| Correct value present in output | 126/200 (63.0%) | 131/200 (65.5%) |
| Outputs truncated at 512 tokens | 199/200 | 199/200 |
| Target present but graded wrong | 84 | 96 |
| Mean repetition rate | 0.4688 | 0.4683 |
| Total output tokens | 102,191 | 102,274 |

Two facts dominate. First, **199/200 outputs hit the 512-token cap in both loops**; the budget is
far too small for this model's repetitive style. Second, the graded-vs-content gap is enormous and
grows: 84 outputs contain the right answer but are graded wrong at loop2, 96 at loop4. The "loop
regression" is mostly the extractor, not the model.

## 3. Degraded cases (loop2 correct → loop4 wrong, n=31)

### 3.1 Primary classification

| Class | Count | Share |
|---|---:|---:|
| Format degradation (truncation/repetition destroys the answer container) | 31 | 100% |
| Drift (answer moved away) | 0 | 0% |
| Over-correction (fixed a non-issue) | 0 primary / 13 secondary | 0% / 42% |
| Verbosity (longer without gain) | 0 primary / 20 co-occurring | 0% / 65% |

The four-class taxonomy is missing the actual dominant cause. Every degraded case is a
**truncation/extraction artifact**, which maps most closely to "format degradation" but is really a
fifth class: *the answer is present, but the grader cannot see it*.

Two subtypes:

- **GSM8K (18/31) — post-answer noise.** The correct value is present in 18/18 loop4 outputs, but
  it is never the last number (0/18). The model states the answer, then repeats the solution and
  intermediate values past the cut. Mean numbers appearing *after* the last target occurrence: 8.0
  (median 4); only 1/29 has zero. The target itself recurs ~10 times per output.
- **MATH (13/31) — lost answer container.** Loop4 outputs contain `\boxed{}` in 0/13 cases, versus
  7/13 at loop2. With no box, grading falls back to the last number, which is an intermediate.
  Example: idx129 (target 225) ends mid-sentence on a combinatorics digression; idx140 (target 12)
  loses a box that was present at loop2.

### 3.2 Drift: essentially absent

No degraded case produces a coherent alternative final answer. In 30/31 the correct value is present
in the loop4 output (18/18 GSM8K, 12/13 MATH). The lone exception, idx147 (target 5), is not drift
either: the output is stuck rereading the prompt ("The response should be exactly one valid example
of a rational function…"), a repetition collapse rather than a changed answer. First-answer-in-context
matches the target in 17/31, confirming the model usually starts on the right path.

### 3.3 Over-correction: a real but secondary signal

13/31 degraded outputs contain explicit doubt markers ("wait", "actually", "double-check",
"reconsider"). Examples: idx74 (6 markers; rounding $2.74 → 3, extracted 0.12), idx19 ("Wait, let me
double-check this interpretation", extracted 4 vs target 6), idx85 (age-range doubt, extracted 34 vs
target 44). In every case, however, the correct answer remains in the output — the revision does not
cause the grade flip. Over-correction is a *co-symptom* of the repetitive degeneration, not the
proximate cause.

### 3.4 Verbosity: present, not causal

Repetition rate rises in 20/31 degraded cases (mean 0.475 → 0.499) and all 31 are truncated, but
loop2 was already at the cap in 30/31. The model is not meaningfully more verbose at loop4; it is
more likely to be cut at a bad point.

## 4. Improved cases (loop2 wrong → loop4 correct, n=22)

| Class | Count |
|---|---:|
| Extraction lottery (target present in both loops) | 19 |
| Truncation recovery (target genuinely gained) | 3 |

19/22 improved cases contain the correct value in *both* loop2 and loop4 outputs; loop2's extractor
missed it, loop4's hit it. Post-target noise is 0 in 21/22 (mean 0.0), versus 8.0 for degraded —
the cleanest discriminator in the dataset. The 3 genuine gains (idx146, idx150, idx152) are all MATH
truncation recovery: loop2 stopped mid-derivation, loop4 reached the answer before the cut (e.g.
idx150 loop2 ends on "7x+53 ≡ 0 mod 8. Since 7"; loop4 completes to `\boxed{203}`).

## 5. Example patterns

- **Degraded, GSM8K (idx28, target 25):** loop4 states "45 − 20 = 25 miles", then repeats the
  explanation five times; the final number is 45, an intermediate position. Extracted 45, graded
  wrong.
- **Degraded, MATH (idx129, target 225):** loop2 ends on a digression about battalion composition;
  loop4 ends mid-sentence on subset selection. No `\boxed{}` in either; the box present at loop2 is
  gone at loop4.
- **Improved, GSM8K (idx3, target 540):** loop2 ends "Answer: 540 meters." then rereads the
  question; loop4 ends on the answer itself. Extracted 3 → 540.
- **Improved, MATH (idx150, target 203):** loop2 truncates inside a modular-arithmetic step; loop4
  completes and boxes 203.

## 6. Mitigation recommendations

1. **Fix the extractor first.** Replace whole-output `extract_last_number` with answer-position
   extraction: the number in the last `...` block, the last `Answer:` / `The answer is` match, or
   `\boxed{}`. This alone would recover most of the 84→96 false negatives.
2. **Stop at the answer boundary.** Add stop sequences (``, `\boxed{`, `Answer:`) or raise
   max_tokens. 199/200 outputs hitting the cap means the current budget measures truncation position,
   not competence.
3. **Suppress repetition.** Add `repetition_penalty` or `no_repeat_ngram_size` to SamplingParams.
   Repetition is the root cause that pushes the answer past the cut.
4. **Grade truncation separately.** Treat "output hit the token cap" as its own failure mode
   (incomplete/format) rather than folding it into correctness.
5. **Report content and graded accuracy side by side.** The -4.5pp loop gain is extraction noise;
   content accuracy is +2.5pp. Do not conclude that loop scaling hurts reasoning from this benchmark.
6. **Protect the MATH box.** Loop4 lost `\boxed{}` in 7/13 degraded MATH cases that had it at loop2;
   a stop-at-boxed or format reward would preserve the container.

## 7. Limitations

The content-accuracy proxy (target value present anywhere) can credit coincidental intermediate
values; for non-numeric MATH targets it was checked by normalized string match. The extraction
analysis assumes the checker in `experiments/05_hard_loop.py` is the grader of record. A re-run with
a robust answer-position grader and repetition penalty is the decisive next step.
