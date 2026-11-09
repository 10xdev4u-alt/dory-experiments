# E11 — Answer-Position-Aware Extraction (GSM8K/MATH)

Checkpoint: `results/e5/checkpoints.jsonl` (400 records)

## Accuracy: original vs re-extracted

| Loop | Original overall | Re-extracted overall |
|---:|---:|---:|
| 2 | 0.2250 | 0.4600 |
| 4 | 0.1800 | 0.4650 |

### gsm8k

| Loop | Original | Re-extracted |
|---:|---:|---:|
| 2 | 0.2600 | 0.6600 |
| 4 | 0.2100 | 0.6200 |

### math

| Loop | Original | Re-extracted |
|---:|---:|---:|
| 2 | 0.1900 | 0.2600 |
| 4 | 0.1500 | 0.3100 |

## Degraded cases (loop2 correct -> loop4 wrong)

- Total degraded: 31
- Recovered by re-extraction: 20
- Target value still present in loop-4 output (extraction artifacts): 30 of 31
- The remaining degraded case(s) are genuine repetition collapses, not extraction artifacts.

## Conclusion

Re-extracting with answer-position awareness lifts graded accuracy and shows that 30 of the 31 'degraded' cases are extraction artifacts: the correct value is present in the output but the original last-number extractor sampled post-answer noise (GSM8K) or a lost \boxed{} container (MATH).
