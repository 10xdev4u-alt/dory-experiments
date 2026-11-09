# E8: Confidence-Gated Escalation

- Checkpoint: `results/e5/checkpoints.jsonl`
- Threshold: `0.5`
- Confidence method: `prompt_length_heuristic (median split: 1.0 if len(question) <= median else 0.0)`
- Loop1 available: `False`
- Samples: `200`
- Escalated to loop4: `99` (49.50%)

## Accuracy comparison

| Strategy | Accuracy |
|---|---:|
| Static loop2 | 0.2250 |
| Static loop4 | 0.1800 |
| Confidence-gated | 0.2300 |
| Oracle (best of 2/4) | 0.3350 |

Gain vs static loop2: +0.0050
Gap to oracle: 0.1050

## Family counts

- gsm8k: 100
- math: 100
