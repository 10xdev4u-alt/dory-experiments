# E4: Difficulty router (escalation predictor)

- Checkpoint: `results/e5/checkpoints.jsonl`
- Backend: `pure_python`
- Samples: `200`
- Escalation positives: `22` (11.000%)

## Metrics (out-of-fold, 5-fold CV)

- Classifier accuracy: **0.5300**
- Misroute rate: **0.4700**
- Net accuracy if router applied: **0.2200**
- Oracle capture %: **63.64%**

## Baselines

- Fixed loop2 accuracy: 0.2250
- Fixed loop4 accuracy: 0.1800
- Oracle (best of 2/4) accuracy: 0.3350

## Confusion matrix (escalation)

- TP: 14  FP: 86
- FN: 8  TN: 92

## Feature coefficients

- prompt_len_chars: -2.6927
- token_count_est: 2.5418
- has_equal: -0.1631
- has_plus: -0.8164
- has_digit: 0.0206
- has_math_marker: 0.0206
- family_gsm8k: -0.1418
- family_math: 0.1418
- family_bbh: 0.0000
