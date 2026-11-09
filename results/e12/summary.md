# E12: Router rerun on position-aware (E11) re-extracted labels

- Checkpoint: `results/e5/checkpoints.jsonl`
- Backend: `pure_python`
- Samples: `200`
- Escalation positives: `17` (8.500%)

## Metrics (out-of-fold, 5-fold CV)

- Classifier accuracy: **0.5650**
- Misroute rate: **0.4350**
- Net accuracy if router applied: **0.4750**
- Oracle capture %: **47.06%**

## Baselines (re-extracted labels)

- Fixed loop2 accuracy: 0.4600
- Fixed loop4 accuracy: 0.4650
- Oracle (best of 2/4) accuracy: 0.5450

## Confusion matrix (escalation)

- TP: 8  FP: 78
- FN: 9  TN: 105

## Feature coefficients

- log_len: 0.1615
- digit_density: 0.0534
- math_symbol: -0.3677
- question_marks: 0.2642
- task_type: -0.6567
