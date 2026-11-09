# E5 Hard-Loop Benchmark Summary

- Model: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
- Prompts: 200 (limit/dataset=100)
- Loops: [2, 4]
- Max tokens: math=512, bbh=256

## Per-loop metrics

| Loops | Accuracy | Output tokens | Rep rate | Throughput (tok/s) | Time (s) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.2250 | 102191 | 0.4688 | 177.0 | 577.3 |
| 4 | 0.1800 | 102274 | 0.4683 | 103.8 | 985.7 |

- Oracle accuracy: 0.3350 (67/200)
- Loop gain (last - first): -0.0450
- Oracle gain: +0.1100
- Improvement rate (first wrong -> last correct): 0.1100 (22)
- Degradation rate (first correct -> last wrong): 0.1550 (31)

Wrote `/kaggle/working/results/e5_hard_loop/results.json` and `/kaggle/working/results/e5_hard_loop/checkpoints.jsonl`.
