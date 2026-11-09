# Dory Experiments on Kaggle 2×T4 — 20h Lap

**Goal:** Push NVIDIA-Nemotron-Labs-Dory1.0-4B (looped normalized-GPT, 4B, elastic 1–4 loops, 256K ctx, NVFP4) to its limits on 2× T4 GPUs, and ship breakthroughs.

**Team (4 agents):**
- **Beam** (conductor) — `nous-portal/reflection/beam-1.0-501b-a23b:free` — E1 harness, benchmark suite, final ranking, integration
- **LongCat** — `nous-portal/meituan/longcat-2.5:free` — memory/quantization, NVFP4, 256K NIAH, KV sharding
- **Crouton Alpha** — `nous-portal/stealth/crouton-alpha` — learning, difficulty router, distillation, curriculum
- **Iceman** — `opencode/iceman-free` — systems, speculative decoding, KV cache, throughput, adversarial

**Hardware:** Kaggle 2× T4 (16GB each, 32GB total), Turing (sm_75), FP16/INT8 only — **no FP4 tensor cores**. `/tmp` + git + gh as `10xdev4u-alt`.

**Repo:** https://github.com/10xdev4u-alt/dory-experiments

**Lap window:** started Mon 2026-11-09 08:39 IST → ends Tue 2026-11-10 04:39 IST (20h).

**Model:** `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` (HF) + arXiv 2611.03998.

## Quick start on Kaggle
```bash
git clone https://github.com/10xdev4u-alt/dory-experiments.git
cd dory-experiments
pip install -r requirements.txt
python experiments/01_loop_ladder.py
```

## Plan
See [PLAN.md](PLAN.md). Team roles: [TEAM.md](TEAM.md). Lap log: [LOG.md](LOG.md).
