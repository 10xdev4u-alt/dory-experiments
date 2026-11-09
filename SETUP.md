# Dory Experiments — Setup Guide

This guide covers running the Dory experiments on **Kaggle** (recommended for GPU access) and on a **local machine**.

## Kaggle setup (GPU)

1. **Create a notebook**
   - Go to [Kaggle](https://www.kaggle.com/) → *Code* → *New Notebook*.
   - Choose a Python notebook.

2. **Enable hardware & internet**
   - In the right sidebar: *Settings* → *Accelerator* → select **GPU T4 x2** (or the best available T4 option).
   - Turn **Internet** to *On* (required to clone the repo and install packages).

3. **Clone the repository**
   ```bash
   !git clone <REPO_URL> /kaggle/working/dory-experiments
   %cd /kaggle/working/dory-experiments
   ```

4. **Install dependencies**
   ```bash
   !pip install -q -r requirements.txt
   ```
   Kaggle images often preinstall PyTorch; if you hit CUDA/torch conflicts, install torch first to match the image, then the rest of `requirements.txt`.

5. **Run experiments**
   - Execute the experiment entrypoint(s) documented in the repo (e.g., `python run_experiment.py --config ...`).
   - Pass output paths under `/kaggle/working` so artifacts persist:
     ```bash
     !python run_experiment.py --output_dir /kaggle/working/outputs
     ```

6. **Save and download outputs**
   - Kaggle keeps files in `/kaggle/working` after the session (and in notebook output when you *Save Version*).
   - Download from the notebook’s *Output* tab, or zip first:
     ```bash
     !zip -r /kaggle/working/results.zip /kaggle/working/outputs
     ```

## Local setup

1. **Prerequisites**: Python 3.10+ and (for GPU runs) a CUDA-capable NVIDIA driver/toolkit compatible with your PyTorch build.
2. **Create an environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Clone and run**:
   ```bash
   git clone <REPO_URL>
   cd dory-experiments
   python run_experiment.py --output_dir ./outputs
   ```
   If you only have CPU, expect slow runs and possible incompatibilities with quantized checkpoints; prefer Kaggle/Colab GPUs for full experiments.

## Git identity & GitHub CLI (gh)

Configure git before committing or pushing from any environment:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Authenticate with GitHub CLI (useful for cloning private forks, opening PRs, or releasing artifacts):

```bash
gh auth login
gh auth status
```

On Kaggle, store tokens via *Add-ons → Secrets* rather than hardcoding them in notebooks.

## Troubleshooting

- **OOM (out of memory)**: Reduce batch size / sequence length, enable gradient checkpointing if supported, use `accelerate` device mapping, or load models in 4/8-bit via `bitsandbytes` (where supported). On Kaggle, confirm you selected **GPU T4 x2** and that the experiment is actually using CUDA (`torch.cuda.is_available()`).
- **NVFP4 / FP4/NVFP4 errors**: NVFP4 kernels and weights typically require newer NVIDIA architectures and matching library/toolchain support. T4 GPUs do **not** support NVFP4. Use BF16/FP16/INT8/INT4 (NF4 via bitsandbytes) alternatives, or run on hardware/software stacks that explicitly support NVFP4.
- **Loop count not applied**: Verify the config/CLI flag name matches what the script parses (e.g., `--num_loops`, `--loop_count`, or a YAML key). Print the resolved config at startup, ensure no default overrides it later, and confirm you’re editing the config file/path the run actually loads. If running in a notebook, restart the kernel after changing code or config modules.

Replace `<REPO_URL>` and placeholder identity values with your project’s actual details.
