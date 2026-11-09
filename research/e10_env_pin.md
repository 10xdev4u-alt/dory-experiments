# E10 Quantization Retry — Environment Pin for Kaggle T4

**Purpose.** E10 retries Dory INT8 inference on a single Kaggle T4 (sm_75, 16 GiB). This guide pins working versions, avoids the `Config() got deprecated kwarg` crash, and documents the GGUF fallback. Use a fresh Kaggle kernel.

## 1. Why the previous environment failed

The repo `requirements.txt` pins `torch>=2.4.0,<2.7.0`, `transformers>=4.45.0,<4.50.0`, and `bitsandbytes>=0.44.0,<0.46.0`. Those pins predate Dory support and conflict with vLLM v0.33.0:

1. **Transformers mismatch.** vLLM v0.33.0 requires `transformers>=5.16.1,<5.21.0`. With Transformers 4.x importable, vLLM's config path hits `Config() got deprecated kwarg` — a renamed/removed `PretrainedConfig` keyword in the 5.x API. This is version skew, not a Dory bug.
2. **BitsAndBytes mismatch.** `vllm-bnb-plugin` 0.0.3 requires `bitsandbytes>=0.48.1`. The repo pin (`<0.46.0`) prevents `quantization="bitsandbytes"` from registering, so INT8 loads fail.

Do not `pip install -r requirements.txt` in the E10 kernel. Use the commands in §3 instead.

## 2. The pinned stack

| Package | Pin | Why |
|---|---|---|
| vLLM | `0.33.0` (released 2026-11-02) | First release with Dory support from PR #60149 (merged 2026-10-25). The model-card PR-install note is stale — use the release. |
| PyTorch | `2.14.0` | Exact pin from vLLM v0.33.0 `requirements/cuda.txt`. |
| CUDA | `13.0.3` | vLLM v0.33.0 `docker/Dockerfile` pins `CUDA_VERSION=13.0.3`. The wheel compiles kernels for sm_75, so T4 is supported compute-side. |
| Transformers | `>=5.16.1,<5.21.0` | Required by vLLM v0.33.0. Fixes the `Config()` error. |
| bitsandbytes | `>=0.48.1` | Minimum for `vllm-bnb-plugin` 0.0.3. |
| vllm-bnb-plugin | `0.0.3` | Registers `quantization="bitsandbytes"` via the `vllm.general_plugins` entry-point. |
| Python | `>=3.11,<3.15` | Kaggle's default 3.11/3.12 is fine. |

**Driver check first.** CUDA 13 needs an R580+ driver. Run `!nvidia-smi` and check. If the driver is in the R535/R570 CUDA forward-compatibility range, set compatibility mode *before* importing vLLM:

```python
import os
os.environ["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "1"
```

If the driver is too old for either mode, stop and use the GGUF fallback in §5. Do not downgrade to a CUDA 12.9 vLLM in this kernel — that is a separate experiment.

## 3. Install commands (fresh Kaggle kernel)

Enable GPU T4 x1 (Dory is TP=1 only) and Internet, then install in order — Torch first to match the CUDA 13.0.3 image, then vLLM, then the quantization stack:

```bash
!nvidia-smi
!python --version
!pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
!pip install -q "vllm==0.33.0"
!pip install -q "transformers>=5.16.1,<5.21.0" "bitsandbytes>=0.48.1" "vllm-bnb-plugin==0.0.3"
```

If `pip` warns about `transformers` or `bitsandbytes` conflicts, the old repo pins are still present. Run `!pip uninstall -y torch transformers bitsandbytes` and retry in a fresh kernel rather than layering installs. Skip NVFP4/MXFP8 tooling — it does not run on sm_75.

## 4. Verification checks

Run in order; do not skip to the model load if any check fails.

**4.1 Import and version check.** Proves the `Config()` mismatch is gone and the CUDA stack is live:

```python
import torch, vllm, transformers, bitsandbytes
import importlib.metadata as m
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "bnb", bitsandbytes.__version__)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert vllm.__version__ == "0.33.0" and torch.__version__.startswith("2.14.0")
```

Expect `available True` and capability `(7, 5)` on T4. If `is_available()` is `False`, the kernel has no GPU enabled — fix Kaggle Settings (GPU T4) and restart.

**4.2 Dory INT8 smoke test.** Smallest full-path test — 4K context, eager mode, `last_loop` KV mode:

```python
from vllm import LLM, SamplingParams
llm = LLM(
    model="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
    quantization="bitsandbytes", load_format="bitsandbytes",
    max_model_len=4096, gpu_memory_utilization=0.85,
    enforce_eager=True,
    hf_overrides={"n_recurrent_loops": 2, "recurrent_kv_cache_mode": "last_loop"},
)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
print(out[0].outputs[0].text)
```

Success is a coherent completion with no `Config()` traceback and no `quantization not supported` error. `last_loop` is mandatory: it holds KV at one-loop size (~1.23 GiB at 32K), so INT8 weights (3.70 GiB) + KV fits with headroom. `per_loop` multiplies recurrent-layer KV and will OOM at long-context rungs.

## 5. Fallback: GGUF / llama.cpp (PR #30306)

Switch here if the driver check fails, or vLLM OOMs at 128K+ from activation/workspace overhead (the E2 failure mode — KV alone at 128K is only 4.61 GiB). Dory support lives in [ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306), still Draft, head `1c3cdd23787750814aa21a070dc4d6dc451af871`. Build from source:

```bash
!git clone https://github.com/ggml-org/llama.cpp.git /kaggle/working/llama.cpp
%cd /kaggle/working/llama.cpp
!git fetch origin pull/30306/head:pr-30306 && git checkout pr-30306
!cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_GRAPHS=OFF
!cmake --build build -j"$(nproc)" --target llama-cli llama-server
```

Download the Q5_K_M export (3,673,288,864 bytes = 3.42 GiB) from `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`, then run:

```bash
!/kaggle/working/llama.cpp/build/bin/llama-cli \
  -m /kaggle/working/…-Q5_K_M.gguf \
  --ctx-size 262144 --no-context-shift --flash-attn off \
  -p "The capital of France is" -n 16
```

llama.cpp has no vLLM workspace overhead, so GGUF Q5_K_M + `last_loop` is the cheapest 256K path (~12.53 GiB). Report the prefill chunk size with every result — the PR notes logits shift with chunk size, and K>4 loops plus context shifting are unsupported.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Config() got deprecated kwarg` | `transformers` 4.x shadowing 5.x | Fresh kernel; `transformers>=5.16.1,<5.21.0`; never install repo `requirements.txt` here |
| `quantization="bitsandbytes"` not recognized | Old `bitsandbytes` or missing plugin | `bitsandbytes>=0.48.1`, `vllm-bnb-plugin==0.0.3` |
| CUDA/driver error on import | Driver older than R580 | `VLLM_ENABLE_CUDA_COMPATIBILITY=1`, or use §5 |
| OOM at 128K with BF16 | BF16 (7.40 GiB) + workspace exceeds T4 | INT8/GGUF + `last_loop` + `enforce_eager=True` |

## Sources

vLLM v0.33.0 release and Dockerfile (`CUDA_VERSION=13.0.3`); `requirements/cuda.txt` (`torch==2.14.0`); PR #60149 (Dory support, merged 2026-10-25); `vllm-bnb-plugin` 0.0.3 (PyPI, 2026-08-24); llama.cpp PR #30306 (Draft); GGUF export `NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`. See `research/dory_vllm_deep_dive.md` for the KV-allocation model behind these numbers.
