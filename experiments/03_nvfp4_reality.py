#!/usr/bin/env python3
"""E3: NVFP4 reality check on T4 + INT8/INT4 quantization.

Loads Dory (default nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16), inspects the
checkpoint dtypes, probes for true NVFP4 hardware support, and benchmarks
BF16/FP16 vs INT8 (bitsandbytes) vs an NVFP4-emulated 4-bit path. The emulation
stores 4-bit E2M1 codes (two per byte) with per-16 fp16 scales, dequantized to
the compute dtype every forward. Outputs <out>/e3_results.json and e3_summary.md.
"""
from __future__ import annotations

import argparse, gc, json, math, os, sys, time, traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
DEFAULT_OUT = "results/e3"
FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_BLOCK = 16
NVFP4_MIN_CC = (10, 0)
ACTIVATION_RESERVE = 1.5 * 1024**3
SPEED_PROMPT = "The quick brown fox jumps over the lazy dog."
FALLBACK_TEXT = (
    "The NVIDIA T4 is a Turing GPU with 16 GB GDDR6 and INT8 tensor cores but no "
    "native FP4/NVFP4 path. NVFP4 needs compute capability 10.0+ (Blackwell). 4-bit "
    "quantization still cuts weight memory, but on T4 compute stays BF16/FP16 and "
    "dequantization overhead can cancel the bandwidth win. INT8 is the practical "
    "low-precision option on Turing."
)
MC_EXAMPLES = [
    ("Capital of France?", ["Paris", "London", "Berlin", "Madrid"], 0),
    ("2 + 3 = ?", ["4", "5", "6", "7"], 1),
    ("The Red Planet?", ["Venus", "Mars", "Jupiter", "Saturn"], 1),
    ("Largest ocean?", ["Atlantic", "Indian", "Arctic", "Pacific"], 3),
    ("Symbol for gold?", ["Ag", "Au", "Gd", "Go"], 1),
    ("Runs natively in a browser?", ["Python", "C++", "JavaScript", "Java"], 2),
]

def log(msg):
    print(f"[e3] {msg}", flush=True)

def device_info():
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "name": None, "cc": None}
    if info["cuda"]:
        info.update(name=torch.cuda.get_device_name(0),
                    cc=list(torch.cuda.get_device_capability(0)))
    return info

def nvfp4_support():
    out = {"cc": None, "probe": None, "supported": False, "reason": ""}
    if torch.cuda.is_available():
        out["cc"] = list(torch.cuda.get_device_capability(0))
        out["supported"] = tuple(out["cc"]) >= NVFP4_MIN_CC
    try:
        out["probe"] = bool(torch._C._get_nvfp4_device_support())
        out["supported"] = out["supported"] and out["probe"]
    except Exception as exc:
        out["probe"] = f"unavailable: {type(exc).__name__}"
    if not out["supported"]:
        why = "no CUDA device" if out["cc"] is None else f"compute capability {out['cc']}"
        out["reason"] = f"{why}; NVFP4 tensor-core dequantization requires Blackwell (SM100+)"
    return out

def checkpoint_dtypes(model_id):
    out = {"dtypes": None, "float4": False, "uint8": False, "nvfp4": False, "error": None}
    try:
        from huggingface_hub import HfApi
        st = getattr(HfApi().model_info(model_id, safetensors=True), "safetensors", None)
        if st is None:
            out["error"] = "no safetensors metadata"
            return out
        out["dtypes"] = dict(getattr(st, "parameters", {}) or {})
        keys = [k.lower() for k in out["dtypes"]]
        out["float4"] = any("fp4" in k or "float4" in k for k in keys)
        out["uint8"] = any("uint8" in k or "int8" in k for k in keys)
        out["nvfp4"] = any("nvfp4" in k for k in keys)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out

def inspect_dtypes(model):
    ph, bh = {}, {}
    for p in model.parameters():
        ph[str(p.dtype)] = ph.get(str(p.dtype), 0) + 1
    for _, b in model.named_buffers():
        bh[str(b.dtype)] = bh.get(str(b.dtype), 0) + 1
    samples = [{"name": n, "class": type(m).__name__,
                "dtypes": sorted({str(p.dtype) for p in m.parameters(recurse=False)})}
               for n, m in list(model.named_modules())[:24] if n]
    try:
        cfg = json.dumps(model.config.to_dict()).lower()
    except Exception:
        cfg = ""
    return {"param_dtypes": dict(sorted(ph.items())), "buffer_dtypes": dict(sorted(bh.items())),
            "has_float4": any("float4" in k for k in ph) or any("float4" in k for k in bh),
            "has_uint8": any("uint8" in k for k in ph) or any("uint8" in k for k in bh),
            "has_nvfp4_marker": "nvfp4" in cfg,
            "sample_modules": samples}

def _mags(device, dtype):
    return torch.tensor(FP4_MAGNITUDES, dtype=dtype, device=device)

def _quantize_fp4(w):
    """NVFP4-style block quantization: E2M1 codes + per-16 fp16 scales."""
    out_f, in_f = w.shape
    if (-in_f) % FP4_BLOCK:
        w = F.pad(w, (0, (-in_f) % FP4_BLOCK))
    blocks = w.reshape(out_f, -1, FP4_BLOCK)
    scale = (blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 6.0).to(torch.float16)
    q = blocks / scale.to(blocks.dtype)
    idx = (q.abs().unsqueeze(-1) - _mags(q.device, q.dtype)).abs().argmin(dim=-1).to(torch.uint8)
    idx = idx | ((q < 0).to(torch.uint8) << 3)  # bit 3 = sign
    return (idx[..., 0::2] | (idx[..., 1::2] << 4)).contiguous(), scale.squeeze(-1).contiguous()

def _dequantize_fp4(packed, scale, in_f, dtype):
    """Inverse of _quantize_fp4; returns a [out, in_f] tensor in `dtype`."""
    code = torch.stack([(packed & 0x0F).to(torch.int8), (packed >> 4).to(torch.int8)], dim=-1)
    code = code.reshape(packed.shape[0], -1)
    mag = _mags(code.device, dtype)[(code & 0x07).long()].to(dtype)
    sign = torch.where(code >= 8, -1.0, 1.0).to(dtype)
    return (mag * sign * scale.repeat_interleave(FP4_BLOCK, dim=1).to(dtype))[:, :in_f]

class FP4Linear(nn.Module):
    """nn.Linear replacement: 4-bit E2M1 codes + per-16 fp16 scales, dequantized in forward."""

    def __init__(self, linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bias = linear.bias
        packed, scale = _quantize_fp4(linear.weight.detach())
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)

    def forward(self, x):
        w = _dequantize_fp4(self.packed, self.scale, self.in_features, x.dtype).to(x.device)
        return F.linear(x, w, self.bias.to(x.device) if self.bias is not None else None)

def apply_fp4(model):
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, attr, FP4Linear(module))
            n += 1
    return n

def weight_bytes(model):
    return (sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for _, b in model.named_buffers()))

def load_model(model_id, device, dtype, quant):
    try:
        import transformers.configuration_utils as _cu
        if "mlp" not in getattr(_cu, "ALLOWED_ATTN_LAYER_TYPES", ()):
            _cu.ALLOWED_ATTN_LAYER_TYPES = _cu.ALLOWED_ATTN_LAYER_TYPES + ("mlp",)
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    common = dict(device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=True)
    if quant == "int8":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True), **common)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16 if dtype == "bf16" else torch.float16, **common)
    model.eval()
    return model, tok

def load_eval_text():
    try:
        from datasets import load_dataset
        rows = [r for r in load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")["text"]
                if r and r.strip()][:20]
        if rows:
            return "\n\n".join(rows), "wikitext-2-raw-v1 validation[:20]"
    except Exception as exc:
        log(f"WikiText unavailable ({type(exc).__name__}); using fallback text")
    return FALLBACK_TEXT, "built-in fallback"

def perplexity_and_accuracy(model, tok, text, max_len=512):
    ids = tok(text, return_tensors="pt").input_ids[:, :max_len].to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
    return math.exp(nll), (logits[:, :-1].argmax(dim=-1) == tgt).float().mean().item()

def mc_accuracy(model, tok, examples, max_len=256):
    device = next(model.parameters()).device
    correct = 0
    for prompt, choices, gold in examples:
        pids = tok(prompt, return_tensors="pt").input_ids
        scores = []
        for choice in choices:
            ids = torch.cat([pids, tok(" " + choice, return_tensors="pt").input_ids], dim=1)[:, :max_len].to(device)
            plen = min(pids.shape[1], ids.shape[1] - 1)
            with torch.no_grad():
                logp = F.log_softmax(model(input_ids=ids).logits[:, plen - 1:-1].float(), dim=-1)
            scores.append(logp.gather(-1, ids[:, plen:].unsqueeze(-1)).squeeze(-1).sum().item())
        correct += int(max(range(len(scores)), key=lambda i: scores[i]) == gold)
    return correct / len(examples)

def benchmark(model, tok, prompt, max_new_tokens, warmup=2):
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        for _ in range(warmup):
            model.generate(ids, max_new_tokens=2, do_sample=False, pad_token_id=pad)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return {"tok_s": round(max_new_tokens / max(dt, 1e-9), 2)}

def kv_stats(model, wbytes, total_vram):
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", None)
    layers = getattr(cfg, "num_hidden_layers", None)
    heads = getattr(cfg, "num_attention_heads", None)
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    head_dim = getattr(cfg, "head_dim", None) or (hidden // heads if hidden and heads else None)
    rate = 2 * layers * kv_heads * head_dim * 2 if layers and kv_heads and head_dim else None
    usable = max(0, total_vram - wbytes - ACTIVATION_RESERVE) if rate and total_vram else 0
    headroom = {"usable_gb": round(usable / 1024**3, 3), "max_tokens": int(usable // rate)} if rate and total_vram else None
    return {"kv_bytes_per_token": rate, "headroom": headroom}

def run_mode(mode, args, eval_text):
    res = {"mode": mode, "status": "ok", "weights_gb": None, "cuda_mem_gb": None, "tok_s": None,
           "perplexity": None, "next_token_accuracy": None, "mc_accuracy": None,
           "kv_bytes_per_token": None, "kv_max_tokens": None, "dtype_used": None,
           "errors": [], "notes": []}
    model = None
    try:
        if mode == "int8":
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                res.update(status="skipped", errors=[f"bitsandbytes not installed: {exc}"])
                return res
            if not torch.cuda.is_available():
                res.update(status="skipped", errors=["bitsandbytes INT8 requires a CUDA GPU"])
                return res
        quant = "int8" if mode == "int8" else "none"
        dtype = mode if mode in ("bf16", "fp16") else args.dtype
        log(f"[{mode}] loading {args.model} dtype={dtype} quant={quant}")
        model, tok = load_model(args.model, args.device, dtype, quant)
        res["dtype_used"] = str(next(model.parameters()).dtype)
        if mode == "nvfp4_emulated":
            n = apply_fp4(model)
            res["notes"].append(f"replaced {n} nn.Linear modules with NVFP4-emulated wrappers")
            if n == 0:
                res.update(status="skipped", errors=["no nn.Linear modules found to quantize"])
                return res
        wb = weight_bytes(model)
        res["weights_gb"] = round(wb / 1024**3, 3)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            res["cuda_mem_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        if mode in ("bf16", "fp16"):
            res["dtype_inspection"] = inspect_dtypes(model)
        vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        kv = kv_stats(model, wb, vram)
        res["kv_bytes_per_token"] = kv["kv_bytes_per_token"]
        res["kv_max_tokens"] = (kv["headroom"] or {}).get("max_tokens")
        res["kv_detail"] = kv
        res.update(benchmark(model, tok, SPEED_PROMPT, args.max_tokens))
        res["perplexity"], res["next_token_accuracy"] = perplexity_and_accuracy(model, tok, eval_text)
        res["mc_accuracy"] = mc_accuracy(model, tok, MC_EXAMPLES)
    except Exception as exc:
        res.update(status="error", errors=[f"{type(exc).__name__}: {exc}"], traceback=traceback.format_exc())
        log(f"[{mode}] ERROR: {type(exc).__name__}: {exc}")
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res

def _fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))

def render_md(results):
    dev, nv, ck = results["device"], results["nvfp4"], results["checkpoint_file"]
    lines = [
        "# E3: NVFP4 reality check on T4 + INT8/INT4 quantization", "",
        f"- Model: `{results['model']}` | Device: {dev.get('name') or 'CPU'} "
        f"(CC {dev.get('cc') or 'n/a'}) | Torch {dev.get('torch')}",
        f"- NVFP4 hardware supported: **{nv.get('supported')}** ({nv.get('reason') or 'n/a'})",
        f"- Eval text: {results.get('eval_text_source')}", "",
        "## Checkpoint dtype inspection", "",
        f"- BF16 repo stored dtypes: `{ck.get('dtypes')}` | "
        f"float4: {ck.get('float4')} | uint8/int8: {ck.get('uint8')} | NVFP4 marker: {ck.get('nvfp4')}", "",
        "## Benchmarks", "",
        "| Mode | Weights GB | CUDA mem GB | tok/s b=1 | PPL | Next-token acc | MC acc | KV B/token | Max KV tokens | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    keys = ("weights_gb", "cuda_mem_gb", "tok_s", "perplexity", "next_token_accuracy",
            "mc_accuracy", "kv_bytes_per_token", "kv_max_tokens")
    for mode, r in results["modes"].items():
        lines.append("| " + " | ".join([mode] + [_fmt(r.get(k)) for k in keys] + [r.get("status", "?")]) + " |")
    lines += ["", "## Notes", ""]
    for mode, r in results["modes"].items():
        for n in r.get("notes", []):
            lines.append(f"- **{mode}**: {n}")
        for e in r.get("errors", []):
            lines.append(f"- **{mode}** error: {e}")
    for e in results.get("errors", []):
        lines.append(f"- global error: {e}")
    return "\n".join(lines) + "\n"

def main(argv=None):
    ap = argparse.ArgumentParser(description="E3: NVFP4 reality check on T4 + INT8/INT4 quantization")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    ap.add_argument("--max-tokens", type=int, default=64, help="tokens for the batch-1 speed benchmark")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16", help="full-precision compute dtype")
    ap.add_argument("--device", default="cuda:0", help="torch device")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if not torch.cuda.is_available():
        args.device = "cpu"
    dev, nv = device_info(), nvfp4_support()
    ck = checkpoint_dtypes(args.model)
    text, src = load_eval_text()
    log(f"device: {dev['name'] or 'CPU'} CC={dev['cc'] or 'n/a'} torch={dev['torch']}")
    log(f"NVFP4 supported: {nv['supported']} ({nv['reason']}) | ckpt dtypes: {ck.get('dtypes')}")
    log(f"eval text: {src} ({len(text)} chars)")
    results = {"model": args.model, "dtype_requested": args.dtype, "max_new_tokens": args.max_tokens,
               "device": dev, "nvfp4": nv, "checkpoint_file": ck, "eval_text_source": src,
               "modes": {}, "errors": []}
    for mode in (args.dtype, "int8", "nvfp4_emulated"):
        r = run_mode(mode, args, text)
        results["modes"][mode] = r
        log(f"[{mode}] status={r['status']} weights={r.get('weights_gb')}GB tok/s={r.get('tok_s')} "
            f"ppl={r.get('perplexity')} mc={r.get('mc_accuracy')}")
    jp, mp = os.path.join(args.out, "e3_results.json"), os.path.join(args.out, "e3_summary.md")
    with open(jp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(mp, "w") as f:
        f.write(render_md(results))
    log(f"wrote {jp} and {mp}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
