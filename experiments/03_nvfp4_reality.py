#!/usr/bin/env python3
"""E3: vLLM backend reality check on T4 + quantization modes.

Loads Dory (default nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) via vLLM and
benchmarks supported modes: bf16/fp16 and bitsandbytes 4-bit/8-bit when vLLM
supports them. Records weights (best-effort), throughput, perplexity, and
multiple-choice accuracy. Outputs <out>/e3_results.json and e3_summary.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
DEFAULT_OUT = "results/e3"
VLLM_REF = "git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head"
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


def log(msg: str) -> None:
    print(f"[e3] {msg}", flush=True)


def ensure_vllm() -> None:
    try:
        import vllm  # noqa: F401

        return
    except ImportError:
        os.environ["VLLM_USE_PRECOMPILED"] = "1"
        log(f"vLLM not found; installing {VLLM_REF}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", VLLM_REF])


def device_info() -> dict:
    try:
        import torch

        info = {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "name": None,
            "cc": None,
        }
        if info["cuda"]:
            info.update(
                name=torch.cuda.get_device_name(0),
                cc=list(torch.cuda.get_device_capability(0)),
            )
        return info
    except Exception as exc:
        return {"torch": None, "cuda": False, "name": None, "cc": None, "error": str(exc)}


def nvfp4_support() -> dict:
    out = {"cc": None, "probe": None, "supported": False, "reason": ""}
    try:
        import torch

        if torch.cuda.is_available():
            out["cc"] = list(torch.cuda.get_device_capability(0))
            out["supported"] = tuple(out["cc"]) >= (10, 0)
        try:
            out["probe"] = bool(torch._C._get_nvfp4_device_support())
            out["supported"] = out["supported"] and out["probe"]
        except Exception as exc:
            out["probe"] = f"unavailable: {type(exc).__name__}"
        if not out["supported"]:
            why = "no CUDA device" if out["cc"] is None else f"compute capability {out['cc']}"
            out["reason"] = f"{why}; NVFP4 tensor-core dequantization requires Blackwell (SM100+)"
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def checkpoint_dtypes(model_id: str) -> dict:
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


def load_eval_text() -> tuple[str, str]:
    try:
        from datasets import load_dataset

        rows = [
            r
            for r in load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")["text"]
            if r and r.strip()
        ][:20]
        if rows:
            return "\n\n".join(rows), "wikitext-2-raw-v1 validation[:20]"
    except Exception as exc:
        log(f"WikiText unavailable ({type(exc).__name__}); using fallback text")
    return FALLBACK_TEXT, "built-in fallback"


def _hf_config(llm):
    for path in (
        ("llm_engine", "model_config", "hf_config"),
        ("llm_engine", "model_config", "hf_text_config"),
    ):
        obj = llm
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    return None


def _estimate_weight_bytes(hf_config, dtype: str | None) -> int | None:
    if hf_config is None:
        return None
    try:
        hidden = getattr(hf_config, "hidden_size", None)
        layers = getattr(hf_config, "num_hidden_layers", None)
        vocab = getattr(hf_config, "vocab_size", None)
        inter = getattr(hf_config, "intermediate_size", None)
        heads = getattr(hf_config, "num_attention_heads", None)
        kv_heads = getattr(hf_config, "num_key_value_heads", heads)
        if not all(isinstance(x, int) and x > 0 for x in (hidden, layers, vocab, inter, heads, kv_heads)):
            return None
        head_dim = getattr(hf_config, "head_dim", None) or (hidden // heads)
        q = layers * hidden * (heads * head_dim)
        kv = layers * hidden * (2 * kv_heads * head_dim)
        o = layers * (heads * head_dim) * hidden
        mlp = layers * 3 * hidden * inter
        emb = vocab * hidden
        params = q + kv + o + mlp + emb
        bpe = 2
        if dtype and ("float16" in str(dtype).lower() or "bfloat16" in str(dtype).lower() or "bf16" in str(dtype).lower() or "fp16" in str(dtype).lower()):
            bpe = 2
        return int(params * bpe)
    except Exception:
        return None


def _weight_bytes_best_effort(llm, mode: str) -> tuple[float | None, str | None]:
    # Try to reach the underlying torch module (fragile across vLLM versions).
    candidates = [
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"),
    ]
    for path in candidates:
        obj = llm
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None:
            try:
                total = sum(p.numel() * p.element_size() for p in obj.parameters())
                total += sum(b.numel() * b.element_size() for b in obj.buffers())
                if total > 0:
                    return round(total / 1024**3, 3), f"measured via {'.'.join(path)}"
            except Exception:
                pass
    # Fallback estimate from config.
    cfg = _hf_config(llm)
    est = _estimate_weight_bytes(cfg, mode)
    if est is not None:
        return round(est / 1024**3, 3), "estimated from HF config (bf16/fp16 baseline assumption)"
    return None, "unavailable from vLLM backend"


def kv_stats(llm, wbytes_gb: float | None, total_vram: int | None) -> dict:
    cfg = _hf_config(llm)
    if cfg is None:
        return {"kv_bytes_per_token": None, "headroom": None}
    hidden = getattr(cfg, "hidden_size", None)
    layers = getattr(cfg, "num_hidden_layers", None)
    heads = getattr(cfg, "num_attention_heads", None)
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    head_dim = getattr(cfg, "head_dim", None) or (hidden // heads if hidden and heads else None)
    rate = 2 * layers * kv_heads * head_dim * 2 if layers and kv_heads and head_dim else None
    wbytes = int(wbytes_gb * 1024**3) if wbytes_gb is not None else 0
    usable = max(0, total_vram - wbytes - ACTIVATION_RESERVE) if rate and total_vram else 0
    headroom = (
        {"usable_gb": round(usable / 1024**3, 3), "max_tokens": int(usable // rate)}
        if rate and total_vram
        else None
    )
    return {"kv_bytes_per_token": rate, "headroom": headroom}


def _prompt_logprob_nll(output) -> tuple[float, int] | None:
    # vLLM output.prompt_logprobs: list[Optional[dict[token_id, Logprob]]] aligned to prompt tokens.
    # First element is usually None.
    plp = getattr(output, "prompt_logprobs", None)
    token_ids = getattr(output, "prompt_token_ids", None)
    if not plp or not token_ids or len(plp) != len(token_ids):
        return None
    nll = 0.0
    n = 0
    for tid, lp_dict in zip(token_ids, plp):
        if lp_dict is None:
            continue
        entry = lp_dict.get(tid)
        if entry is None and lp_dict:
            # Sometimes keyed differently; try first value as fallback only if unambiguous.
            try:
                vals = list(lp_dict.values())
                entry = vals[0] if len(vals) == 1 else None
            except Exception:
                entry = None
        if entry is None:
            continue
        lp = getattr(entry, "logprob", entry)
        try:
            nll += -float(lp)
            n += 1
        except Exception:
            continue
    if n == 0:
        return None
    return nll, n


def perplexity_and_accuracy(llm, text: str, max_len: int = 512) -> tuple[float | None, float | None, str | None]:
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    ids = tok.encode(text)
    if not ids:
        return None, None, "empty tokenization"
    ids = ids[:max_len]
    prompt = tok.decode(ids, skip_special_tokens=False)
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    try:
        outs = llm.generate([prompt], sp)
        if not outs:
            return None, None, "no output from vLLM"
        scored = _prompt_logprob_nll(outs[0])
        if scored is None:
            return None, None, "prompt_logprobs unavailable or unusable in this vLLM build"
        nll, n = scored
        ppl = math.exp(nll / max(n, 1))
        # Next-token accuracy proxy not directly available from prompt logprobs; return None.
        return ppl, None, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _sequence_logprob(llm, text: str, max_len: int) -> float | None:
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    ids = tok.encode(text)[:max_len]
    if len(ids) < 2:
        return None
    prompt = tok.decode(ids, skip_special_tokens=False)
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    outs = llm.generate([prompt], sp)
    if not outs:
        return None
    plp = getattr(outs[0], "prompt_logprobs", None)
    tids = getattr(outs[0], "prompt_token_ids", None)
    if not plp or not tids:
        return None
    total = 0.0
    n = 0
    for tid, lp_dict in zip(tids, plp):
        if lp_dict is None:
            continue
        entry = lp_dict.get(tid)
        if entry is None:
            continue
        try:
            total += float(getattr(entry, "logprob", entry))
            n += 1
        except Exception:
            continue
    return total if n else None


def mc_accuracy(llm, examples: list[tuple[str, list[str], int]], max_len: int = 256) -> float | None:
    correct = 0
    tried = 0
    for prompt, choices, gold in examples:
        scores = []
        for choice in choices:
            s = _sequence_logprob(llm, f"{prompt} {choice}", max_len)
            scores.append(float("-inf") if s is None else s)
        if any(math.isfinite(s) for s in scores):
            tried += 1
            correct += int(max(range(len(scores)), key=lambda i: scores[i]) == gold)
    if tried == 0:
        return None
    return correct / tried


def benchmark(llm, prompt: str, max_new_tokens: int, warmup: int = 1) -> dict:
    from vllm import SamplingParams

    sp_warm = SamplingParams(max_tokens=min(2, max_new_tokens), temperature=0)
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    try:
        for _ in range(warmup):
            llm.generate([prompt], sp_warm)
        t0 = time.perf_counter()
        llm.generate([prompt], sp)
        dt = time.perf_counter() - t0
        return {"tok_s": round(max_new_tokens / max(dt, 1e-9), 2), "error": None}
    except Exception as exc:
        return {"tok_s": None, "error": f"{type(exc).__name__}: {exc}"}


def _llm_kwargs(mode: str, args) -> tuple[dict, list[str]]:
    notes: list[str] = []
    kw: dict = {
        "model": args.model,
        "trust_remote_code": True,
        "enforce_eager": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    if args.tensor_parallel_size and args.tensor_parallel_size > 1:
        kw["tensor_parallel_size"] = args.tensor_parallel_size
    if mode in ("bf16", "fp16"):
        kw["dtype"] = "bfloat16" if mode == "bf16" else "float16"
        kw["quantization"] = None
    elif mode == "bnb_8bit":
        kw["quantization"] = "bitsandbytes"
        kw["load_format"] = "bitsandbytes"
        # vLLM BNB config varies; try passing via additional kwargs if supported by this build.
        kw["dtype"] = "auto"
        notes.append("requested vLLM bitsandbytes 8-bit (load_format=bitsandbytes)")
    elif mode == "bnb_4bit":
        kw["quantization"] = "bitsandbytes"
        kw["load_format"] = "bitsandbytes"
        kw["dtype"] = "auto"
        notes.append("requested vLLM bitsandbytes 4-bit (load_format=bitsandbytes); may require prequantized model or version-specific config")
    else:
        raise ValueError(f"unknown mode: {mode}")
    # Remove Nones to avoid API strictness issues.
    kw = {k: v for k, v in kw.items() if v is not None}
    return kw, notes


def run_mode(mode: str, args, eval_text: str) -> dict:
    res = {
        "mode": mode,
        "backend": "vllm",
        "status": "ok",
        "weights_gb": None,
        "weights_source": None,
        "cuda_mem_gb": None,
        "tok_s": None,
        "perplexity": None,
        "next_token_accuracy": None,
        "mc_accuracy": None,
        "kv_bytes_per_token": None,
        "kv_max_tokens": None,
        "dtype_used": None,
        "errors": [],
        "notes": [],
    }
    llm = None
    try:
        import torch
        from vllm import LLM

        kw, notes = _llm_kwargs(mode, args)
        res["notes"].extend(notes)
        log(f"[{mode}] loading {args.model} with vLLM kwargs={kw}")
        llm = LLM(**kw)
        res["dtype_used"] = mode
        wb, wsrc = _weight_bytes_best_effort(llm, mode)
        res["weights_gb"] = wb
        res["weights_source"] = wsrc
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                res["cuda_mem_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
            except Exception:
                pass
        vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        kv = kv_stats(llm, wb, vram)
        res["kv_bytes_per_token"] = kv["kv_bytes_per_token"]
        res["kv_max_tokens"] = (kv["headroom"] or {}).get("max_tokens")
        res["kv_detail"] = kv
        bench = benchmark(llm, SPEED_PROMPT, args.max_tokens)
        res["tok_s"] = bench.get("tok_s")
        if bench.get("error"):
            res["errors"].append(bench["error"])
        ppl, nta, perr = perplexity_and_accuracy(llm, eval_text)
        res["perplexity"] = ppl
        res["next_token_accuracy"] = nta
        if perr:
            res["notes"].append(f"perplexity/next-token accuracy: {perr}")
        try:
            res["mc_accuracy"] = mc_accuracy(llm, MC_EXAMPLES)
        except Exception as exc:
            res["notes"].append(f"mc_accuracy unavailable: {type(exc).__name__}: {exc}")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        res.update(status="error", errors=[msg], traceback=traceback.format_exc())
        low = msg.lower()
        if mode.startswith("bnb") and ("bitsandbytes" in low or "quantization" in low or "unsupported" in low or "not supported" in low):
            res["status"] = "skipped"
            res["notes"].append("bitsandbytes mode not supported by this vLLM build/model combination")
        log(f"[{mode}] ERROR: {msg}")
    finally:
        if llm is not None:
            try:
                # Best-effort cleanup for vLLM engine.
                engine = getattr(llm, "llm_engine", None)
                if engine is not None and hasattr(engine, "engine_core"):
                    pass
                del llm
            except Exception:
                try:
                    del llm
                except Exception:
                    pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return res


def _fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def render_md(results: dict) -> str:
    dev = results.get("device", {})
    nv = results.get("nvfp4", {})
    ck = results.get("checkpoint_file", {})
    lines = [
        "# E3: vLLM backend reality check + quantization",
        "",
        f"- Model: `{results.get('model')}` | Backend: vLLM | Device: {dev.get('name') or 'CPU'} "
        f"(CC {dev.get('cc') or 'n/a'}) | Torch {dev.get('torch')}",
        f"- NVFP4 hardware supported: **{nv.get('supported')}** ({nv.get('reason') or 'n/a'})",
        f"- Eval text: {results.get('eval_text_source')}",
        "",
        "## Checkpoint dtype inspection",
        "",
        f"- Stored dtypes: `{ck.get('dtypes')}` | float4: {ck.get('float4')} | "
        f"uint8/int8: {ck.get('uint8')} | NVFP4 marker: {ck.get('nvfp4')}",
        "",
        "## Benchmarks",
        "",
        "| Mode | Weights GB | CUDA mem GB | tok/s b=1 | PPL | Next-token acc | MC acc | KV B/token | Max KV tokens | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    keys = (
        "weights_gb",
        "cuda_mem_gb",
        "tok_s",
        "perplexity",
        "next_token_accuracy",
        "mc_accuracy",
        "kv_bytes_per_token",
        "kv_max_tokens",
    )
    for mode, r in results.get("modes", {}).items():
        lines.append(
            "| "
            + " | ".join([mode] + [_fmt(r.get(k)) for k in keys] + [r.get("status", "?")])
            + " |"
        )
    lines += ["", "## Notes", ""]
    for mode, r in results.get("modes", {}).items():
        if r.get("weights_source"):
            lines.append(f"- **{mode}** weights source: {r['weights_source']}")
        for n in r.get("notes", []):
            lines.append(f"- **{mode}**: {n}")
        for e in r.get("errors", []):
            lines.append(f"- **{mode}** error: {e}")
    for e in results.get("errors", []):
        lines.append(f"- global error: {e}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E3: vLLM backend reality check + quantization")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    ap.add_argument("--max-tokens", type=int, default=64, help="tokens for the batch-1 speed benchmark")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16", help="full-precision compute dtype to include")
    ap.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="modes to run (default: <dtype> bnb_8bit bnb_4bit)",
    )
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    dev = device_info()
    nv = nvfp4_support()
    ck = checkpoint_dtypes(args.model)
    text, src = load_eval_text()
    log(f"device: {dev.get('name') or 'CPU'} CC={dev.get('cc') or 'n/a'} torch={dev.get('torch')}")
    log(f"NVFP4 supported: {nv.get('supported')} ({nv.get('reason')}) | ckpt dtypes: {ck.get('dtypes')}")
    log(f"eval text: {src} ({len(text)} chars)")

    ensure_vllm()
    modes = args.modes or [args.dtype, "bnb_8bit", "bnb_4bit"]
    results = {
        "model": args.model,
        "backend": "vllm",
        "dtype_requested": args.dtype,
        "max_new_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "device": dev,
        "nvfp4": nv,
        "checkpoint_file": ck,
        "eval_text_source": src,
        "modes_requested": modes,
        "modes": {},
        "errors": [],
    }
    for mode in modes:
        r = run_mode(mode, args, text)
        results["modes"][mode] = r
        log(
            f"[{mode}] status={r['status']} weights={r.get('weights_gb')}GB tok/s={r.get('tok_s')} "
            f"ppl={r.get('perplexity')} mc={r.get('mc_accuracy')}"
        )

    jp = os.path.join(args.out, "e3_results.json")
    mp = os.path.join(args.out, "e3_summary.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(render_md(results))
    log(f"wrote {jp} and {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
