#!/usr/bin/env python3
"""
E8: INT4 keep-rate + cross-model soup (kill fast).

Compares FP16, INT8, and INT4 (NF4 via bitsandbytes) for a causal LM:
- weight memory footprint (GB)
- generation throughput (tok/s)
- perplexity (lower is better)
- accuracy on a small deterministic QA set (higher is better)

Optionally averages weights of two checkpoints (cross-model soup) and evaluates.
Kill-fast: exits early if INT4 accuracy drop exceeds --threshold points.

Usage:
  python 08_quant_soup.py --model <path_or_id> --out results/e8
  python 08_quant_soup.py --model m1 --model2 m2 --out results/e8 --threshold 5
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    try:
        import transformers.configuration_utils as _cu
        if "mlp" not in getattr(_cu, "ALLOWED_ATTN_LAYER_TYPES", ()):
            _cu.ALLOWED_ATTN_LAYER_TYPES = _cu.ALLOWED_ATTN_LAYER_TYPES + ("mlp",)
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    import bitsandbytes as bnb  # noqa: F401

    HAS_BNB = True
except Exception:  # pragma: no cover
    HAS_BNB = False


@dataclass
class EvalResult:
    name: str
    quant: str
    weights_gb: float
    tok_per_s: float
    perplexity: float
    accuracy: float
    notes: str = ""
    error: str = ""


QA_SET: List[Tuple[str, str]] = [
    ("The capital of France is", " Paris"),
    ("2 + 2 =", " 4"),
    ("The color of the clear daytime sky is", " blue"),
    ("Water freezes at 0 degrees", " Celsius"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The opposite of hot is", " cold"),
    ("A cat says", " meow"),
    ("The first month of the year is", " January"),
    ("Humans breathe", " oxygen"),
    ("The chemical symbol for water is", " H2O"),
]

PPL_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can be compressed using quantization techniques.",
    "In a distant galaxy, stars formed from clouds of gas and dust.",
    "Python is a popular language for scientific computing and data analysis.",
    "The experiment measured throughput, memory usage, and model quality.",
]


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def require_deps() -> None:
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        die(
            "Missing required dependencies. Install: pip install torch transformers. "
            "For INT4/INT8 also install: pip install bitsandbytes accelerate"
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def device_str() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def param_bytes(model: Any) -> int:
    total = 0
    for p in model.parameters():
        try:
            total += p.numel() * p.element_size()
        except Exception:
            pass
    for b in model.buffers():
        try:
            total += b.numel() * b.element_size()
        except Exception:
            pass
    return total


def weights_gb(model: Any) -> float:
    # Prefer actual allocated CUDA memory when available; fallback to param bytes.
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            if allocated > 0:
                return allocated / (1024**3)
        except Exception:
            pass
    return param_bytes(model) / (1024**3)


@torch.inference_mode()
def measure_throughput(
    model: Any, tokenizer: Any, prompt: str = "Once upon a time", max_new_tokens: int = 32, runs: int = 2
) -> float:
    if torch is None:
        return 0.0
    dev = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    # warmup
    _ = model.generate(**inputs, max_new_tokens=min(8, max_new_tokens), do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    best = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new_tokens = max(1, out.shape[-1] - inputs["input_ids"].shape[-1])
        best = max(best, new_tokens / dt if dt > 0 else 0.0)
    return float(best)


@torch.inference_mode()
def measure_perplexity(model: Any, tokenizer: Any, texts: List[str]) -> float:
    if torch is None or not texts:
        return float("inf")
    dev = next(model.parameters()).device
    nlls: List[float] = []
    tok_count = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        if input_ids.shape[-1] < 2:
            continue
        out = model(input_ids, labels=input_ids)
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            continue
        n = input_ids.shape[-1]
        nlls.append(float(loss.item()) * n)
        tok_count += n
    if tok_count == 0 or not nlls:
        return float("inf")
    return float(math.exp(sum(nlls) / tok_count))


@torch.inference_mode()
def measure_accuracy(model: Any, tokenizer: Any, qa: List[Tuple[str, str]]) -> float:
    if torch is None or not qa:
        return 0.0
    dev = next(model.parameters()).device
    correct = 0
    for prompt, expected in qa:
        enc = tokenizer(prompt, return_tensors="pt").to(dev)
        # generate a few tokens and check prefix/containment
        gen = model.generate(
            **enc,
            max_new_tokens=max(1, len(tokenizer.encode(expected, add_special_tokens=False)) + 2),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0][enc["input_ids"].shape[-1] :], skip_special_tokens=True)
        norm = text.strip().lower()
        exp = expected.strip().lower()
        if norm.startswith(exp) or exp in norm:
            correct += 1
    return 100.0 * correct / len(qa)


def load_model(model_id: str, quant: str, tokenizer: Any) -> Any:
    if torch is None or AutoModelForCausalLM is None:
        die("transformers/torch unavailable")
    common: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    if quant == "fp16":
        kwargs = dict(common)
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if quant == "int8":
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not installed; INT8 unavailable")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto" if torch.cuda.is_available() else None,
            **common,
        )
    if quant == "int4":
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not installed; INT4 unavailable")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            device_map="auto" if torch.cuda.is_available() else None,
            **common,
        )
    raise ValueError(f"unknown quant: {quant}")


def evaluate(model_id: str, quant: str) -> EvalResult:
    if torch is None or AutoTokenizer is None:
        return EvalResult(model_id, quant, 0.0, 0.0, float("inf"), 0.0, error="deps missing")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_model(model_id, quant, tokenizer)
        model.eval()
        wgb = weights_gb(model)
        tps = measure_throughput(model, tokenizer)
        ppl = measure_perplexity(model, tokenizer, PPL_TEXTS)
        acc = measure_accuracy(model, tokenizer, QA_SET)
        res = EvalResult(
            name=model_id,
            quant=quant,
            weights_gb=round(wgb, 4),
            tok_per_s=round(tps, 3),
            perplexity=round(ppl, 4) if math.isfinite(ppl) else ppl,
            accuracy=round(acc, 2),
            notes=f"device={device_str()}",
        )
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res
    except Exception as exc:
        return EvalResult(model_id, quant, 0.0, 0.0, float("inf"), 0.0, error=str(exc))


def soup_two_models(model_a: str, model_b: str) -> EvalResult:
    """Average FP32/FP16 state dicts of two compatible checkpoints on CPU."""
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        return EvalResult(f"soup({model_a},{model_b})", "soup-fp16", 0.0, 0.0, float("inf"), 0.0, error="deps missing")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        tok = AutoTokenizer.from_pretrained(model_a, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        ma = AutoModelForCausalLM.from_pretrained(model_a, torch_dtype=torch.float16, trust_remote_code=True)
        mb = AutoModelForCausalLM.from_pretrained(model_b, torch_dtype=torch.float16, trust_remote_code=True)
        sda = ma.state_dict()
        sdb = mb.state_dict()
        keys = set(sda.keys()) & set(sdb.keys())
        if not keys:
            raise RuntimeError("no overlapping state_dict keys to average")
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            ta, tb = sda[k], sdb[k]
            if ta.shape != tb.shape:
                out[k] = ta
                continue
            if torch.is_floating_point(ta) and torch.is_floating_point(tb):
                out[k] = ((ta.detach().cpu().float() + tb.detach().cpu().float()) / 2.0).to(ta.dtype)
            else:
                out[k] = ta
        # keep unmatched keys from A
        for k, v in sda.items():
            if k not in out:
                out[k] = v
        ma.load_state_dict(out, strict=False)
        if torch.cuda.is_available():
            ma = ma.to("cuda")
        ma.eval()
        res = EvalResult(
            name=f"soup({Path(model_a).name}+{Path(model_b).name})",
            quant="soup-fp16",
            weights_gb=round(weights_gb(ma), 4),
            tok_per_s=round(measure_throughput(ma, tok), 3),
            perplexity=round(measure_perplexity(ma, tok, PPL_TEXTS), 4),
            accuracy=round(measure_accuracy(ma, tok, QA_SET), 2),
            notes=f"averaged_keys={len(keys)} device={device_str()}",
        )
        del ma, mb, sda, sdb, out, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res
    except Exception as exc:
        return EvalResult(f"soup({model_a},{model_b})", "soup-fp16", 0.0, 0.0, float("inf"), 0.0, error=str(exc))


def to_markdown(results: List[EvalResult], meta: Dict[str, Any]) -> str:
    lines = [
        "# E8: INT4 Keep-Rate + Cross-Model Soup",
        "",
        f"- model: `{meta.get('model')}`",
        f"- model2: `{meta.get('model2')}`",
        f"- threshold: {meta.get('threshold')} accuracy points",
        f"- bitsandbytes: {HAS_BNB}",
        f"- device: {device_str()}",
        "",
        "| name | quant | weights_gb | tok/s | perplexity | accuracy | error | notes |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        ppl = r.perplexity if isinstance(r.perplexity, str) else f"{r.perplexity:.4f}" if math.isfinite(r.perplexity) else "inf"
        lines.append(
            f"| {r.name} | {r.quant} | {r.weights_gb:.4f} | {r.tok_per_s:.3f} | {ppl} | {r.accuracy:.2f} | {r.error.replace('|','/')} | {r.notes.replace('|','/')} |"
        )
    if meta.get("killed"):
        lines += ["", f"**KILL**: {meta['killed']}"]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="E8 INT4 keep-rate + cross-model soup")
    parser.add_argument("--model", required=True, help="Primary model path or HF id")
    parser.add_argument("--model2", default=None, help="Optional second model for weight soup")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=5.0, help="Max INT4 accuracy drop (points) before KILL")
    args = parser.parse_args()

    require_deps()
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    results: List[EvalResult] = []
    meta: Dict[str, Any] = {
        "model": args.model,
        "model2": args.model2,
        "threshold": args.threshold,
        "has_bitsandbytes": HAS_BNB,
        "killed": "",
    }

    print(f"[E8] device={device_str()} model={args.model} bnb={HAS_BNB}")
    for quant in ("fp16", "int8", "int4"):
        print(f"[E8] evaluating {quant}...")
        res = evaluate(args.model, quant)
        results.append(res)
        print(
            f"  -> {quant}: weights_gb={res.weights_gb} tok/s={res.tok_per_s} ppl={res.perplexity} acc={res.accuracy}%"
            + (f" ERROR={res.error}" if res.error else "")
        )
        if quant == "int4":
            fp16 = next((r for r in results if r.quant == "fp16"), None)
            if fp16 and not fp16.error and not res.error:
                drop = fp16.accuracy - res.accuracy
                if drop > args.threshold:
                    msg = f"INT4 accuracy drop {drop:.2f} > {args.threshold:.2f} (fp16={fp16.accuracy:.2f}, int4={res.accuracy:.2f})"
                    print(f"KILL: {msg}")
                    meta["killed"] = msg
                    payload = {"meta": meta, "results": [asdict(r) for r in results]}
                    (out_dir / "e8_quant_soup.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    (out_dir / "e8_quant_soup.md").write_text(to_markdown(results, meta), encoding="utf-8")
                    return 1

    if args.model2:
        print(f"[E8] evaluating cross-model soup: {args.model} + {args.model2}")
        soup = soup_two_models(args.model, args.model2)
        results.append(soup)
        print(
            f"  -> soup: weights_gb={soup.weights_gb} tok/s={soup.tok_per_s} ppl={soup.perplexity} acc={soup.accuracy}%"
            + (f" ERROR={soup.error}" if soup.error else "")
        )
    else:
        meta["soup_note"] = "model2 not provided; cross-model soup skipped (single-checkpoint run)"

    payload = {"meta": meta, "results": [asdict(r) for r in results]}
    json_path = out_dir / "e8_quant_soup.json"
    md_path = out_dir / "e8_quant_soup.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(results, meta), encoding="utf-8")
    print(f"[E8] wrote {json_path}")
    print(f"[E8] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
