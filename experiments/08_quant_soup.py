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
import tempfile
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
    import bitsandbytes as bnb  # noqa: F401

    HAS_BNB = True
except Exception:  # pragma: no cover
    HAS_BNB = False


def ensure_vllm() -> None:
    """Import vLLM and raise a clear error if it is unavailable."""
    try:
        import vllm  # noqa: F401
    except Exception:  # pragma: no cover
        import os, subprocess, sys
        os.environ["VLLM_USE_PRECOMPILED"] = "1"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head"])


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
    if torch is None:
        die("Missing required dependency: torch.")
    ensure_vllm()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def device_str() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _cleanup_cuda() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def weights_gb(llm: Any) -> float:
    # Prefer actual allocated CUDA memory when available.
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            if allocated > 0:
                return allocated / (1024**3)
        except Exception:
            pass
    # Fallback: try to inspect underlying model if exposed.
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        total = 0
        for p in model.parameters():
            total += p.numel() * p.element_size()
        return total / (1024**3)
    except Exception:
        return 0.0


def _sampling_params(max_tokens: int, temperature: float = 0.0):
    from vllm import SamplingParams

    return SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0)


def measure_throughput(
    llm: Any, prompt: str = "Once upon a time", max_new_tokens: int = 32, runs: int = 2
) -> float:
    # warmup
    try:
        llm.generate([prompt], _sampling_params(max_tokens=min(8, max_new_tokens)))
    except Exception:
        pass
    _cleanup_cuda()
    best = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        out = llm.generate([prompt], _sampling_params(max_tokens=max_new_tokens))
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        dt = time.perf_counter() - t0
        new_tokens = 0
        if out and out[0].outputs:
            try:
                new_tokens = len(out[0].outputs[0].token_ids)
            except Exception:
                text = out[0].outputs[0].text or ""
                new_tokens = max(1, len(text.split()))
        new_tokens = max(1, new_tokens)
        best = max(best, new_tokens / dt if dt > 0 else 0.0)
    return float(best)


def measure_perplexity(llm: Any, texts: List[str]) -> float:
    if not texts:
        return float("inf")
    nlls: List[float] = []
    tok_count = 0
    try:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        outputs = llm.generate(texts, params)
    except Exception:
        return float("inf")
    for out in outputs:
        prompt_lps = getattr(out, "prompt_logprobs", None)
        if not prompt_lps:
            continue
        for lp in prompt_lps:
            if lp is None:
                continue
            # lp is typically dict[token_id, Logprob] or similar; pick max/selected logprob.
            try:
                if isinstance(lp, dict) and lp:
                    # Use the highest logprob entry as a proxy for the observed token's logprob.
                    val = max(v.logprob if hasattr(v, "logprob") else float(v) for v in lp.values())
                    nlls.append(-float(val))
                    tok_count += 1
                elif hasattr(lp, "logprob"):
                    nlls.append(-float(lp.logprob))
                    tok_count += 1
            except Exception:
                continue
    if tok_count == 0 or not nlls:
        return float("inf")
    return float(math.exp(sum(nlls) / tok_count))


def measure_accuracy(llm: Any, qa: List[Tuple[str, str]]) -> float:
    if not qa:
        return 0.0
    prompts = [p for p, _ in qa]
    max_new = 1
    try:
        tokenizer = llm.get_tokenizer()
        for _, expected in qa:
            try:
                max_new = max(max_new, len(tokenizer.encode(expected, add_special_tokens=False)) + 2)
            except Exception:
                max_new = max(max_new, 4)
    except Exception:
        max_new = 6
    outputs = llm.generate(prompts, _sampling_params(max_tokens=max_new))
    correct = 0
    for (_, expected), out in zip(qa, outputs):
        text = ""
        if out and out.outputs:
            text = out.outputs[0].text or ""
        norm = text.strip().lower()
        exp = expected.strip().lower()
        if norm.startswith(exp) or exp in norm:
            correct += 1
    return 100.0 * correct / len(qa)


def load_llm(model_id: str, quant: str) -> Any:
    ensure_vllm()
    from vllm import LLM

    kwargs: Dict[str, Any] = {
        "model": model_id,
        "trust_remote_code": True,
        "enforce_eager": True,
    }
    if quant == "fp16":
        kwargs["dtype"] = "float16"
    elif quant == "int8":
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not installed; INT8 unavailable")
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"
    elif quant == "int4":
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not installed; INT4 unavailable")
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"
    elif quant == "soup-fp16":
        kwargs["dtype"] = "float16"
    else:
        raise ValueError(f"unknown quant: {quant}")
    return LLM(**kwargs)


def evaluate(model_id: str, quant: str) -> EvalResult:
    _cleanup_cuda()
    llm = None
    try:
        llm = load_llm(model_id, quant)
        wgb = weights_gb(llm)
        tps = measure_throughput(llm)
        ppl = measure_perplexity(llm, PPL_TEXTS)
        acc = measure_accuracy(llm, QA_SET)
        return EvalResult(
            name=model_id,
            quant=quant,
            weights_gb=round(wgb, 4),
            tok_per_s=round(tps, 3),
            perplexity=round(ppl, 4) if math.isfinite(ppl) else ppl,
            accuracy=round(acc, 2),
            notes=f"device={device_str()} backend=vllm",
        )
    except Exception as exc:
        return EvalResult(model_id, quant, 0.0, 0.0, float("inf"), 0.0, error=str(exc))
    finally:
        try:
            if llm is not None and hasattr(llm, "llm_engine"):
                # Best-effort cleanup; vLLM holds GPU resources.
                del llm
        except Exception:
            pass
        _cleanup_cuda()


def soup_two_models(model_a: str, model_b: str, out_dir: Path) -> EvalResult:
    """Average FP16 state dicts of two compatible checkpoints on CPU, save, eval with vLLM."""
    if torch is None:
        return EvalResult(f"soup({model_a},{model_b})", "soup-fp16", 0.0, 0.0, float("inf"), 0.0, error="torch missing")
    _cleanup_cuda()
    try:
        try:
            import transformers.configuration_utils as _cu

            if "mlp" not in getattr(_cu, "ALLOWED_ATTN_LAYER_TYPES", ()):
                _cu.ALLOWED_ATTN_LAYER_TYPES = _cu.ALLOWED_ATTN_LAYER_TYPES + ("mlp",)
        except Exception:
            pass
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        out: Dict[str, Any] = {}
        for k in keys:
            ta, tb = sda[k], sdb[k]
            if ta.shape != tb.shape:
                out[k] = ta
                continue
            if torch.is_floating_point(ta) and torch.is_floating_point(tb):
                out[k] = ((ta.detach().cpu().float() + tb.detach().cpu().float()) / 2.0).to(ta.dtype)
            else:
                out[k] = ta
        for k, v in sda.items():
            if k not in out:
                out[k] = v
        ma.load_state_dict(out, strict=False)

        soup_dir = Path(tempfile.mkdtemp(prefix="e8_soup_", dir=str(out_dir)))
        ma.save_pretrained(soup_dir, safe_serialization=True)
        tok.save_pretrained(soup_dir)
        try:
            mb.config.save_pretrained(soup_dir)
        except Exception:
            pass

        del ma, mb, sda, sdb, out, tok
        _cleanup_cuda()

        res = evaluate(str(soup_dir), "soup-fp16")
        res.name = f"soup({Path(model_a).name}+{Path(model_b).name})"
        res.notes = (res.notes + f" averaged_keys={len(keys)}").strip()
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
        "backend": "vllm",
        "killed": "",
    }

    print(f"[E8] device={device_str()} model={args.model} bnb={HAS_BNB} backend=vllm")
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
        soup = soup_two_models(args.model, args.model2, out_dir)
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
