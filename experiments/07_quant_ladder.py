#!/usr/bin/env python3
"""E7: T4 quantization ladder for Dory with vLLM.

Stages:
1) TinyLlama 1.1B speed gate across quant modes (records if quant is slower than 2x).
2) Dory 4B across quant modes, loops 1 and 4, hard subset prompts, and NIAH 64K probe.

Quant modes attempted: fp16, bnb_8bit, bnb_4bit (errors recorded, not fatal).
NVFP4 is not supported on SM7.5 (T4); this script records that fact.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DORY_MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
DEFAULT_MODES = ["fp16", "bnb_8bit", "bnb_4bit"]
VLLM_REF = "git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head"


def ensure_vllm() -> None:
    """Import vLLM; install the pinned PR ref if missing."""
    try:
        import vllm  # noqa: F401

        return
    except Exception:
        os.environ["VLLM_USE_PRECOMPILED"] = "1"
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                VLLM_REF,
            ]
        )


@dataclass
class RunResult:
    stage: str
    model: str
    mode: str
    loops: Optional[int] = None
    status: str = "ok"
    weights_gb: Optional[float] = None
    tok_per_s: Optional[float] = None
    accuracy: Optional[float] = None
    rep_rate: Optional[float] = None
    max_context: Optional[int] = None
    n_prompts: int = 0
    nvfp4_supported: bool = False
    slower_than_2x: Optional[bool] = None
    speed_ratio_vs_fp16: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


HARD_SUBSET_50: List[Dict[str, str]] = [
    {"task": "math", "prompt": "Question: If 3x - 7 = 2x + 11, what is x?\nAnswer:", "gold": "18"},
    {"task": "math", "prompt": "Question: What is 17 * 23?\nAnswer:", "gold": "391"},
    {"task": "math", "prompt": "Question: A train travels 60 km in 45 minutes. What is its speed in km/h?\nAnswer:", "gold": "80"},
    {"task": "math", "prompt": "Question: What is the remainder when 100 is divided by 7?\nAnswer:", "gold": "2"},
    {"task": "math", "prompt": "Question: Simplify: (12/18) * (9/4).\nAnswer:", "gold": "1.5"},
    {"task": "math", "prompt": "Question: If a rectangle has perimeter 30 and length 8, what is its width?\nAnswer:", "gold": "7"},
    {"task": "math", "prompt": "Question: What is 2^10?\nAnswer:", "gold": "1024"},
    {"task": "math", "prompt": "Question: Solve for y: 5y + 3 = 2y + 18.\nAnswer:", "gold": "5"},
    {"task": "math", "prompt": "Question: What is 15% of 240?\nAnswer:", "gold": "36"},
    {"task": "math", "prompt": "Question: The average of 4, 8, 10, and x is 9. What is x?\nAnswer:", "gold": "14"},
    {"task": "reasoning", "prompt": "Question: All bloops are razzies. Some razzies are lazzies. Can we conclude some bloops are lazzies?\nAnswer with Yes or No:", "gold": "no"},
    {"task": "reasoning", "prompt": "Question: If it rains, the ground gets wet. The ground is wet. Did it necessarily rain?\nAnswer with Yes or No:", "gold": "no"},
    {"task": "reasoning", "prompt": "Question: Tom is taller than Sam. Sam is taller than Alex. Who is shortest?\nAnswer:", "gold": "alex"},
    {"task": "reasoning", "prompt": "Question: A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball in cents?\nAnswer:", "gold": "5"},
    {"task": "reasoning", "prompt": "Question: If today is Wednesday, what day was it 100 days ago? (7-day week)\nAnswer:", "gold": "monday"},
    {"task": "reasoning", "prompt": "Question: Which is heavier: a kilogram of steel or a kilogram of feathers?\nAnswer:", "gold": "same"},
    {"task": "reasoning", "prompt": "Question: You have 3 apples and take away 2. How many apples do you have?\nAnswer:", "gold": "2"},
    {"task": "reasoning", "prompt": "Question: A farmer has 17 sheep; all but 9 die. How many are left?\nAnswer:", "gold": "9"},
    {"task": "reasoning", "prompt": "Question: If some cats are black and all black things absorb heat, do some cats absorb heat?\nAnswer with Yes or No:", "gold": "yes"},
    {"task": "reasoning", "prompt": "Question: What comes next: 2, 6, 12, 20, 30, ?\nAnswer:", "gold": "42"},
    {"task": "trivia", "prompt": "Question: What is the chemical symbol for gold?\nAnswer:", "gold": "au"},
    {"task": "trivia", "prompt": "Question: Who wrote '1984'?\nAnswer:", "gold": "orwell"},
    {"task": "trivia", "prompt": "Question: What planet is known as the Red Planet?\nAnswer:", "gold": "mars"},
    {"task": "trivia", "prompt": "Question: In what year did World War II end?\nAnswer:", "gold": "1945"},
    {"task": "trivia", "prompt": "Question: What is the capital of Canada?\nAnswer:", "gold": "ottawa"},
    {"task": "trivia", "prompt": "Question: What gas do plants absorb from the atmosphere?\nAnswer:", "gold": "carbon dioxide"},
    {"task": "trivia", "prompt": "Question: Who painted the Mona Lisa?\nAnswer:", "gold": "da vinci"},
    {"task": "trivia", "prompt": "Question: What is the largest ocean on Earth?\nAnswer:", "gold": "pacific"},
    {"task": "trivia", "prompt": "Question: What is the square root of 144?\nAnswer:", "gold": "12"},
    {"task": "trivia", "prompt": "Question: Which element has atomic number 1?\nAnswer:", "gold": "hydrogen"},
    {"task": "code", "prompt": "Complete the Python function to return the maximum of a list:\ndef max_val(xs):\n    ", "gold": "max"},
    {"task": "code", "prompt": "Complete the Python function to reverse a string:\ndef rev(s):\n    return ", "gold": "[::-1]"},
    {"task": "code", "prompt": "What does `len({'a':1,'b':2})` return in Python?\nAnswer:", "gold": "2"},
    {"task": "code", "prompt": "Complete: return sum of a list in Python.\ndef s(x):\n    return ", "gold": "sum"},
    {"task": "code", "prompt": "In Python, what keyword defines a function?\nAnswer:", "gold": "def"},
    {"task": "code", "prompt": "What is the output type of `range(5)` in Python 3?\nAnswer:", "gold": "range"},
    {"task": "code", "prompt": "Complete to check if key 'k' is in dict 'd':\nif ", "gold": "in d"},
    {"task": "code", "prompt": "What method adds an element to the end of a Python list?\nAnswer:", "gold": "append"},
    {"task": "code", "prompt": "In Python, what does `bool([])` evaluate to?\nAnswer:", "gold": "false"},
    {"task": "code", "prompt": "Complete: list comprehension squaring numbers 0..4 in Python:\n[", "gold": "for"},
    {"task": "science", "prompt": "Question: What force keeps planets in orbit around the Sun?\nAnswer:", "gold": "gravity"},
    {"task": "science", "prompt": "Question: What is H2O commonly known as?\nAnswer:", "gold": "water"},
    {"task": "science", "prompt": "Question: How many chromosomes do humans typically have?\nAnswer:", "gold": "46"},
    {"task": "science", "prompt": "Question: What part of the cell contains genetic material?\nAnswer:", "gold": "nucleus"},
    {"task": "science", "prompt": "Question: At what temperature Celsius does water boil at sea level?\nAnswer:", "gold": "100"},
    {"task": "science", "prompt": "Question: What is the speed of light approximately in km/s?\nAnswer:", "gold": "300000"},
    {"task": "science", "prompt": "Question: What organ pumps blood through the body?\nAnswer:", "gold": "heart"},
    {"task": "science", "prompt": "Question: What is the powerhouse of the cell?\nAnswer:", "gold": "mitochond"},
    {"task": "science", "prompt": "Question: Sound travels fastest through solid, liquid, or gas?\nAnswer:", "gold": "solid"},
    {"task": "science", "prompt": "Question: What gas makes up most of Earth's atmosphere?\nAnswer:", "gold": "nitrogen"},
]


def log(msg: str) -> None:
    print(f"[e7] {msg}", flush=True)


def nvfp4_supported() -> bool:
    """NVFP4 is unsupported on SM7.5 (T4). Probe if torch is available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        cc = torch.cuda.get_device_capability(0)
        if tuple(cc) < (10, 0):
            return False
        probe = getattr(torch._C, "_get_nvfp4_device_support", None)
        if probe is None:
            return False
        return bool(probe())
    except Exception:
        return False


def device_note() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        return f"{name} cc={list(cc)}"
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def _cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception:
        pass


def rep_rate(text: str) -> float:
    toks = text.split()
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def weights_gb(llm: Any) -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            if allocated > 0:
                return round(allocated / (1024**3), 4)
    except Exception:
        pass
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
                    return round(total / (1024**3), 4)
            except Exception:
                pass
    return None


def build_llm_kwargs(
    model: str,
    mode: str,
    loops: Optional[int],
    max_model_len: int,
    gpu_memory_utilization: float,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    if loops is not None:
        kwargs["hf_overrides"] = {
            "n_recurrent_loops": int(loops),
            "recurrent_kv_cache_mode": "last_loop",
        }
    if mode == "fp16":
        kwargs["dtype"] = "float16"
        kwargs["quantization"] = None
    elif mode == "bf16":
        kwargs["dtype"] = "bfloat16"
        kwargs["quantization"] = None
    elif mode == "bnb_8bit":
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"
        kwargs["dtype"] = "auto"
    elif mode == "bnb_4bit":
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"
        kwargs["dtype"] = "auto"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return {k: v for k, v in kwargs.items() if v is not None}


def load_llm(model: str, mode: str, loops: Optional[int], max_model_len: int, gpu_mem: float) -> Any:
    ensure_vllm()
    from vllm import LLM

    kwargs = build_llm_kwargs(model, mode, loops, max_model_len, gpu_mem)
    return LLM(**kwargs)


def generate_texts(llm: Any, prompts: List[str], max_tokens: int) -> Tuple[List[str], List[int], float]:
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    dt = time.perf_counter() - t0
    texts: List[str] = []
    ntoks: List[int] = []
    for out in outputs:
        if out and out.outputs:
            txt = out.outputs[0].text or ""
            texts.append(txt)
            try:
                ntoks.append(len(out.outputs[0].token_ids))
            except Exception:
                ntoks.append(max(1, len(txt.split())))
        else:
            texts.append("")
            ntoks.append(0)
    return texts, ntoks, dt


def measure_tok_per_s(llm: Any, prompt: str, max_tokens: int, runs: int = 2) -> float:
    from vllm import SamplingParams

    sp_w = SamplingParams(max_tokens=min(8, max_tokens), temperature=0.0, top_p=1.0)
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    try:
        llm.generate([prompt], sp_w)
    except Exception:
        pass
    _cleanup()
    best = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        dt = max(time.perf_counter() - t0, 1e-9)
        n = 0
        if out and out[0].outputs:
            try:
                n = len(out[0].outputs[0].token_ids)
            except Exception:
                n = max(1, len((out[0].outputs[0].text or "").split()))
        best = max(best, n / dt)
    return round(best, 3)


def eval_accuracy_and_rep(llm: Any, cases: List[Dict[str, str]], max_tokens: int) -> Tuple[float, float, int]:
    prompts = [c["prompt"] for c in cases]
    texts, _, _ = generate_texts(llm, prompts, max_tokens)
    correct = 0
    reps: List[float] = []
    for case, text in zip(cases, texts):
        gold = case["gold"].lower()
        ok = gold in (text or "").lower()
        correct += 1 if ok else 0
        reps.append(rep_rate(text or ""))
    acc = 100.0 * correct / max(1, len(cases))
    avg_rep = sum(reps) / max(1, len(reps))
    return round(acc, 2), round(avg_rep, 4), len(cases)


def make_niah_prompt(tokenizer: Any, target_tokens: int = 64000) -> Tuple[str, int]:
    needle = "The secret passcode is DORY-LADDER-7749."
    question = "\nQuestion: What is the secret passcode? Answer with only the passcode.\nAnswer:"
    filler_sentence = "Dory is a recurrent language model evaluated on long-context retrieval. "
    # Build by repeating filler; tokenizer length is approximate until final encode.
    filler = filler_sentence * max(1, (target_tokens // 12))
    prompt = filler + "\n" + needle + "\n" + filler + question
    try:
        ids = tokenizer.encode(prompt)
        n = len(ids)
        if n > target_tokens:
            # Keep tail with question and needle context by truncating middle-left.
            keep_head = ids[: max(0, target_tokens // 2)]
            keep_tail = ids[-(target_tokens - len(keep_head)) :]
            ids = keep_head + keep_tail
            prompt = tokenizer.decode(ids, skip_special_tokens=False)
            n = len(ids)
        return prompt, n
    except Exception:
        # Fallback: character approximation (~4 chars/token).
        approx_chars = target_tokens * 4
        prompt = prompt[:approx_chars]
        return prompt, target_tokens


def run_niah_64k(llm: Any, target_tokens: int = 64000) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "target_tokens": target_tokens,
        "actual_prompt_tokens": None,
        "max_context": None,
        "ok": False,
        "output": "",
        "error": "",
    }
    try:
        tokenizer = llm.get_tokenizer()
        prompt, n = make_niah_prompt(tokenizer, target_tokens)
        result["actual_prompt_tokens"] = n
        texts, _, _ = generate_texts(llm, [prompt], max_tokens=32)
        out = texts[0] if texts else ""
        result["output"] = (out or "")[:300]
        result["ok"] = "DORY-LADDER-7749" in (out or "")
        result["max_context"] = n
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        # Keep a short tail for OOM/context errors.
        result["traceback_tail"] = "\n".join(tb.splitlines()[-8:])
    return result


def run_speed_gate(modes: Sequence[str], args: argparse.Namespace) -> Tuple[List[RunResult], Dict[str, float]]:
    results: List[RunResult] = []
    fp16_tps: Dict[str, float] = {}
    model = args.tiny_model
    for mode in modes:
        r = RunResult(stage="tinyllama_speed_gate", model=model, mode=mode, nvfp4_supported=nvfp4_supported())
        llm = None
        try:
            log(f"speed gate: model={model} mode={mode}")
            llm = load_llm(model, mode, loops=None, max_model_len=args.tiny_max_model_len, gpu_mem=args.gpu_memory_utilization)
            r.weights_gb = weights_gb(llm)
            r.tok_per_s = measure_tok_per_s(llm, "The quick brown fox jumps over the lazy dog.", args.speed_tokens)
            r.max_context = args.tiny_max_model_len
            r.n_prompts = 1
            if mode == "fp16" and r.tok_per_s is not None:
                fp16_tps[model] = r.tok_per_s
            base = fp16_tps.get(model)
            if base and r.tok_per_s is not None and mode != "fp16":
                ratio = base / max(r.tok_per_s, 1e-9)
                r.speed_ratio_vs_fp16 = round(ratio, 4)
                r.slower_than_2x = ratio > 2.0
                r.notes.append(f"fp16_tps={base} ratio={ratio:.3f} (>2x means quant slower than 2x)")
        except Exception as exc:
            r.status = "error"
            r.errors.append(f"{type(exc).__name__}: {exc}")
            r.details["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-10:])
            log(f"speed gate ERROR {model}/{mode}: {r.errors[-1]}")
        finally:
            if llm is not None:
                try:
                    del llm
                except Exception:
                    pass
            _cleanup()
        results.append(r)
    # Fill comparison for non-fp16 if fp16 came later (unlikely, but safe).
    base = fp16_tps.get(model)
    if base:
        for r in results:
            if r.mode != "fp16" and r.tok_per_s is not None and r.slower_than_2x is None:
                ratio = base / max(r.tok_per_s, 1e-9)
                r.speed_ratio_vs_fp16 = round(ratio, 4)
                r.slower_than_2x = ratio > 2.0
    return results, fp16_tps


def run_dory_ladder(modes: Sequence[str], loops_list: Sequence[int], cases: List[Dict[str, str]], args: argparse.Namespace) -> List[RunResult]:
    results: List[RunResult] = []
    model = args.dory_model
    for mode in modes:
        for loops in loops_list:
            r = RunResult(stage="dory_ladder", model=model, mode=mode, loops=int(loops), nvfp4_supported=nvfp4_supported())
            llm = None
            try:
                log(f"dory: mode={mode} loops={loops} prompts={len(cases)}")
                llm = load_llm(model, mode, loops=loops, max_model_len=args.dory_max_model_len, gpu_mem=args.gpu_memory_utilization)
                r.weights_gb = weights_gb(llm)
                r.tok_per_s = measure_tok_per_s(llm, cases[0]["prompt"], args.max_tokens)
                acc, avg_rep, n = eval_accuracy_and_rep(llm, cases, args.max_tokens)
                r.accuracy = acc
                r.rep_rate = avg_rep
                r.n_prompts = n
                # NIAH 64K probe: attempt only when max_model_len allows or as an explicit probe.
                niah = run_niah_64k(llm, target_tokens=64000)
                r.details["niah_64k"] = niah
                if niah.get("max_context") is not None:
                    r.max_context = int(niah["max_context"])
                elif niah.get("error"):
                    r.max_context = args.dory_max_model_len
                    r.notes.append(f"NIAH 64K probe failed; max_context recorded as dory_max_model_len={args.dory_max_model_len}")
                    r.errors.append(f"NIAH_64K: {niah['error']}")
                if niah.get("error") and "out of memory" in niah["error"].lower():
                    r.notes.append("NIAH 64K OOM observed")
            except Exception as exc:
                r.status = "error"
                r.errors.append(f"{type(exc).__name__}: {exc}")
                r.details["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-12:])
                log(f"dory ERROR {model}/{mode}/loops={loops}: {r.errors[-1]}")
            finally:
                if llm is not None:
                    try:
                        del llm
                    except Exception:
                        pass
                _cleanup()
            results.append(r)
    return results


def render_markdown(results: List[RunResult], meta: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# E7: Quantization Ladder (vLLM)")
    lines.append("")
    lines.append(f"- Device: `{meta.get('device')}`")
    lines.append(f"- NVFP4 supported (SM7.5/T4 expected false): `{meta.get('nvfp4_supported')}`")
    lines.append(f"- Modes requested: `{meta.get('modes')}`")
    lines.append(f"- TinyLlama: `{meta.get('tiny_model')}`")
    lines.append(f"- Dory: `{meta.get('dory_model')}`")
    lines.append(f"- Hard subset limit: `{meta.get('limit')}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| stage | model | mode | loops | weights_gb | tok/s | accuracy | rep_rate | max_context | status | notes/errors |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in results:
        note = "; ".join([*r.notes, *r.errors]).replace("|", "/").replace("\n", " ")[:220]
        lines.append(
            "| {stage} | {model} | {mode} | {loops} | {w} | {tps} | {acc} | {rep} | {mc} | {status} | {note} |".format(
                stage=r.stage,
                model=Path(r.model).name,
                mode=r.mode,
                loops="" if r.loops is None else r.loops,
                w="" if r.weights_gb is None else f"{r.weights_gb:.4f}",
                tps="" if r.tok_per_s is None else f"{r.tok_per_s:.3f}",
                acc="" if r.accuracy is None else f"{r.accuracy:.2f}",
                rep="" if r.rep_rate is None else f"{r.rep_rate:.4f}",
                mc="" if r.max_context is None else r.max_context,
                status=r.status,
                note=note,
            )
        )
    lines.append("")
    lines.append("## TinyLlama Speed Gate")
    lines.append("")
    for r in results:
        if r.stage != "tinyllama_speed_gate":
            continue
        extra = ""
        if r.slower_than_2x is not None:
            extra = f" slower_than_2x={r.slower_than_2x} ratio_vs_fp16={r.speed_ratio_vs_fp16}"
        lines.append(f"- `{r.mode}`: tok/s={r.tok_per_s} weights_gb={r.weights_gb} status={r.status}{extra}")
        for e in r.errors:
            lines.append(f"  - error: {e}")
    lines.append("")
    lines.append("## NIAH 64K Probe (Dory)")
    lines.append("")
    for r in results:
        if r.stage != "dory_ladder":
            continue
        niah = r.details.get("niah_64k") or {}
        lines.append(
            f"- mode={r.mode} loops={r.loops}: ok={niah.get('ok')} actual_tokens={niah.get('actual_prompt_tokens')} "
            f"error={str(niah.get('error',''))[:120]}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E7 quantization ladder (TinyLlama gate + Dory ladder)")
    ap.add_argument("--out", default="results/e7_quant_ladder", help="Output directory")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="Quant modes: fp16 bnb_8bit bnb_4bit ...")
    ap.add_argument("--limit", type=int, default=50, help="Hard subset prompt limit (max 50 built-in)")
    ap.add_argument("--tiny-model", default=TINYLLAMA_MODEL, help="TinyLlama model for speed gate")
    ap.add_argument("--dory-model", default=DORY_MODEL, help="Dory model for quant ladder")
    ap.add_argument("--max-tokens", type=int, default=96, help="Max new tokens for eval/speed")
    ap.add_argument("--speed-tokens", type=int, default=64, help="Max new tokens for speed measurement")
    ap.add_argument("--tiny-max-model-len", type=int, default=2048, help="Max model len for TinyLlama gate")
    ap.add_argument("--dory-max-model-len", type=int, default=4096, help="Max model len for Dory ladder")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    args = ap.parse_args(argv)
    if args.limit < 1:
        ap.error("--limit must be >= 1")
    if not args.modes:
        ap.error("--modes must not be empty")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = list(args.modes)
    cases = HARD_SUBSET_50[: min(args.limit, len(HARD_SUBSET_50))]
    meta: Dict[str, Any] = {
        "experiment": "E7_quant_ladder",
        "backend": "vllm",
        "device": device_note(),
        "nvfp4_supported": nvfp4_supported(),
        "modes": modes,
        "tiny_model": args.tiny_model,
        "dory_model": args.dory_model,
        "limit": len(cases),
        "vllm_ref": VLLM_REF,
        "vllm_use_precompiled": os.environ.get("VLLM_USE_PRECOMPILED", "1"),
    }
    if not meta["nvfp4_supported"]:
        meta["nvfp4_note"] = "NVFP4 unsupported on this device (SM7.5/T4 expected)."
    log(f"meta: {json.dumps(meta, ensure_ascii=False)}")

    all_results: List[RunResult] = []
    speed_results, _ = run_speed_gate(modes, args)
    all_results.extend(speed_results)

    dory_results = run_dory_ladder(modes, [1, 4], cases, args)
    all_results.extend(dory_results)

    payload = {"meta": meta, "results": [asdict(r) for r in all_results]}
    json_path = out_dir / "e7_quant_ladder.json"
    md_path = out_dir / "e7_quant_ladder.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(all_results, meta), encoding="utf-8")
    log(f"wrote {json_path}")
    log(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
