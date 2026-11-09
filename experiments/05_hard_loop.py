#!/usr/bin/env python3
"""E5 hard-loop benchmark: recurrent loop scaling on hard tasks.

Datasets: GSM8K (test), MATH-500, BBH (if available). Falls back to
60 built-in hard prompts when datasets cannot be loaded.
Prompts are plain text, no chat template: "Question: {q}\nAnswer:"
Each loop count reloads vLLM with hf_overrides for recurrent loops.
"""

import argparse
import collections
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def ensure_vllm() -> None:
    """Make sure vLLM is importable, preferring precompiled wheels."""
    os.environ.setdefault("VLLM_USE_PRECOMPILED", "1")
    try:
        import vllm  # noqa: F401
        return
    except ImportError:
        pass
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "vllm"],
            env={**os.environ, "VLLM_USE_PRECOMPILED": "1"},
        )
    except Exception as exc:  # pragma: no cover - runtime env only
        raise RuntimeError(f"vLLM is not available and install failed: {exc}") from exc
    import vllm  # noqa: F401


def build_fallback_prompts():
    """Deterministic 60 hard prompts: 20 GSM8K-style, 20 MATH, 20 BBH."""
    items = []

    # --- 20 GSM8K-style word problems (multi-step) ---
    for i in range(20):
        a = 24 + 3 * i
        b = 5 + (i % 4)
        c = 18 + 2 * i
        d = 3 + (i % 3)
        e = 7 + i
        # Problem: start a, give away b*x, buy c, give d*e away
        give1 = b * ((i % 3) + 1)
        give2 = d * e
        ans = a - give1 + c - give2
        if ans <= 0:
            ans = abs(ans) + 5
        q = (
            f"Janet has {a} marbles. She gives {b} of them to each of "
            f"{(i % 3) + 1} friends, then buys {c} more packs with "
            f"{d} in each pack, then gives {e} marbles to each of {d} cousins. "
            f"How many marbles does she have left?"
        )
        # Recompute with the actual phrasing above
        bought = c * d
        give1_actual = b * ((i % 3) + 1)
        give2_actual = e * d
        ans = a - give1_actual + bought - give2_actual
        items.append({"dataset": "gsm8k", "question": q, "target": str(ans)})

    # --- 20 MATH-style problems (answer is boxed content) ---
    for i in range(20):
        n = i + 2
        # Quadratic root sum / product style with computed answer
        r1 = n + 1
        r2 = n + 4
        # Question: find sum of squares of roots of x^2 - s x + p = 0
        s = r1 + r2
        p = r1 * r2
        ans = s * s - 2 * p
        q = (
            f"If r and s are the roots of x^2 - {s}x + {p} = 0, "
            f"find the value of r^2 + s^2."
        )
        items.append({"dataset": "math", "question": q, "target": str(ans)})

    for i in range(20):
        pass  # placeholder to keep counts explicit above

    # Actually build 20 MATH above already; add BBH 20
    # --- 20 BBH-style reasoning tasks ---
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
    for i in range(20):
        n1 = names[i % len(names)]
        n2 = names[(i + 1) % len(names)]
        n3 = names[(i + 2) % len(names)]
        # Tracking task: who has the ball after swaps (deterministic)
        holder = n1
        swaps = [(n1, n2), (n2, n3), (n3, n1)]
        cur = n1
        for x, y in swaps:
            if cur == x:
                cur = y
            elif cur == y:
                cur = x
        q = (
            f"{n1} has the ball. Then {n1} and {n2} swap places, "
            f"then {n2} and {n3} swap, then {n3} and {n1} swap. "
            f"Who has the ball now? Answer with exactly one name."
        )
        items.append({"dataset": "bbh", "question": q, "target": cur})

    # Ensure exactly 60 (20 gsm8k + 20 math + 20 bbh)
    assert len(items) == 60, len(items)
    return items


# ---------------------------------------------------------------------------
# Dataset loading (internet) with fallback
# ---------------------------------------------------------------------------

def _first(row, keys):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return str(row[k])
    return ""


def load_gsm8k(limit: int):
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = []
        for row in ds:
            if len(out) >= limit:
                break
            q = _first(row, ["question"])
            raw = _first(row, ["answer", "solution"])
            m = re.search(r"####\s*(-?[\d\.,]+)", raw)
            target = m.group(1).replace(",", "") if m else _first(row, ["target", "answer"])
            out.append({"dataset": "gsm8k", "question": q, "target": target})
        return out
    except Exception as exc:
        print(f"[load_gsm8k] fallback (reason: {exc})", file=sys.stderr)
        return []


def load_math(limit: int):
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for row in ds:
            if len(out) >= limit:
                break
            q = _first(row, ["problem", "question", "prompt"])
            target = _first(row, ["answer", "target"])
            if not target:
                sol = _first(row, ["solution", "final_answer"])
                m = re.search(r"\\boxed\{([^}]+)\}", sol)
                target = m.group(1) if m else sol[-20:]
            out.append({"dataset": "math", "question": q, "target": target})
        return out
    except Exception as exc:
        print(f"[load_math] fallback (reason: {exc})", file=sys.stderr)
        return []


def load_bbh(limit: int):
    try:
        from datasets import load_dataset

        # lukaemon/bbh is the common HF mirror of BIG-Bench-Hard.
        ds = load_dataset("lukaemon/bbh", split="test")
        out = []
        for row in ds:
            if len(out) >= limit:
                break
            q = _first(row, ["input", "question", "prompt"])
            target = _first(row, ["target", "target_label", "answer", "label"])
            out.append({"dataset": "bbh", "question": q, "target": target})
        return out
    except Exception as exc:
        print(f"[load_bbh] fallback (reason: {exc})", file=sys.stderr)
        return []


def load_all(limit: int):
    items = []
    items.extend(load_gsm8k(limit))
    items.extend(load_math(limit))
    items.extend(load_bbh(limit))
    if not items:
        print("[load_all] all datasets failed; using 60 built-in fallback prompts", file=sys.stderr)
        items = build_fallback_prompts()
    # Stratify: cap each dataset at limit
    counts = collections.Counter()
    capped = []
    for it in items:
        if counts[it["dataset"]] >= limit:
            continue
        capped.append(it)
        counts[it["dataset"]] += 1
    print(f"[load_all] dataset counts: {dict(counts)} total={len(capped)}")
    return capped


# ---------------------------------------------------------------------------
# Answer extraction / grading
# ---------------------------------------------------------------------------

def normalize_number(s: str):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_last_number(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else ""


def extract_boxed(text: str):
    m = re.findall(r"\\boxed\{([^}]*)\}", text or "")
    return m[-1].strip() if m else ""


def extract_bbh(text: str):
    if not text:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def grade(item, output: str) -> bool:
    ds = item["dataset"]
    target = str(item["target"]).strip()
    if ds == "gsm8k":
        got = extract_last_number(output)
        a, b = normalize_number(got), normalize_number(target)
        if a is not None and b is not None:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return got == target
    if ds == "math":
        got = extract_boxed(output) or extract_last_number(output)
        a, b = normalize_number(got), normalize_number(target)
        if a is not None and b is not None:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        norm = lambda x: re.sub(r"\s+", "", str(x).strip().lower())
        return norm(got) == norm(target)
    # BBH: target string must appear (case-insensitive); fallback to last-line match
    if not target:
        return False
    low_out = (output or "").lower()
    low_t = target.lower().strip()
    if low_t in low_out:
        return True
    return extract_bbh(output).lower() == low_t


def repetition_rate(text: str) -> float:
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


# ---------------------------------------------------------------------------
# vLLM loop
# ---------------------------------------------------------------------------

def format_prompt(q: str) -> str:
    return f"Question: {q}\nAnswer:"


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def run_loop(n_loops: int, items, args, checkpoint_path: Path):
    from vllm import LLM, SamplingParams

    print(f"[loop n={n_loops}] starting LLM...")
    llm = LLM(
        model=args.model,
        hf_overrides={"n_recurrent_loops": n_loops, "recurrent_kv_cache_mode": "last_loop"},
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    sp_math = SamplingParams(temperature=0, max_tokens=args.max_tokens_math)
    sp_bbh = SamplingParams(temperature=0, max_tokens=args.max_tokens_bbh)

    t0 = time.time()
    results = []
    total_tokens = 0
    total_correct = 0

    with open(checkpoint_path, "a") as ckpt:
        for start, chunk in chunked(items, 25):
            # Group by max_tokens needs inside the chunk
            groups = {"math": [], "bbh": []}
            for j, it in enumerate(chunk):
                key = "bbh" if it["dataset"] == "bbh" else "math"
                groups[key].append((start + j, it))
            outputs_by_idx = {}
            for key, lst in groups.items():
                if not lst:
                    continue
                prompts = [format_prompt(it["question"]) for _, it in lst]
                sp = sp_bbh if key == "bbh" else sp_math
                outs = llm.generate(prompts, sp)
                for (idx, it), out in zip(lst, outs):
                    outputs_by_idx[idx] = (it, out)
            # Preserve chunk order for checkpointing
            for idx in sorted(outputs_by_idx):
                it, out = outputs_by_idx[idx]
                text = out.outputs[0].text if out.outputs else ""
                tok_ids = out.outputs[0].token_ids if out.outputs else []
                ntok = len(tok_ids) if tok_ids else len(text.split())
                correct = grade(it, text)
                rr = repetition_rate(text)
                total_tokens += ntok
                total_correct += int(correct)
                rec = {
                    "loop": n_loops,
                    "idx": idx,
                    "dataset": it["dataset"],
                    "question": it["question"],
                    "target": str(it["target"]),
                    "output": text,
                    "extracted": extract_last_number(text) if it["dataset"] == "gsm8k" else (extract_boxed(text) if it["dataset"] == "math" else extract_bbh(text)),
                    "correct": bool(correct),
                    "output_tokens": int(ntok),
                    "rep_rate": float(rr),
                }
                results.append(rec)
                ckpt.write(json.dumps(rec) + "\n")
            ckpt.flush()  # checkpoint every 25 prompts
            print(f"[loop n={n_loops}] checkpoint {min(start + 25, len(items))}/{len(items)}")

    elapsed = time.time() - t0
    metrics = {
        "n_loops": n_loops,
        "accuracy": total_correct / len(items) if items else 0.0,
        "correct": total_correct,
        "total": len(items),
        "output_tokens": total_tokens,
        "rep_rate": sum(r["rep_rate"] for r in results) / len(results) if results else 0.0,
        "throughput_prompts_per_s": len(items) / elapsed if elapsed > 0 else 0.0,
        "throughput_tokens_per_s": total_tokens / elapsed if elapsed > 0 else 0.0,
        "time_s": elapsed,
    }

    del llm
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return results, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="E5 hard-loop benchmark")
    p.add_argument("--out", default="results/e5_hard_loop", help="Output directory")
    p.add_argument("--loops", type=int, nargs="+", default=[2, 4], help="Recurrent loop counts")
    p.add_argument("--limit", type=int, default=100, help="Prompts per dataset (stratify)")
    p.add_argument("--max-tokens-math", type=int, default=512, dest="max_tokens_math")
    p.add_argument("--max-tokens-bbh", type=int, default=256, dest="max_tokens_bbh")
    p.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", help="vLLM model path / HF id")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_vllm()

    items = load_all(args.limit)
    out_dir = Path(args.out)
    # Support --out being a .json/.md file path: use its parent dir.
    if out_dir.suffix in {".json", ".md"}:
        out_dir = out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoints.jsonl"
    checkpoint_path.write_text("")  # fresh checkpoint for this run

    per_loop = {}
    results_by_loop = {}

    for n in args.loops:
        results, metrics = run_loop(n, items, args, checkpoint_path)
        results_by_loop[n] = results
        per_loop[str(n)] = metrics
        print(f"[loop n={n}] acc={metrics['accuracy']:.4f} tokens={metrics['output_tokens']} time={metrics['time_s']:.1f}s")

    # Oracle + improvement / degradation vs first loop
    first_n = args.loops[0]
    last_n = args.loops[-1]
    first_res = {r["idx"]: r for r in results_by_loop[first_n]}
    last_res = {r["idx"]: r for r in results_by_loop[last_n]}
    oracle_correct = 0
    for idx in range(len(items)):
        if any(results_by_loop[n][idx]["correct"] for n in args.loops):
            oracle_correct += 1
    oracle_accuracy = oracle_correct / len(items) if items else 0.0
    first_acc = per_loop[str(first_n)]["accuracy"]
    last_acc = per_loop[str(last_n)]["accuracy"]
    improved = sum(1 for idx in range(len(items)) if not first_res[idx]["correct"] and last_res[idx]["correct"])
    degraded = sum(1 for idx in range(len(items)) if first_res[idx]["correct"] and not last_res[idx]["correct"])

    summary = {
        "config": {
            "out": str(out_dir),
            "loops": args.loops,
            "limit": args.limit,
            "max_tokens_math": args.max_tokens_math,
            "max_tokens_bbh": args.max_tokens_bbh,
            "model": args.model,
            "total_prompts": len(items),
        },
        "per_loop": per_loop,
        "oracle_accuracy": oracle_accuracy,
        "oracle_correct": oracle_correct,
        "loop_gain": last_acc - first_acc,
        "oracle_gain": oracle_accuracy - first_acc,
        "improvement_rate": improved / len(items) if items else 0.0,
        "degradation_rate": degraded / len(items) if items else 0.0,
        "improved": improved,
        "degraded": degraded,
    }

    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# E5 Hard-Loop Benchmark Summary",
        "",
        f"- Model: `{args.model}`",
        f"- Prompts: {len(items)} (limit/dataset={args.limit})",
        f"- Loops: {args.loops}",
        f"- Max tokens: math={args.max_tokens_math}, bbh={args.max_tokens_bbh}",
        "",
        "## Per-loop metrics",
        "",
        "| Loops | Accuracy | Output tokens | Rep rate | Throughput (tok/s) | Time (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in args.loops:
        m = per_loop[str(n)]
        lines.append(
            f"| {n} | {m['accuracy']:.4f} | {m['output_tokens']} | "
            f"{m['rep_rate']:.4f} | {m['throughput_tokens_per_s']:.1f} | {m['time_s']:.1f} |"
        )
    lines += [
        "",
        f"- Oracle accuracy: {oracle_accuracy:.4f} ({oracle_correct}/{len(items)})",
        f"- Loop gain (last - first): {summary['loop_gain']:+.4f}",
        f"- Oracle gain: {summary['oracle_gain']:+.4f}",
        f"- Improvement rate (first wrong -> last correct): {summary['improvement_rate']:.4f} ({improved})",
        f"- Degradation rate (first correct -> last wrong): {summary['degradation_rate']:.4f} ({degraded})",
        "",
        f"Wrote `{out_dir / 'results.json'}` and `{checkpoint_path}`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Done. Wrote {out_dir / 'results.json'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
