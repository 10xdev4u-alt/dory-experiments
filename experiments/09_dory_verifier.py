#!/usr/bin/env python3
"""E9: Dory-as-verifier self-consistency.

Uses Dory at loops 1-4 to generate candidate answers, then uses Dory as a
verifier scoring each candidate (yes/no verification prompt). Compares:
  - self-consistency (majority vote over normalized candidates)
  - verifier-selected (highest P(yes) / yes-no logprob margin)
  - single 4-loop (or max loop) baseline

Deterministic greedy decoding throughout. vLLM backend with hf_overrides
for loop count. Verification uses generated yes/no with logprobs when
available, with prompt-text fallback.

Usage:
  python experiments/09_dory_verifier.py --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
      --out results/e9 --loops 1 2 3 4 --n-candidates 4 --limit 50
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"


# -----------------------------
# vLLM backend
# -----------------------------

def ensure_vllm() -> None:
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
                "git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head",
            ]
        )


class VLLMBackend:
    """Caches one vLLM LLM instance per loop count (hf_overrides)."""

    name = "vllm"

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
    ):
        ensure_vllm()
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llms: Dict[int, Any] = {}

    def get_llm(self, loops: int) -> Any:
        loops = int(loops)
        if loops not in self._llms:
            from vllm import LLM

            self._llms[loops] = LLM(
                model=self.model_name,
                dtype="auto",
                hf_overrides={
                    "n_recurrent_loops": loops,
                    "recurrent_kv_cache_mode": "last_loop",
                },
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
            )
        return self._llms[loops]

    def generate(self, prompt: str, max_new_tokens: int, loops: int) -> Dict[str, Any]:
        from vllm import SamplingParams

        llm = self.get_llm(loops)
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        t0 = time.time()
        out = llm.generate([prompt], sp)
        dt = time.time() - t0
        choice = out[0].outputs[0]
        text = (choice.text or "").strip()
        token_ids = getattr(choice, "token_ids", None) or []
        ntok = len(token_ids)
        return {"text": text, "ntok": int(ntok), "dt": dt}


# -----------------------------
# Verification (yes/no with logprobs)
# -----------------------------

_YES_TOKENS = {"yes", "Yes", "YES", " yes", " Yes", "YES", " y", "y", "Y"}
_NO_TOKENS = {"no", "No", "NO", " no", " No", "NO", " n", "n", "N"}


def _best_logprob(token_logprobs: Optional[Dict[Any, Any]], candidates: set[str]) -> Optional[float]:
    if not token_logprobs:
        return None
    best: Optional[float] = None
    for key, lp_obj in token_logprobs.items():
        tok = getattr(lp_obj, "decoded_token", None)
        if tok is None and isinstance(key, str):
            tok = key
        if tok is None:
            continue
        # Normalize common whitespace variants but keep exact candidates too.
        if tok in candidates or tok.strip() in {c.strip() for c in candidates}:
            lp = getattr(lp_obj, "logprob", None)
            if lp is None:
                continue
            lp_f = float(lp)
            if best is None or lp_f > best:
                best = lp_f
    return best


def _fallback_yes_no(generated_text: str) -> Tuple[float, float, str]:
    t = (generated_text or "").strip().lower()
    # Match explicit leading yes/no to avoid picking up "no" inside other words.
    if re.match(r"^yes\b", t):
        return 0.0, -10.0, "yes"
    if re.match(r"^no\b", t):
        return -10.0, 0.0, "no"
    if "yes" in t and "no" not in t:
        return -1.0, -5.0, "yes"
    if "no" in t and "yes" not in t:
        return -5.0, -1.0, "no"
    return -5.0, -5.0, "no"


def verify_candidate(
    backend: VLLMBackend,
    question: str,
    candidate: str,
    loops: int,
) -> Dict[str, Any]:
    """Score candidate with a yes/no verification prompt.

    Uses generated yes/no with vLLM logprobs when available. Falls back to
    prompt-text parsing. Returns margin = log P(yes) - log P(no) over the
    yes/no pair for compatibility with downstream selection/metrics.
    """
    prompt = (
        "You are a strict verifier. Judge whether the proposed answer is correct.\n"
        f"Question: {question}\n"
        f"Proposed answer: {candidate}\n"
        "Is the proposed answer correct? Answer yes or no:"
    )
    from vllm import SamplingParams

    llm = backend.get_llm(loops)
    sp = SamplingParams(max_tokens=4, temperature=0.0, logprobs=20)
    out = llm.generate([prompt], sp)[0]
    choice = out.outputs[0]
    gen_text = (choice.text or "").strip()
    gen_logprobs = getattr(choice, "logprobs", None) or []
    first = gen_logprobs[0] if gen_logprobs else None
    lp_yes = _best_logprob(first, _YES_TOKENS)
    lp_no = _best_logprob(first, _NO_TOKENS)
    if lp_yes is None or lp_no is None:
        lp_yes, lp_no, verdict = _fallback_yes_no(gen_text)
    else:
        verdict = "yes" if lp_yes >= lp_no else "no"
    margin = float(lp_yes - lp_no)
    m = max(lp_yes, lp_no)
    try:
        p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))
    except Exception:
        p_yes = 0.5
    return {
        "lp_yes": float(lp_yes),
        "lp_no": float(lp_no),
        "margin": margin,
        "p_yes_pair": float(p_yes),
        "verdict": verdict,
        "verifier_prompt": prompt,
        "verifier_text": gen_text,
    }


# -----------------------------
# Evaluation helpers
# -----------------------------

def normalize_answer(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"^(final answer|answer|a)\s*[:\-]\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .,!?:;\"'`")
    # Pull a short trailing/leading numeric or token span when obvious.
    m = re.search(r"(-?\d+(?:[./]\d+)?)", t)
    if m and len(t) <= 40:
        return m.group(1)
    return t


def is_correct(pred: str, gold: str) -> bool:
    pn = normalize_answer(pred)
    gn = normalize_answer(gold)
    if not pn and not gn:
        return True
    if not pn or not gn:
        return False
    return gn == pn or gn in pn or pn in gn


def majority_vote(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {"text": "", "normalized": "", "count": 0, "support": 0.0}
    norms = [c["normalized"] for c in candidates]
    counts = Counter(norms)
    best_norm, best_count = counts.most_common(1)[0]
    # Tie-break: prefer candidate from highest loop, then first seen.
    tied = [n for n, c in counts.items() if c == best_count]
    chosen = candidates[0]
    for c in candidates:
        if c["normalized"] in tied:
            chosen = c
            if len(tied) == 1:
                break
    # If there was a tie, pick among tied by highest loop.
    if len(tied) > 1:
        tied_cands = [c for c in candidates if c["normalized"] in tied]
        chosen = sorted(tied_cands, key=lambda c: (-c["loops"], c["index"]))[0]
        best_norm = chosen["normalized"]
        best_count = counts[best_norm]
    return {
        "text": chosen["text"],
        "normalized": best_norm,
        "count": int(best_count),
        "support": best_count / max(1, len(candidates)),
        "counts": dict(counts),
    }


def verifier_select(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {"text": "", "normalized": "", "score": float("-inf")}
    # Highest margin, tie-break by p_yes_pair then higher loop.
    chosen = sorted(
        candidates,
        key=lambda c: (c["verify"]["margin"], c["verify"]["p_yes_pair"], c["loops"]),
        reverse=True,
    )[0]
    return {
        "text": chosen["text"],
        "normalized": chosen["normalized"],
        "score": chosen["verify"]["margin"],
        "p_yes_pair": chosen["verify"]["p_yes_pair"],
        "selected_loops": chosen["loops"],
    }


# -----------------------------
# Data / experiment
# -----------------------------

def build_suite(limit: int) -> List[Dict[str, str]]:
    base = [
        {"task": "math", "question": "What is 17*23?", "gold": "391"},
        {"task": "math", "question": "Solve for x: 2x+5=13. Give only x.", "gold": "4"},
        {"task": "math", "question": "What is 100/4+7?", "gold": "32"},
        {"task": "math", "question": "What is 15% of 200?", "gold": "30"},
        {"task": "trivia", "question": "What is the capital of France?", "gold": "Paris"},
        {"task": "trivia", "question": "What is the largest planet in our solar system?", "gold": "Jupiter"},
        {"task": "trivia", "question": "Who wrote Romeo and Juliet?", "gold": "Shakespeare"},
        {"task": "reasoning", "question": "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no.", "gold": "yes"},
        {"task": "reasoning", "question": "Tom has 3 apples and buys 2 more. How many apples does he have?", "gold": "5"},
    ]
    if limit <= 0:
        return base
    return (base * (limit // len(base) + 1))[:limit]


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.seed is not None:
        random.seed(args.seed)

    loop_values = list(args.loops) if args.loops else [1, 2, 3, 4]
    n_candidates = min(args.n_candidates, len(loop_values))
    candidate_loops = loop_values[:n_candidates]
    baseline_loop = 4 if 4 in loop_values else max(loop_values)
    verifier_loop = baseline_loop

    suite = build_suite(args.limit)
    records: List[Dict[str, Any]] = []
    summary_counts = Counter()

    backend = VLLMBackend(
        args.model,
        max_model_len=getattr(args, "max_model_len", 4096),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.85),
    )

    for i, ex in enumerate(suite):
        question = ex["question"]
        gold = ex["gold"]
        prompt = f"Question: {question}\nAnswer briefly and exactly:"

        candidates: List[Dict[str, Any]] = []
        for j, L in enumerate(candidate_loops):
            gen = backend.generate(prompt, args.max_new_tokens, loops=L)
            cand = {
                "index": j,
                "loops": L,
                "text": gen["text"],
                "normalized": normalize_answer(gen["text"]),
                "ntok": gen["ntok"],
                "dt": gen["dt"],
            }
            ver = verify_candidate(backend, question, gen["text"], loops=verifier_loop)
            cand["verify"] = ver
            candidates.append(cand)

        mv = majority_vote(candidates)
        vs = verifier_select(candidates)

        baseline = None
        for c in candidates:
            if c["loops"] == baseline_loop:
                baseline = {"text": c["text"], "normalized": c["normalized"], "loops": c["loops"]}
                break
        if baseline is None:
            gen_b = backend.generate(prompt, args.max_new_tokens, loops=baseline_loop)
            baseline = {
                "text": gen_b["text"],
                "normalized": normalize_answer(gen_b["text"]),
                "loops": baseline_loop,
            }

        mv_ok = is_correct(mv["text"], gold)
        vs_ok = is_correct(vs["text"], gold)
        base_ok = is_correct(baseline["text"], gold)

        summary_counts["n"] += 1
        summary_counts["majority_correct"] += int(mv_ok)
        summary_counts["verifier_correct"] += int(vs_ok)
        summary_counts["single4_correct"] += int(base_ok)

        rec = {
            "i": i,
            "task": ex["task"],
            "question": question,
            "gold": gold,
            "candidates": [
                {
                    "loops": c["loops"],
                    "text": c["text"],
                    "normalized": c["normalized"],
                    "ntok": c["ntok"],
                    "dt": c["dt"],
                    "verify": {k: v for k, v in c["verify"].items() if k != "verifier_prompt"},
                }
                for c in candidates
            ],
            "majority_vote": mv,
            "verifier_selected": vs,
            "single_loop_baseline": baseline,
            "correct": {
                "majority_vote": mv_ok,
                "verifier_selected": vs_ok,
                "single_loop_baseline": base_ok,
            },
        }
        records.append(rec)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i+1}/{len(suite)}] mv={mv_ok} ver={vs_ok} base{baseline_loop}={base_ok} "
                f"q={question[:60]}",
                flush=True,
            )
        gc.collect()

    n = max(1, summary_counts["n"])
    summary = {
        "model": args.model,
        "candidate_loops": candidate_loops,
        "verifier_loop": verifier_loop,
        "single_loop_baseline": baseline_loop,
        "n": summary_counts["n"],
        "accuracy": {
            "majority_vote": summary_counts["majority_correct"] / n,
            "verifier_selected": summary_counts["verifier_correct"] / n,
            "single_loop_baseline": summary_counts["single4_correct"] / n,
        },
        "correct_counts": {
            "majority_vote": int(summary_counts["majority_correct"]),
            "verifier_selected": int(summary_counts["verifier_correct"]),
            "single_loop_baseline": int(summary_counts["single4_correct"]),
        },
    }

    return {
        "experiment": "E9_dory_verifier_self_consistency",
        "backend": "vllm",
        "deterministic_greedy": True,
        "args": {
            "model": args.model,
            "out": args.out,
            "loops": loop_values,
            "n_candidates": args.n_candidates,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": getattr(args, "max_model_len", 4096),
            "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", 0.85),
            "seed": args.seed,
        },
        "summary": summary,
        "records": records,
    }


# -----------------------------
# Output
# -----------------------------

def render_markdown(result: Dict[str, Any]) -> str:
    s = result["summary"]
    acc = s["accuracy"]
    backend = result.get("backend", "vllm")
    lines: List[str] = []
    lines.append("# E9: Dory-as-verifier self-consistency")
    lines.append("")
    lines.append(f"- Model: `{s['model']}`")
    lines.append(f"- Backend: `{backend}` (deterministic greedy)")
    lines.append(f"- Candidate loops: `{s['candidate_loops']}`")
    lines.append(f"- Verifier loop: `{s['verifier_loop']}`")
    lines.append(f"- Single-loop baseline: `{s['single_loop_baseline']}`")
    lines.append(f"- Cases: `{s['n']}`")
    lines.append("")
    lines.append("## Accuracy comparison")
    lines.append("")
    lines.append(f"- Self-consistency (majority vote): **{acc['majority_vote']:.3f}** ({s['correct_counts']['majority_vote']}/{s['n']})")
    lines.append(f"- Verifier-selected (yes/no score): **{acc['verifier_selected']:.3f}** ({s['correct_counts']['verifier_selected']}/{s['n']})")
    lines.append(f"- Single {s['single_loop_baseline']}-loop baseline: **{acc['single_loop_baseline']:.3f}** ({s['correct_counts']['single_loop_baseline']}/{s['n']})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Candidates are generated greedily at the specified loop counts via vLLM with `hf_overrides.n_recurrent_loops`.")
    lines.append("- Each candidate is scored by a yes/no verification prompt. The verifier generates a yes/no answer and uses vLLM logprobs for yes/no when available; it falls back to parsing the generated verdict.")
    lines.append("- Verifier score is reported as `margin = log P(yes) - log P(no)` over the yes/no pair.")
    lines.append("- Majority vote normalizes answers (case/punctuation/short numeric extraction) before counting support.")
    lines.append("")
    lines.append("## Sample cases")
    lines.append("")
    for rec in result["records"][:10]:
        lines.append(f"### Case {rec['i']}: {rec['task']}")
        lines.append("")
        lines.append(f"- Q: {rec['question']}")
        lines.append(f"- Gold: `{rec['gold']}`")
        lines.append(f"- Majority: `{rec['majority_vote']['text']}` (correct={rec['correct']['majority_vote']})")
        lines.append(f"- Verifier: `{rec['verifier_selected']['text']}` score={rec['verifier_selected']['score']:.3f} (correct={rec['correct']['verifier_selected']})")
        lines.append(f"- Single loop: `{rec['single_loop_baseline']['text']}` (correct={rec['correct']['single_loop_baseline']})")
        cand_bits = []
        for c in rec["candidates"]:
            v = c["verify"]
            cand_bits.append(f"L{c['loops']}:{c['normalized'] or '∅'}(margin={v['margin']:.2f},{v['verdict']})")
        lines.append(f"- Candidates: {' | '.join(cand_bits)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_outputs(result: Dict[str, Any], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "e9_dory_verifier.json")
    md_path = os.path.join(out_dir, "e9_dory_verifier.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
    return json_path, md_path


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E9: Dory-as-verifier self-consistency")
    p.add_argument("--model", default=DEFAULT_MODEL, help="vLLM/Dory model name or path")
    p.add_argument("--out", default="results/e9", help="Output directory for JSON and markdown")
    p.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3, 4], help="Loop counts used as candidate generators")
    p.add_argument("--n-candidates", type=int, default=4, help="Number of loop candidates to use (from --loops order)")
    p.add_argument("--limit", type=int, default=50, help="Number of suite cases to evaluate")
    p.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for candidate generation")
    p.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    p.add_argument("--seed", type=int, default=42, help="Random seed (deterministic greedy decoding)")
    args = p.parse_args(argv)
    if args.n_candidates < 1:
        p.error("--n-candidates must be >= 1")
    if args.limit < 0:
        p.error("--limit must be >= 0")
    if not args.loops:
        p.error("--loops must include at least one loop count")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_experiment(args)
    json_path, md_path = save_outputs(result, args.out)
    acc = result["summary"]["accuracy"]
    print("Experiment: E9_dory_verifier_self_consistency")
    print(f"Model: {args.model}")
    print(f"Cases: {result['summary']['n']} Candidate loops: {result['summary']['candidate_loops']}")
    print(
        "Accuracy: "
        f"majority_vote={acc['majority_vote']:.3f} "
        f"verifier_selected={acc['verifier_selected']:.3f} "
        f"single_loop={acc['single_loop_baseline']:.3f}"
    )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
