#!/usr/bin/env python3
"""E9: Dory-as-verifier self-consistency.

Uses Dory at loops 1-4 to generate candidate answers, then uses Dory as a
verifier scoring each candidate (yes/no verification prompt scored by
log-likelihood). Compares:
  - self-consistency (majority vote over normalized candidates)
  - verifier-selected (highest P(yes) / yes-no log-likelihood margin)
  - single 4-loop (or max loop) baseline

Deterministic greedy decoding throughout.

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
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"


# -----------------------------
# Model loading / loop control
# -----------------------------

def set_loop_count(model: Any, loops: int) -> None:
    """Best-effort loop count setter for Dory-style recurrent models."""
    if hasattr(model, "config") and hasattr(model.config, "n_recurrent_loops"):
        model.config.n_recurrent_loops = loops
    if hasattr(model, "set_num_loops"):
        model.set_num_loops(loops)
    # Some implementations expose generation/forward config variants.
    if hasattr(model, "model") and hasattr(model.model, "config"):
        cfg = model.model.config
        if hasattr(cfg, "n_recurrent_loops"):
            cfg.n_recurrent_loops = loops


def load_model(model_id: str, loops: int, device: str = "cuda:0") -> Tuple[Any, Any]:
    try:
        import transformers.configuration_utils as _cu
        if "mlp" not in getattr(_cu, "ALLOWED_ATTN_LAYER_TYPES", ()):
            _cu.ALLOWED_ATTN_LAYER_TYPES = _cu.ALLOWED_ATTN_LAYER_TYPES + ("mlp",)
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    set_loop_count(model, loops)
    model.eval()
    return model, tok


# -----------------------------
# Generation / scoring
# -----------------------------

def greedy_generate(
    model: Any,
    tok: Any,
    prompt: str,
    max_new_tokens: int,
    loops: Optional[int] = None,
) -> Dict[str, Any]:
    if loops is not None:
        set_loop_count(model, loops)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return {"text": text, "ntok": int(gen_ids.shape[0]), "dt": dt}


def token_logprob_for_completion(model: Any, tok: Any, prompt: str, completion: str) -> float:
    """Log-likelihood of completion tokens given prompt (teacher-forced)."""
    full = prompt + completion
    enc_prompt = tok(prompt, return_tensors="pt")
    enc_full = tok(full, return_tensors="pt")
    enc_prompt = {k: v.to(model.device) for k, v in enc_prompt.items()}
    enc_full = {k: v.to(model.device) for k, v in enc_full.items()}
    prompt_len = enc_prompt["input_ids"].shape[1]
    full_ids = enc_full["input_ids"]
    if full_ids.shape[1] <= prompt_len:
        return float("-inf")
    with torch.inference_mode():
        outputs = model(**enc_full)
        logits = outputs.logits  # [1, seq, vocab]
    # Score completion tokens: token at t predicted by logits at t-1.
    target_ids = full_ids[0, prompt_len:]
    pred_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    token_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return float(token_lp.sum().item())


def verify_candidate(
    model: Any,
    tok: Any,
    question: str,
    candidate: str,
    loops: Optional[int] = None,
) -> Dict[str, Any]:
    """Score candidate with a yes/no verification prompt.

    Primary score is margin: log P(' yes') - log P(' no') on the same prompt.
    Also returns both log-likelihoods.
    """
    if loops is not None:
        set_loop_count(model, loops)
    prompt = (
        "You are a strict verifier. Judge whether the proposed answer is correct.\n"
        f"Question: {question}\n"
        f"Proposed answer: {candidate}\n"
        "Is the proposed answer correct? Answer yes or no:"
    )
    lp_yes = token_logprob_for_completion(model, tok, prompt, " yes")
    lp_no = token_logprob_for_completion(model, tok, prompt, " no")
    margin = lp_yes - lp_no
    # Convert to a stable probability-like score for reporting.
    # P(yes) over {yes,no} by softmax of the two completions.
    m = max(lp_yes, lp_no)
    p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))
    verdict = "yes" if lp_yes >= lp_no else "no"
    return {
        "lp_yes": lp_yes,
        "lp_no": lp_no,
        "margin": margin,
        "p_yes_pair": p_yes,
        "verdict": verdict,
        "verifier_prompt": prompt,
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
        torch.manual_seed(args.seed)

    loop_values = list(args.loops) if args.loops else [1, 2, 3, 4]
    n_candidates = min(args.n_candidates, len(loop_values))
    candidate_loops = loop_values[:n_candidates]
    baseline_loop = 4 if 4 in loop_values else max(loop_values)
    verifier_loop = baseline_loop

    suite = build_suite(args.limit)
    records: List[Dict[str, Any]] = []
    summary_counts = Counter()

    # Single model instance; loop count is switched per operation.
    model, tok = load_model(args.model, loops=candidate_loops[0])

    for i, ex in enumerate(suite):
        question = ex["question"]
        gold = ex["gold"]
        prompt = f"Question: {question}\nAnswer briefly and exactly:"

        candidates: List[Dict[str, Any]] = []
        for j, L in enumerate(candidate_loops):
            gen = greedy_generate(model, tok, prompt, args.max_new_tokens, loops=L)
            cand = {
                "index": j,
                "loops": L,
                "text": gen["text"],
                "normalized": normalize_answer(gen["text"]),
                "ntok": gen["ntok"],
                "dt": gen["dt"],
            }
            ver = verify_candidate(model, tok, question, gen["text"], loops=verifier_loop)
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
            gen_b = greedy_generate(model, tok, prompt, args.max_new_tokens, loops=baseline_loop)
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
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

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
        "backend": "transformers",
        "deterministic_greedy": True,
        "args": {
            "model": args.model,
            "out": args.out,
            "loops": loop_values,
            "n_candidates": args.n_candidates,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
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
    lines: List[str] = []
    lines.append("# E9: Dory-as-verifier self-consistency")
    lines.append("")
    lines.append(f"- Model: `{s['model']}`")
    lines.append(f"- Backend: `transformers` (deterministic greedy)")
    lines.append(f"- Candidate loops: `{s['candidate_loops']}`")
    lines.append(f"- Verifier loop: `{s['verifier_loop']}`")
    lines.append(f"- Single-loop baseline: `{s['single_loop_baseline']}`")
    lines.append(f"- Cases: `{s['n']}`")
    lines.append("")
    lines.append("## Accuracy comparison")
    lines.append("")
    lines.append(f"- Self-consistency (majority vote): **{acc['majority_vote']:.3f}** ({s['correct_counts']['majority_vote']}/{s['n']})")
    lines.append(f"- Verifier-selected (yes/no log-likelihood): **{acc['verifier_selected']:.3f}** ({s['correct_counts']['verifier_selected']}/{s['n']})")
    lines.append(f"- Single {s['single_loop_baseline']}-loop baseline: **{acc['single_loop_baseline']:.3f}** ({s['correct_counts']['single_loop_baseline']}/{s['n']})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Candidates are generated greedily at the specified loop counts; each candidate is scored by a yes/no verification prompt.")
    lines.append("- Verifier score uses `log P(' yes') - log P(' no')` (teacher-forced completion log-likelihood margin).")
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
    p.add_argument("--model", default=DEFAULT_MODEL, help="Transformers/Dory model name or path")
    p.add_argument("--out", default="results/e9", help="Output directory for JSON and markdown")
    p.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3, 4], help="Loop counts used as candidate generators")
    p.add_argument("--n-candidates", type=int, default=4, help="Number of loop candidates to use (from --loops order)")
    p.add_argument("--limit", type=int, default=50, help="Number of suite cases to evaluate")
    p.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for candidate generation")
    p.add_argument("--seed", type=int, default=42, help="Random/torch seed (deterministic greedy still used)")
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
