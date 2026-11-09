#!/usr/bin/env python3
"""E8: Confidence-gated escalation.

For each prompt loaded from E5 checkpoints, compute a confidence score and
escalate to loop 4 only when confidence < threshold. Compares the gated
strategy against static loop2, static loop4, and the loop2/loop4 oracle.

Confidence:
  - ideal: agreement across N=3 greedy samples at loop2 (unavailable here)
  - proxy 1: 1 if loop2 correct else 0 (requires labels)
  - proxy 2: 1 if loop2 answer matches loop1 answer (requires loop1 data)
  - fallback (used when loop1 data is unavailable): prompt-length heuristic,
    confidence = 1.0 if len(question) <= median prompt length else 0.0
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_checkpoints(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_answer(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def group_by_prompt(rows):
    by_idx = defaultdict(dict)
    for r in rows:
        try:
            loop = int(r.get("loop"))
        except (TypeError, ValueError):
            continue
        by_idx[r.get("idx")][loop] = r
    return by_idx


def build_samples(ckpt_path):
    rows = load_checkpoints(ckpt_path)
    by_idx = group_by_prompt(rows)

    samples = []
    for idx, loops in by_idx.items():
        r2 = loops.get(2)
        r4 = loops.get(4)
        if r2 is None or r4 is None:
            continue
        samples.append(
            {
                "idx": idx,
                "dataset": r2.get("dataset"),
                "question": r2.get("question", ""),
                "target": r2.get("target"),
                "loop1": loops.get(1),
                "loop2": r2,
                "loop4": r4,
            }
        )
    return samples


def compute_confidence(samples):
    """Return (confidences list, method description, loop1_available bool)."""
    loop1_available = any(s["loop1"] is not None for s in samples)
    confidences = []

    if loop1_available:
        for s in samples:
            r1 = s["loop1"]
            if r1 is not None:
                conf = (
                    1.0
                    if normalize_answer(r1.get("extracted"))
                    == normalize_answer(s["loop2"].get("extracted"))
                    else 0.0
                )
            else:
                conf = None
            confidences.append(conf)
        # Fill missing with prompt-length heuristic among those missing.
        missing_indices = [i for i, c in enumerate(confidences) if c is None]
        if missing_indices:
            lengths = [len(samples[i]["question"]) for i in missing_indices]
            median_len = statistics.median(lengths) if lengths else 0
            for i in missing_indices:
                confidences[i] = (
                    1.0 if len(samples[i]["question"]) <= median_len else 0.0
                )
        return confidences, "loop1_match (prompt-length fallback for missing)", True

    # Fallback: prompt-length heuristic.
    lengths = [len(s["question"]) for s in samples]
    median_len = statistics.median(lengths) if lengths else 0
    for s in samples:
        confidences.append(1.0 if len(s["question"]) <= median_len else 0.0)
    return (
        confidences,
        "prompt_length_heuristic (median split: 1.0 if len(question) <= median else 0.0)",
        False,
    )


def evaluate(samples, confidences, threshold):
    decisions = []
    gated_correct = 0
    n = len(samples)
    n_escalated = 0
    for s, conf in zip(samples, confidences):
        escalate = bool(conf < threshold)
        if escalate:
            n_escalated += 1
            chosen = s["loop4"]
        else:
            chosen = s["loop2"]
        chosen_correct = bool(chosen.get("correct", False))
        if chosen_correct:
            gated_correct += 1
        decisions.append(
            {
                "idx": s["idx"],
                "dataset": s["dataset"],
                "confidence": float(conf),
                "escalated": escalate,
                "loop2_correct": bool(s["loop2"].get("correct", False)),
                "loop4_correct": bool(s["loop4"].get("correct", False)),
                "chosen_correct": chosen_correct,
            }
        )

    static_loop2 = sum(1 for s in samples if s["loop2"].get("correct")) / n if n else 0.0
    static_loop4 = sum(1 for s in samples if s["loop4"].get("correct")) / n if n else 0.0
    oracle = sum(
        1 for s in samples if (s["loop2"].get("correct") or s["loop4"].get("correct"))
    ) / n if n else 0.0
    gated = gated_correct / n if n else 0.0

    return {
        "n": n,
        "n_escalated": n_escalated,
        "escalation_rate": n_escalated / n if n else 0.0,
        "net_accuracy_gated": gated,
        "static_loop2_accuracy": static_loop2,
        "static_loop4_accuracy": static_loop4,
        "oracle_accuracy": oracle,
        "gated_correct": gated_correct,
        "decisions": decisions,
    }


def render_markdown(result):
    s = result["summary"]
    lines = []
    lines.append("# E8: Confidence-Gated Escalation")
    lines.append("")
    lines.append(f"- Checkpoint: `{result['checkpoint']}`")
    lines.append(f"- Threshold: `{result['threshold']}`")
    lines.append(f"- Confidence method: `{result['confidence_method']}`")
    lines.append(f"- Loop1 available: `{result['loop1_available']}`")
    lines.append(f"- Samples: `{s['n']}`")
    lines.append(f"- Escalated to loop4: `{s['n_escalated']}` ({s['escalation_rate']:.2%})")
    lines.append("")
    lines.append("## Accuracy comparison")
    lines.append("")
    lines.append("| Strategy | Accuracy |")
    lines.append("|---|---:|")
    lines.append(f"| Static loop2 | {s['static_loop2_accuracy']:.4f} |")
    lines.append(f"| Static loop4 | {s['static_loop4_accuracy']:.4f} |")
    lines.append(f"| Confidence-gated | {s['net_accuracy_gated']:.4f} |")
    lines.append(f"| Oracle (best of 2/4) | {s['oracle_accuracy']:.4f} |")
    lines.append("")
    lines.append(
        f"Gain vs static loop2: {s['net_accuracy_gated'] - s['static_loop2_accuracy']:+.4f}"
    )
    lines.append(
        f"Gap to oracle: {s['oracle_accuracy'] - s['net_accuracy_gated']:.4f}"
    )
    lines.append("")
    lines.append("## Family counts")
    lines.append("")
    for fam, cnt in sorted(result.get("family_counts", {}).items()):
        lines.append(f"- {fam}: {cnt}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="E8 confidence-gated escalation")
    ap.add_argument("--out", default="results/e8", help="Output directory")
    ap.add_argument("--ckpt", default="results/e5/checkpoints.jsonl", help="E5 checkpoints.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5, help="Escalate if confidence < threshold")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(args.ckpt)
    if not samples:
        raise SystemExit(f"No samples built from {args.ckpt}")

    confidences, method, loop1_available = compute_confidence(samples)
    eval_result = evaluate(samples, confidences, args.threshold)

    family_counts = dict(Counter(s["dataset"] for s in samples))

    result = {
        "experiment": "E8 confidence-gated escalation",
        "checkpoint": str(args.ckpt),
        "threshold": float(args.threshold),
        "confidence_method": method,
        "loop1_available": bool(loop1_available),
        "family_counts": family_counts,
        "summary": {
            "n": eval_result["n"],
            "n_escalated": eval_result["n_escalated"],
            "escalation_rate": eval_result["escalation_rate"],
            "net_accuracy_gated": eval_result["net_accuracy_gated"],
            "static_loop2_accuracy": eval_result["static_loop2_accuracy"],
            "static_loop4_accuracy": eval_result["static_loop4_accuracy"],
            "oracle_accuracy": eval_result["oracle_accuracy"],
            "gated_correct": eval_result["gated_correct"],
        },
        "decisions": eval_result["decisions"],
    }

    md = render_markdown(result)

    # Write primary named files.
    (out_dir / "08_confidence_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "08_confidence_gate.md").write_text(md, encoding="utf-8")
    # Compatibility aliases.
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    s = result["summary"]
    print(f"Samples: {s['n']}  escalated: {s['n_escalated']} ({s['escalation_rate']:.2%})")
    print(f"Static loop2: {s['static_loop2_accuracy']:.4f}")
    print(f"Static loop4: {s['static_loop4_accuracy']:.4f}")
    print(f"Confidence-gated: {s['net_accuracy_gated']:.4f}")
    print(f"Oracle: {s['oracle_accuracy']:.4f}")
    print(f"Wrote: {out_dir / '08_confidence_gate.json'}")
    print(f"Wrote: {out_dir / '08_confidence_gate.md'}")


if __name__ == "__main__":
    main()
