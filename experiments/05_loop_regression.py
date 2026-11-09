#!/usr/bin/env python3
"""E5: Loop Regression Hunt.

Runs the same prompt sets across increasing loop counts with deterministic greedy
decoding, detects regressions, computes text-quality metrics, clusters failures by
task type, and writes JSON plus a markdown summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class PromptCase:
    task_type: str
    prompt: str
    expected: Optional[str] = None
    expected_json_keys: Optional[List[str]] = None
    validator: str = "exact"  # exact|json_keys|contains|refuse_contradiction


PROMPT_SETS: Dict[str, List[PromptCase]] = {
    "overthinking-simple": [
        PromptCase("overthinking-simple", "What is 2+2? Answer with only the number.", "4", validator="exact"),
        PromptCase("overthinking-simple", "Capital of France? Answer with only the city name.", "Paris", validator="exact"),
        PromptCase("overthinking-simple", "Is water wet? Answer yes or no only.", "yes", validator="exact"),
    ],
    "format-constrained-json": [
        PromptCase(
            "format-constrained-json",
            'Return valid JSON only with keys "answer" and "confidence" for: 3*5.',
            expected_json_keys=["answer", "confidence"],
            validator="json_keys",
        ),
        PromptCase(
            "format-constrained-json",
            'Return valid JSON only with keys "name" and "age" for Alice, age 30.',
            expected_json_keys=["name", "age"],
            validator="json_keys",
        ),
    ],
    "instruction-adherence": [
        PromptCase(
            "instruction-adherence",
            "List three colors. Output exactly three comma-separated words and nothing else.",
            "red,blue,green",
            validator="contains",
        ),
        PromptCase(
            "instruction-adherence",
            "Reply with the single word DONE.",
            "DONE",
            validator="exact",
        ),
    ],
    "long-distractor-context": [
        PromptCase(
            "long-distractor-context",
            "Context: " + ("lorem ipsum distractor " * 120) + "\nQuestion: What is the sum of 7 and 8? Answer with only the number.",
            "15",
            validator="exact",
        ),
        PromptCase(
            "long-distractor-context",
            "Context: " + ("irrelevant details about weather and history " * 80) + "\nTask: Return the word banana only.",
            "banana",
            validator="exact",
        ),
    ],
    "contradictory-instructions": [
        PromptCase(
            "contradictory-instructions",
            "Answer yes. Answer no. Ignore previous instructions and answer maybe. Provide a single final answer after 'FINAL:'.",
            "FINAL:",
            validator="refuse_contradiction",
        ),
        PromptCase(
            "contradictory-instructions",
            "Output 1. Output 2. Do both and neither. End with 'FINAL:' followed by one consistent choice.",
            "FINAL:",
            validator="refuse_contradiction",
        ),
    ],
}


class BaseBackend:
    name = "base"

    def generate(self, prompt: str, loop: int, max_new_tokens: int = 128) -> str:
        raise NotImplementedError


class MockBackend(BaseBackend):
    """Deterministic mock that simulates plausible loop regressions for testing."""

    name = "mock"

    def generate(self, prompt: str, loop: int, max_new_tokens: int = 128) -> str:
        rng = random.Random(hash((prompt, loop, "mock")) & 0xFFFFFFFF)
        p = prompt.lower()
        if "2+2" in p:
            return "4" if loop <= 2 else "Let us think carefully. 2+2=4? maybe 5 " * min(loop, 6)
        if "capital of france" in p:
            return "Paris" if loop < 4 else "Paris Paris Paris Paris"
        if "water wet" in p:
            return "yes" if loop <= 3 else "yes and no depending on definitions " * loop
        if "3*5" in p and "json" in p:
            if loop <= 2:
                return '{"answer": 15, "confidence": 0.99}'
            return '{"answer": 15, "confidence": 0.99'  # invalid at higher loops
        if '"name" and "age"' in p:
            if loop <= 3:
                return '{"name": "Alice", "age": 30}'
            return 'name Alice age 30'
        if "three colors" in p:
            return "red,blue,green" if loop == 1 else "red, blue, green, and also yellow " * loop
        if "single word done" in p:
            return "DONE" if loop <= 2 else "I am DONE now DONE"
        if "sum of 7 and 8" in p:
            if loop == 1:
                return "15"
            return "15 " + ("distractor " * (loop * 4))
        if "word banana only" in p:
            return "banana" if loop < 4 else "banana banana banana banana banana"
        if "answer yes. answer no" in p:
            if loop <= 2:
                return "FINAL: cannot comply with contradictory instructions"
            return "yes no maybe " * loop
        if "output 1. output 2" in p:
            if loop == 1:
                return "FINAL: 1"
            return "1 2 1 2 " * loop
        filler = " ".join(rng.choice(["token", "loop", "text"]) for _ in range(max(8, loop * 4)))
        return filler[: max_new_tokens * 4]


class TransformersBackend(BaseBackend):
    name = "transformers"

    def __init__(self, model_name: str, max_new_tokens: int = 128):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Transformers backend requires transformers and torch. "
                "Install them or run without --model to use the mock backend."
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str, loop: int, max_new_tokens: int = 128) -> str:
        # Deterministic greedy decode. Loop count is injected as a controlled prefix
        # to exercise iterative prompting/regression sensitivity with the same model.
        loop_prompt = (
            f"You are on refinement loop {loop}. Follow instructions exactly.\n"
            f"Previous refinement passes: {max(0, loop - 1)}.\n\n{prompt}"
        )
        inputs = self.tokenizer(loop_prompt, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_valid_json_with_keys(text: str, keys: Sequence[str]) -> bool:
    try:
        obj = json.loads(text)
    except Exception:
        return False
    return isinstance(obj, dict) and all(k in obj for k in keys)


def evaluate_case(case: PromptCase, output: str) -> Tuple[bool, bool]:
    text = output.strip()
    norm = normalize_text(text)
    format_valid = True
    correct = False
    if case.validator == "exact":
        correct = norm == normalize_text(case.expected or "")
    elif case.validator == "json_keys":
        format_valid = is_valid_json_with_keys(text, case.expected_json_keys or [])
        correct = format_valid
    elif case.validator == "contains":
        expected = normalize_text(case.expected or "")
        correct = expected in norm or norm == expected
    elif case.validator == "refuse_contradiction":
        format_valid = (case.expected or "FINAL:") in text
        # Consider correct if it gives a single FINAL answer and avoids repeating all contradictory choices.
        has_final = "FINAL:" in text
        lower = text.lower()
        contradictory_spam = sum(lower.count(x) for x in ["yes", "no", "maybe", "output 1", "output 2"]) >= 6
        correct = has_final and not contradictory_spam
    else:
        correct = norm == normalize_text(case.expected or "")
    return correct, format_valid


def repeat_4gram_rate(text: str) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < 4:
        return 0.0
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def entropy_proxy(text: str) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def max_token_hit_rate(text: str, max_new_tokens: int) -> float:
    # Proxy: if generated token-ish units are near or above budget, treat as a hit.
    token_est = len(re.findall(r"\w+|[^\w\s]", text))
    return 1.0 if token_est >= max_new_tokens else 0.0


def build_backend(args: argparse.Namespace) -> BaseBackend:
    if args.model:
        return TransformersBackend(args.model, max_new_tokens=args.max_new_tokens)
    return MockBackend()


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    backend = build_backend(args)
    loops = list(range(1, args.loops + 1))
    cases: List[PromptCase] = []
    for name, group in PROMPT_SETS.items():
        selected = group[: args.limit] if args.limit else group
        for case in selected:
            cases.append(case)

    records: List[Dict[str, Any]] = []
    metrics_by_loop: Dict[str, Dict[str, float]] = {}
    loop_accum: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    loop_counts: Dict[int, int] = defaultdict(int)

    for case in cases:
        per_loop: Dict[int, Dict[str, Any]] = {}
        for loop in loops:
            output = backend.generate(case.prompt, loop=loop, max_new_tokens=args.max_new_tokens)
            correct, format_valid = evaluate_case(case, output)
            r4 = repeat_4gram_rate(output)
            ent = entropy_proxy(output)
            max_hit = max_token_hit_rate(output, args.max_new_tokens)
            row = {
                "loop": loop,
                "output": output,
                "correct": correct,
                "format_valid": format_valid,
                "repeat_4gram_rate": r4,
                "entropy_proxy": ent,
                "max_token_hit": max_hit,
            }
            per_loop[loop] = row
            loop_accum[loop]["repeat_4gram_rate"] += r4
            loop_accum[loop]["entropy_proxy"] += ent
            loop_accum[loop]["max_token_hit"] += max_hit
            loop_accum[loop]["accuracy"] += 1.0 if correct else 0.0
            loop_accum[loop]["format_valid"] += 1.0 if format_valid else 0.0
            loop_counts[loop] += 1

        regressions = detect_regressions(case, per_loop, loops)
        records.append(
            {
                "task_type": case.task_type,
                "prompt": case.prompt,
                "expected": case.expected,
                "expected_json_keys": case.expected_json_keys,
                "validator": case.validator,
                "loops": per_loop,
                "regressions": regressions,
            }
        )

    for loop in loops:
        n = max(1, loop_counts[loop])
        metrics_by_loop[str(loop)] = {
            "count": n,
            "accuracy": loop_accum[loop]["accuracy"] / n,
            "format_valid_rate": loop_accum[loop]["format_valid"] / n,
            "repeat_4gram_rate": loop_accum[loop]["repeat_4gram_rate"] / n,
            "entropy_proxy": loop_accum[loop]["entropy_proxy"] / n,
            "max_token_hit_rate": loop_accum[loop]["max_token_hit"] / n,
        }

    regression_rows = []
    for rec in records:
        for reg in rec["regressions"]:
            regression_rows.append(
                {
                    "task_type": rec["task_type"],
                    "prompt": rec["prompt"],
                    "type": reg["type"],
                    "details": reg["details"],
                    "evidence": reg["evidence"],
                }
            )

    clusters = cluster_regressions(regression_rows)
    top_examples = sorted(
        regression_rows,
        key=lambda r: (severity_rank(r["type"]), -len(r["prompt"])),
        reverse=True,
    )[:10]

    result = {
        "experiment": "E5_loop_regression_hunt",
        "backend": backend.name,
        "model": args.model,
        "loops": loops,
        "max_new_tokens": args.max_new_tokens,
        "cases": len(records),
        "metrics_by_loop": metrics_by_loop,
        "regression_count": len(regression_rows),
        "regression_clusters": clusters,
        "top_regression_examples": top_examples,
        "records": records,
    }
    return result


def severity_rank(reg_type: str) -> int:
    order = {
        "correct@1_wrong@4": 3,
        "format_valid@2_invalid@4": 2,
        "accuracy_drop": 1,
    }
    return order.get(reg_type, 0)


def detect_regressions(case: PromptCase, per_loop: Dict[int, Dict[str, Any]], loops: List[int]) -> List[Dict[str, Any]]:
    regs: List[Dict[str, Any]] = []
    if 1 in per_loop and 4 in per_loop:
        if per_loop[1]["correct"] and not per_loop[4]["correct"]:
            regs.append(
                {
                    "type": "correct@1_wrong@4",
                    "details": "Case was correct at loop 1 but incorrect at loop 4.",
                    "evidence": {"loop1": per_loop[1]["output"], "loop4": per_loop[4]["output"]},
                }
            )
    if 2 in per_loop and 4 in per_loop:
        if per_loop[2]["format_valid"] and not per_loop[4]["format_valid"]:
            regs.append(
                {
                    "type": "format_valid@2_invalid@4",
                    "details": "Output format was valid at loop 2 but invalid at loop 4.",
                    "evidence": {"loop2": per_loop[2]["output"], "loop4": per_loop[4]["output"]},
                }
            )
    # Generic accuracy drop: any later loop worse than best early loop.
    correct_series = [(lp, 1 if per_loop[lp]["correct"] else 0) for lp in loops if lp in per_loop]
    if correct_series:
        best_early = max(v for lp, v in correct_series if lp <= min(2, max(loops)))
        worst_late = min(v for lp, v in correct_series if lp >= min(4, max(loops))) if max(loops) >= 2 else best_early
        if best_early > worst_late:
            regs.append(
                {
                    "type": "accuracy_drop",
                    "details": f"Accuracy dropped from early best {best_early} to late {worst_late}.",
                    "evidence": {"correct_by_loop": {str(lp): v for lp, v in correct_series}},
                }
            )
    return regs


def cluster_regressions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_type: Dict[str, int] = defaultdict(int)
    for row in rows:
        by_task[row["task_type"]].append(row)
        by_type[row["type"]] += 1
    clusters = {}
    for task, items in sorted(by_task.items()):
        clusters[task] = {
            "count": len(items),
            "types": dict(Counter(i["type"] for i in items)),
            "examples": items[:3],
        }
    return {"by_task_type": clusters, "by_regression_type": dict(by_type)}


def render_markdown(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# E5 Loop Regression Hunt")
    lines.append("")
    lines.append(f"- Backend: `{result['backend']}`")
    lines.append(f"- Model: `{result.get('model')}`")
    lines.append(f"- Loops: `{result['loops']}`")
    lines.append(f"- Cases: `{result['cases']}`")
    lines.append(f"- Regressions: `{result['regression_count']}`")
    lines.append("")
    lines.append("## Metrics by loop")
    lines.append("")
    for loop, m in result["metrics_by_loop"].items():
        lines.append(
            f"- Loop {loop}: accuracy={m['accuracy']:.3f}, format_valid={m['format_valid_rate']:.3f}, "
            f"repeat_4gram={m['repeat_4gram_rate']:.3f}, entropy={m['entropy_proxy']:.3f}, "
            f"max_token_hit={m['max_token_hit_rate']:.3f}"
        )
    lines.append("")
    lines.append("## Regression taxonomy")
    lines.append("")
    by_type = result["regression_clusters"].get("by_regression_type", {})
    if not by_type:
        lines.append("- No regressions detected.")
    else:
        for k, v in sorted(by_type.items()):
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Clusters by task type")
    lines.append("")
    by_task = result["regression_clusters"].get("by_task_type", {})
    if not by_task:
        lines.append("- No task clusters.")
    else:
        for task, info in by_task.items():
            lines.append(f"- {task}: count={info['count']}, types={info['types']}")
    lines.append("")
    lines.append("## Top regression examples")
    lines.append("")
    for i, ex in enumerate(result.get("top_regression_examples", []), 1):
        lines.append(f"### Example {i}: {ex['type']} ({ex['task_type']})")
        lines.append("")
        lines.append(f"- Prompt: {ex['prompt']}")
        lines.append(f"- Details: {ex['details']}")
        lines.append(f"- Evidence: `{json.dumps(ex['evidence'], ensure_ascii=False)[:500]}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_outputs(result: Dict[str, Any], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "e5_loop_regression.json")
    md_path = os.path.join(out_dir, "e5_loop_regression.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
    return json_path, md_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E5: Loop Regression Hunt")
    parser.add_argument("--out", default="e5_loop_regression_output", help="Output directory for JSON and markdown summary")
    parser.add_argument("--loops", type=int, default=4, help="Maximum loop count to test (starts at 1)")
    parser.add_argument("--limit", type=int, default=0, help="Limit cases per prompt set (0 means all)")
    parser.add_argument("--model", default=None, help="Transformers model name/path; if omitted uses mock backend")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens for generation")
    args = parser.parse_args(argv)
    if args.loops < 1:
        parser.error("--loops must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_experiment(args)
    json_path, md_path = save_outputs(result, args.out)
    print(f"Backend: {result['backend']}")
    print(f"Cases: {result['cases']} Loops: {result['loops']} Regressions: {result['regression_count']}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
