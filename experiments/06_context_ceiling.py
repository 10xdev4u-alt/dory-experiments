#!/usr/bin/env python3
"""E6: Context ceiling experiment (synthetic NIAH).

Runs vLLM with last_loop/per_loop modes at varying context lengths and loop
counts, measures NIAH retrieval accuracy, captures OOM/errors, and runs a
memory diagnostic (loop1 vs loop4 at 64K).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_vllm() -> None:
    """Ensure vLLM is importable in the current environment.

    This is a best-effort helper for experiment runners that may execute in a
    prepared virtualenv. It does not install packages automatically unless the
    environment explicitly requests it via E6_AUTO_INSTALL_VLLM=1.
    """
    try:
        import vllm  # noqa: F401

        return
    except Exception as exc:  # pragma: no cover - environment dependent
        if os.environ.get("E6_AUTO_INSTALL_VLLM") == "1":
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm"])
            import vllm  # noqa: F401

            return
        raise RuntimeError(
            "vLLM is not importable. Activate the experiment environment or set "
            "E6_AUTO_INSTALL_VLLM=1."
        ) from exc


@dataclass
class RunConfig:
    context_tokens: int
    mode: str
    loops: int
    model: str
    gen_tokens: int
    depths: list[float] = field(default_factory=list)
    needles_per_prompt: int = 1
    seed: int = 42


@dataclass
class NIAHCase:
    context_tokens: int
    depth_pct: float
    needle_index: int
    secret: str
    prompt: str
    question: str
    expected: str


@dataclass
class CaseResult:
    context_tokens: int
    depth_pct: float
    needle_index: int
    secret: str
    expected: str
    generated: str
    correct: bool
    elapsed_s: float
    error: str | None = None


@dataclass
class ExperimentResult:
    config: dict[str, Any]
    status: str
    cases: list[dict[str, Any]] = field(default_factory=list)
    accuracy: float | None = None
    n_correct: int = 0
    n_total: int = 0
    error: str | None = None
    traceback: str | None = None
    elapsed_s: float = 0.0
    memory: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6 context ceiling / NIAH experiment")
    parser.add_argument(
        "--out",
        default="results/e6_context_ceiling.json",
        help="Output JSON path (markdown written alongside).",
    )
    parser.add_argument(
        "--contexts",
        default="32768,65536,131072",
        help="Comma-separated context lengths. Add 262144 for optional probe.",
    )
    parser.add_argument(
        "--loops",
        default="1,4",
        help="Comma-separated loop counts.",
    )
    parser.add_argument(
        "--modes",
        default="last_loop,per_loop",
        help="Comma-separated output modes.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=64,
        dest="gen_tokens",
        help="Max generation tokens for NIAH answers.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("E6_MODEL", os.environ.get("MODEL", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")),
        help="Model path/name for vLLM.",
    )
    parser.add_argument(
        "--depths",
        default="0,25,50,75,90,99",
        help="Comma-separated needle depths in percent.",
    )
    parser.add_argument(
        "--needles",
        type=int,
        default=1,
        help="Needles per prompt (1-3).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-262144-probe",
        action="store_true",
        help="Append 262144 context probe if not already present.",
    )
    parser.add_argument(
        "--skip-memory-diagnostic",
        action="store_true",
        help="Skip loop1 vs loop4 memory diagnostic at 64K.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        dest="gpu_memory_utilization",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        dest="max_num_batched_tokens",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        dest="max_num_seqs",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def make_haystack_text(target_chars: int, rng: random.Random) -> str:
    # Deterministic, low-information filler that tokenizes reasonably.
    sentences = [
        "The weather in the valley changes slowly across the seasons.",
        "Engineers recorded measurements and compared them with previous reports.",
        "A small library contained maps, journals, and technical manuals.",
        "The river curved around the hills before reaching the open plain.",
        "Researchers discussed methods, data quality, and experimental design.",
        "The catalog listed items by date, location, and reference number.",
        "Morning light entered the room through a narrow window.",
        "The team reviewed notes and prepared a summary for the archive.",
    ]
    parts: list[str] = []
    total = 0
    while total < target_chars:
        s = sentences[rng.randrange(len(sentences))]
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)[:target_chars]


def tokenize_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def trim_to_token_count(tokenizer: Any, text: str, target_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= target_tokens:
        return text
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_niah_prompt(
    tokenizer: Any,
    context_tokens: int,
    depth_pct: float,
    needle_index: int,
    needles_per_prompt: int,
    rng: random.Random,
) -> NIAHCase:
    # Keep answer short and unambiguous for exact/substring matching.
    secret = f"DORY-SECRET-{needle_index}-{rng.randrange(100000, 999999)}"
    needle = (
        f"\n[IMPORTANT FACT {needle_index}] The secret passcode is {secret}. "
        f"Remember this passcode exactly.\n"
    )
    question = "What is the secret passcode stated in the important fact? Answer with only the passcode."
    suffix = f"\n\nQuestion: {question}\nAnswer:"

    # Reserve space for needles/question; build haystack to fill remainder.
    overhead = tokenize_len(tokenizer, needle * max(1, needles_per_prompt) + suffix) + 32
    target_haystack = max(64, context_tokens - overhead)
    # Approx chars-to-tokens heuristic, then trim precisely.
    approx_chars = target_haystack * 4
    haystack = make_haystack_text(approx_chars, rng)
    haystack = trim_to_token_count(tokenizer, haystack, target_haystack)
    hay_ids = tokenizer.encode(haystack, add_special_tokens=False)
    if not hay_ids:
        hay_ids = tokenizer.encode("filler " * target_haystack, add_special_tokens=False)[
            :target_haystack
        ]

    depth = max(0.0, min(100.0, float(depth_pct))) / 100.0
    insert_pos = int(len(hay_ids) * depth)
    insert_pos = max(0, min(len(hay_ids), insert_pos))

    # Insert selected needle(s). For multi-needle prompts, place the queried
    # needle at the requested depth and spread others deterministically.
    needles: list[tuple[int, str, str]] = []
    for i in range(1, max(1, needles_per_prompt) + 1):
        if i == needle_index:
            s = secret
            n = needle
            pos = insert_pos
        else:
            other_secret = f"DORY-DECOY-{i}-{rng.randrange(100000, 999999)}"
            n = (
                f"\n[IMPORTANT FACT {i}] The secret passcode is {other_secret}. "
                f"Remember this passcode exactly.\n"
            )
            s = other_secret
            spread = (i / (max(1, needles_per_prompt) + 1))
            pos = int(len(hay_ids) * spread)
        needles.append((pos, n, s))

    # Apply insertions from right to left to keep positions stable.
    combined_ids = list(hay_ids)
    for pos, n, _ in sorted(needles, key=lambda x: x[0], reverse=True):
        n_ids = tokenizer.encode(n, add_special_tokens=False)
        combined_ids[pos:pos] = n_ids

    prompt_ids = combined_ids + tokenizer.encode(suffix, add_special_tokens=False)
    # Trim or pad minimally to get close to requested context. Padding uses filler.
    if len(prompt_ids) > context_tokens:
        # Keep suffix/question at the end.
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        keep = max(0, context_tokens - len(suffix_ids))
        prompt_ids = prompt_ids[:keep] + suffix_ids
    elif len(prompt_ids) < context_tokens:
        pad_id = tokenizer.encode(" filler", add_special_tokens=False) or [tokenizer.eos_token_id or 0]
        need = context_tokens - len(prompt_ids)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        body = prompt_ids[:-len(suffix_ids)] if suffix_ids else prompt_ids
        pad = (pad_id * ((need + len(pad_id) - 1) // len(pad_id)))[:need]
        prompt_ids = body + pad + suffix_ids

    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    return NIAHCase(
        context_tokens=context_tokens,
        depth_pct=depth_pct,
        needle_index=needle_index,
        secret=secret,
        prompt=prompt,
        question=question,
        expected=secret,
    )


def check_accuracy(expected: str, generated: str) -> bool:
    if not generated:
        return False
    return expected.strip() in generated.strip()


def get_torch_memory_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        torch.cuda.synchronize()
        return {
            "cuda_available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception as exc:  # pragma: no cover
        return {"cuda_available": False, "error": str(exc)}


def reset_torch_memory_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def make_llm(
    *,
    model: str,
    context_tokens: int,
    mode: str,
    loops: int,
    args: argparse.Namespace,
) -> Any:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "max_model_len": int(context_tokens),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enforce_eager": True,
        "max_num_seqs": int(args.max_num_seqs),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "enable_chunked_prefill": True,
        "trust_remote_code": True,
    }

    # Dory-specific knobs vary by fork. Try a few common names defensively.
    loop_kwargs_candidates = [
        {"num_loops": int(loops), "loop_output_mode": mode},
        {"dory_num_loops": int(loops), "dory_loop_output_mode": mode},
        {"loop_count": int(loops), "loop_mode": mode},
        {"num_loops": int(loops)},
    ]

    last_error: Exception | None = None
    for extra in loop_kwargs_candidates:
        try:
            return LLM(**llm_kwargs, **extra)
        except TypeError as exc:
            last_error = exc
            continue
    # Fallback: create without loop knobs and record behavior in output.
    if last_error is not None:
        print(f"[warn] Could not pass loop/mode kwargs directly: {last_error}", file=sys.stderr)
    return LLM(**llm_kwargs)


def generate_answers(llm: Any, prompts: list[str], gen_tokens: int, seed: int) -> list[str]:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=int(gen_tokens), seed=int(seed))
    outputs = llm.generate(prompts, params)
    answers: list[str] = []
    for out in outputs:
        if not out.outputs:
            answers.append("")
            continue
        answers.append(out.outputs[0].text or "")
    return answers


def run_niah_condition(
    *,
    model: str,
    context_tokens: int,
    mode: str,
    loops: int,
    gen_tokens: int,
    depths: list[float],
    needles_per_prompt: int,
    seed: int,
    args: argparse.Namespace,
) -> ExperimentResult:
    cfg = RunConfig(
        context_tokens=context_tokens,
        mode=mode,
        loops=loops,
        model=model,
        gen_tokens=gen_tokens,
        depths=depths,
        needles_per_prompt=needles_per_prompt,
        seed=seed,
    )
    result = ExperimentResult(config=asdict(cfg), status="ok")
    start = time.time()
    llm = None
    try:
        ensure_vllm()
        rng = random.Random(seed + context_tokens + loops + (0 if mode == "last_loop" else 1))
        llm = make_llm(model=model, context_tokens=context_tokens, mode=mode, loops=loops, args=args)
        tokenizer = llm.get_tokenizer()

        cases: list[NIAHCase] = []
        for i, depth in enumerate(depths, start=1):
            needle_index = ((i - 1) % max(1, min(3, needles_per_prompt))) + 1
            cases.append(
                build_niah_prompt(
                    tokenizer=tokenizer,
                    context_tokens=context_tokens,
                    depth_pct=depth,
                    needle_index=needle_index,
                    needles_per_prompt=max(1, min(3, needles_per_prompt)),
                    rng=rng,
                )
            )

        prompts = [c.prompt for c in cases]
        t0 = time.time()
        answers = generate_answers(llm, prompts, gen_tokens=gen_tokens, seed=seed)
        elapsed = time.time() - t0

        case_results: list[CaseResult] = []
        correct = 0
        for case, ans in zip(cases, answers):
            ok = check_accuracy(case.expected, ans)
            correct += int(ok)
            case_results.append(
                CaseResult(
                    context_tokens=case.context_tokens,
                    depth_pct=case.depth_pct,
                    needle_index=case.needle_index,
                    secret=case.secret,
                    expected=case.expected,
                    generated=ans,
                    correct=ok,
                    elapsed_s=elapsed / max(1, len(cases)),
                )
            )

        result.cases = [asdict(c) for c in case_results]
        result.n_correct = correct
        result.n_total = len(case_results)
        result.accuracy = (correct / len(case_results)) if case_results else None
        result.memory = get_torch_memory_snapshot()
        result.status = "ok"
    except Exception as exc:
        msg = str(exc)
        lower = msg.lower()
        if "out of memory" in lower or "oom" in lower or "cuda" in lower and "memory" in lower:
            result.status = "oom"
        else:
            result.status = "error"
        result.error = msg
        result.traceback = traceback.format_exc()
        result.memory = get_torch_memory_snapshot()
    finally:
        try:
            if llm is not None:
                del llm
        except Exception:
            pass
        gc.collect()
        reset_torch_memory_stats()
        result.elapsed_s = time.time() - start
    return result


def run_memory_diagnostic(*, model: str, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    """Compare loop1 vs loop4 memory at 64K for the selected mode."""
    diag: dict[str, Any] = {"context_tokens": 65536, "mode": mode, "runs": []}
    for loops in (1, 4):
        reset_torch_memory_stats()
        entry: dict[str, Any] = {"loops": loops}
        llm = None
        try:
            llm = make_llm(model=model, context_tokens=65536, mode=mode, loops=loops, args=args)
            # Minimal generation to materialize execution state.
            _ = generate_answers(llm, ["Say OK."], gen_tokens=4, seed=args.seed)
            entry["status"] = "ok"
            entry["memory"] = get_torch_memory_snapshot()
        except Exception as exc:
            msg = str(exc)
            lower = msg.lower()
            entry["status"] = "oom" if ("out of memory" in lower or "oom" in lower) else "error"
            entry["error"] = msg
            entry["memory"] = get_torch_memory_snapshot()
        finally:
            try:
                if llm is not None:
                    del llm
            except Exception:
                pass
            gc.collect()
            reset_torch_memory_stats()
        diag["runs"].append(entry)
    return diag


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# E6 Context Ceiling (NIAH)")
    lines.append("")
    lines.append(f"- Generated: `{payload.get('generated_at', '')}`")
    lines.append(f"- Model: `{payload.get('model', '')}`")
    lines.append(f"- Contexts: `{payload.get('contexts', [])}`")
    lines.append(f"- Modes: `{payload.get('modes', [])}`")
    lines.append(f"- Loops: `{payload.get('loops', [])}`")
    lines.append(f"- Gen tokens: `{payload.get('gen_tokens', '')}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| context | mode | loops | status | accuracy | correct/total | elapsed_s |")
    lines.append("|---:|---|---:|---|---:|---:|---:|")
    for r in payload.get("results", []):
        cfg = r.get("config", {})
        acc = r.get("accuracy")
        acc_s = "" if acc is None else f"{acc:.3f}"
        lines.append(
            "| {context} | {mode} | {loops} | {status} | {acc} | {c}/{t} | {elapsed:.2f} |".format(
                context=cfg.get("context_tokens", ""),
                mode=cfg.get("mode", ""),
                loops=cfg.get("loops", ""),
                status=r.get("status", ""),
                acc=acc_s,
                c=r.get("n_correct", 0),
                t=r.get("n_total", 0),
                elapsed=float(r.get("elapsed_s", 0.0) or 0.0),
            )
        )
        if r.get("error"):
            lines.append("")
            lines.append(f"<details><summary>Error: context={cfg.get('context_tokens')} mode={cfg.get('mode')} loops={cfg.get('loops')}</summary>")
            lines.append("")
            lines.append("```text")
            lines.append(str(r.get("error")))
            lines.append("```")
            lines.append("</details>")
            lines.append("")
    lines.append("")
    lines.append("## Memory Diagnostic (64K)")
    lines.append("")
    diag = payload.get("memory_diagnostic")
    if not diag:
        lines.append("Skipped.")
    else:
        lines.append(f"- Mode: `{diag.get('mode', '')}`")
        lines.append("")
        lines.append("| loops | status | allocated | reserved | max_allocated | max_reserved |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for run in diag.get("runs", []):
            mem = run.get("memory", {}) or {}
            lines.append(
                "| {loops} | {status} | {a} | {r} | {ma} | {mr} |".format(
                    loops=run.get("loops", ""),
                    status=run.get("status", ""),
                    a=mem.get("allocated_bytes", ""),
                    r=mem.get("reserved_bytes", ""),
                    ma=mem.get("max_allocated_bytes", ""),
                    mr=mem.get("max_reserved_bytes", ""),
                )
            )
            if run.get("error"):
                lines.append("")
                lines.append(f"<details><summary>Diagnostic error (loops={run.get('loops')})</summary>")
                lines.append("")
                lines.append("```text")
                lines.append(str(run.get("error")))
                lines.append("```")
                lines.append("</details>")
                lines.append("")
    lines.append("")
    return "\n".join(lines)


def save_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")


def main() -> int:
    args = parse_args()
    contexts = parse_csv_ints(args.contexts)
    if args.include_262144_probe and 262144 not in contexts:
        contexts.append(262144)
    loops = parse_csv_ints(args.loops)
    modes = parse_csv_strs(args.modes)
    depths = parse_csv_floats(args.depths)
    needles_per_prompt = max(1, min(3, int(args.needles)))

    if not args.model:
        print("--model is required (or set E6_MODEL/MODEL).", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "experiment": "E6_context_ceiling",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "contexts": contexts,
        "loops": loops,
        "modes": modes,
        "depths": depths,
        "gen_tokens": args.gen_tokens,
        "needles_per_prompt": needles_per_prompt,
        "seed": args.seed,
        "vllm_config": {
            "max_model_len": "target",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_chunked_prefill": True,
        },
        "results": [],
        "memory_diagnostic": None,
    }

    out_json = Path(args.out)
    out_md = out_json.with_suffix(".md")

    # Run memory diagnostic once (loop1 vs loop4 at 64K) unless skipped.
    if not args.skip_memory_diagnostic and 65536 in contexts and modes:
        try:
            ensure_vllm()
            payload["memory_diagnostic"] = run_memory_diagnostic(
                model=args.model,
                mode=modes[0],
                args=args,
            )
        except Exception as exc:
            payload["memory_diagnostic"] = {
                "context_tokens": 65536,
                "mode": modes[0],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        save_json(out_json, payload)
        save_markdown(out_md, payload)

    for context_tokens in contexts:
        for mode in modes:
            for loop_count in loops:
                print(
                    f"[E6] context={context_tokens} mode={mode} loops={loop_count}",
                    flush=True,
                )
                res = run_niah_condition(
                    model=args.model,
                    context_tokens=context_tokens,
                    mode=mode,
                    loops=loop_count,
                    gen_tokens=args.gen_tokens,
                    depths=depths,
                    needles_per_prompt=needles_per_prompt,
                    seed=args.seed,
                    args=args,
                )
                payload["results"].append(asdict(res))
                save_json(out_json, payload)
                save_markdown(out_md, payload)

    save_json(out_json, payload)
    save_markdown(out_md, payload)
    print(f"[E6] wrote {out_json}")
    print(f"[E6] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
