"""E1: Loop-count scaling law + exchange rate.

Runs Dory at loops in {1,2,3,4} on a mixed benchmark suite, records accuracy,
tok/s, peak VRAM, repetition rate, and oracle best-loop per prompt (labels for E4).

Kaggle: 2x T4. Run on a single GPU for memory headroom; use device_map='auto'
only if needed. Set N_LOOPS via env or edit LOOPS list.

Usage:
  python experiments/01_loop_ladder.py --out results/e1 --loops 1 2 3 4 --max-tokens 256
"""
import argparse, json, os, time, gc, sys
from pathlib import Path
import torch

MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

def load_model(loops, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": device},
        trust_remote_code=True, low_cpu_mem_usage=True)
    # set loop count
    if hasattr(model.config, "n_recurrent_loops"):
        model.config.n_recurrent_loops = loops
    if hasattr(model, "set_num_loops"):
        model.set_num_loops(loops)
    model.eval()
    return model, tok

def peak_vram():
    return torch.cuda.max_memory_allocated() / 1e9

def gen(model, tok, prompt, max_new_tokens=256, loops=None):
    if loops is not None:
        if hasattr(model.config, "n_recurrent_loops"):
            model.config.n_recurrent_loops = loops
        if hasattr(model, "set_num_loops"):
            model.set_num_loops(loops)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    ntok = out.shape[1] - inputs.input_ids.shape[1]
    return text, ntok, dt

def rep_rate(text):
    toks = text.split()
    if len(toks) < 8: return 0.0
    grams = [tuple(toks[i:i+4]) for i in range(len(toks)-3)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--loops", nargs="+", type=int, default=[1,2,3,4])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=50, help="per-dataset cap for lap speed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # Tiny mixed suite (self-contained, no network needed beyond model download)
    suite = []
    # math
    for q,a in [("What is 17*23?", "391"), ("Solve 2x+5=13", "4"), ("100/4+7", "32")]:
        suite.append({"task":"math","prompt":f"Question: {q}\nAnswer:", "gold":a})
    # trivia
    for q,a in [("Capital of France?", "Paris"), ("Largest planet?", "Jupiter")]:
        suite.append({"task":"trivia","prompt":f"Q: {q}\nA:", "gold":a})
    # code
    suite.append({"task":"code","prompt":"# python: return sum of list\ndef s(x):", "gold":"return sum(x)"})

    # expand with repeats for timing stability
    suite = (suite * (args.limit // len(suite) + 1))[:args.limit]

    all_rows = []
    for L in args.loops:
        print(f"=== loops={L} ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        model, tok = load_model(L)
        rows = []
        for i, ex in enumerate(suite):
            try:
                text, ntok, dt = gen(model, tok, ex["prompt"], args.max_tokens, L)
            except RuntimeError as e:
                print(f"  OOM/ERR at i={i}: {str(e).splitlines()[0][:120]}", flush=True)
                gc.collect(); torch.cuda.empty_cache()
                text, ntok, dt = "", 0, 0.0
            ok = ex["gold"].lower() in text.lower() if text else False
            rows.append({"i":i, "task":ex["task"], "loops":L, "ok":ok,
                        "ntok":ntok, "dt":dt, "toks":(ntok/max(dt,1e-6)) if dt>0 else 0.0,
                        "peak_gb":peak_vram(), "rep":rep_rate(text),
                        "out":text[:200], "error": None if text else "oom_or_error"})
            if i % 10 == 0:
                print(f"  {i}/{len(suite)} ok={ok} toks={ntok} peak={peak_vram():.2f}GB", flush=True)
        (out / f"loops_{L}.json").write_text(json.dumps(rows, indent=2))
        all_rows.extend(rows)
        del model; gc.collect(); torch.cuda.empty_cache()

    # oracle labels
    by_i = {}
    for r in all_rows:
        by_i.setdefault(r["i"], []).append(r)
    oracle = []
    for i, rs in by_i.items():
        best = max(rs, key=lambda r: (r["ok"], -r["loops"]))
        oracle.append({"i":i, "best_loops":best["loops"], "ok":best["ok"]})
    (out / "oracle.json").write_text(json.dumps(oracle, indent=2))

    # summary
    summ = {}
    for L in args.loops:
        rs = [r for r in all_rows if r["loops"]==L]
        summ[L] = {
            "n": len(rs),
            "acc": sum(r["ok"] for r in rs)/len(rs),
            "avg_tok": sum(r["ntok"] for r in rs)/len(rs),
            "avg_tps": sum(r["toks"] for r in rs)/len(rs),
            "peak_gb": max(r["peak_gb"] for r in rs),
            "avg_rep": sum(r["rep"] for r in rs)/len(rs),
        }
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
