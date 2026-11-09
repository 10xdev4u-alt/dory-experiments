"""E1: Loop-count scaling law + exchange rate (vLLM backend)."""
import argparse, json, os, time, gc, sys, subprocess
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

def ensure_vllm():
    try:
        import vllm
        return
    except Exception:
        os.environ["VLLM_USE_PRECOMPILED"] = "1"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head"])

def load_llm(loops, max_model_len=4096):
    ensure_vllm()
    from vllm import LLM
    return LLM(model=MODEL_ID, dtype="auto",
               hf_overrides={"n_recurrent_loops": int(loops),
                             "recurrent_kv_cache_mode": "last_loop"},
               max_model_len=max_model_len, gpu_memory_utilization=0.85,
               trust_remote_code=True)

def gen(llm, prompt, max_new_tokens=256):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    t0 = time.time()
    out = llm.generate([prompt], sp)
    dt = time.time() - t0
    text = out[0].outputs[0].text
    ntok = len(out[0].outputs[0].token_ids)
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
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    suite = []
    for q,a in [("What is 17*23?", "391"), ("Solve 2x+5=13", "4"), ("100/4+7", "32")]:
        suite.append({"task":"math","prompt":f"Question: {q}\nAnswer:", "gold":a})
    for q,a in [("Capital of France?", "Paris"), ("Largest planet?", "Jupiter")]:
        suite.append({"task":"trivia","prompt":f"Q: {q}\nA:", "gold":a})
    suite.append({"task":"code","prompt":"# python: return sum of list\ndef s(x):", "gold":"return sum(x)"})
    suite = (suite * (args.limit // len(suite) + 1))[:args.limit]

    all_rows = []
    for L in args.loops:
        print(f"=== loops={L} ===", flush=True)
        llm = load_llm(L)
        rows = []
        for i, ex in enumerate(suite):
            try:
                text, ntok, dt = gen(llm, ex["prompt"], args.max_tokens)
            except Exception as e:
                print(f"  ERR at i={i}: {str(e).splitlines()[0][:120]}", flush=True)
                text, ntok, dt = "", 0, 0.0
            ok = ex["gold"].lower() in text.lower() if text else False
            rows.append({"i":i, "task":ex["task"], "loops":L, "ok":ok,
                        "ntok":ntok, "dt":dt, "toks":(ntok/max(dt,1e-6)) if dt>0 else 0.0,
                        "rep":rep_rate(text), "out":text[:200]})
            if i % 10 == 0:
                print(f"  {i}/{len(suite)} ok={ok} toks={ntok} peak? ", flush=True)
        (out / f"loops_{L}.json").write_text(json.dumps(rows, indent=2))
        all_rows.extend(rows)
        del llm; gc.collect()

    by_i = {}
    for r in all_rows:
        by_i.setdefault(r["i"], []).append(r)
    oracle = []
    for i, rs in by_i.items():
        best = max(rs, key=lambda r: (r["ok"], -r["loops"]))
        oracle.append({"i":i, "best_loops":best["loops"], "ok":best["ok"]})
    (out / "oracle.json").write_text(json.dumps(oracle, indent=2))

    summ = {}
    for L in args.loops:
        rs = [r for r in all_rows if r["loops"]==L]
        summ[L] = {
            "n": len(rs),
            "acc": sum(r["ok"] for r in rs)/len(rs) if rs else 0.0,
            "avg_tok": sum(r["ntok"] for r in rs)/len(rs) if rs else 0.0,
            "avg_tps": sum(r["toks"] for r in rs)/len(rs) if rs else 0.0,
            "avg_rep": sum(r["rep"] for r in rs)/len(rs) if rs else 0.0,
        }
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
