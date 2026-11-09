"""E2: 256K context feasibility on 2xT4 for Dory (vLLM backend)."""
import argparse, gc, json, os, random, sys, time, subprocess
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
LADDER = [8192, 16384, 32768, 65536, 131072, 262144]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
FILLER = "the forest river stone cloud light path "

def ensure_vllm():
    try:
        import vllm; return
    except Exception:
        os.environ["VLLM_USE_PRECOMPILED"]="1"
        subprocess.check_call([sys.executable,"-m","pip","install","-q","git+https://github.com/vllm-project/vllm.git@refs/pull/60149/head"])

def load_llm(loops, max_model_len):
    ensure_vllm()
    from vllm import LLM
    return LLM(model=MODEL_ID, dtype="auto",
               hf_overrides={"n_recurrent_loops": int(loops),
                             "recurrent_kv_cache_mode": "last_loop"},
               max_model_len=int(max_model_len), gpu_memory_utilization=0.90,
               trust_remote_code=True)

def build_niah(tok, ctx, depth, secret):
    # vLLM uses its own tokenizer; approximate with char filler by tokens via simple repeat
    needle = f"\nThe secret code is {secret}.\n"
    q = "\n\nQuestion: What is the secret code?\nAnswer:"
    budget = max(32, ctx - 64)
    hay = (FILLER * (budget // len(FILLER) + 1))[:budget]
    ins = min(len(hay), int(len(hay)*depth))
    return hay[:ins] + needle + hay[ins:] + q

def run_gen(llm, prompt, max_tokens=16):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    t0 = time.time()
    out = llm.generate([prompt], sp)
    dt = time.time() - t0
    text = out[0].outputs[0].text
    ntok = len(out[0].outputs[0].token_ids)
    # prompt token count approximated by vLLM? use output prompt_token_ids if available
    ptok = len(getattr(out[0], "prompt_token_ids", []) or [])
    return text, ntok, dt, ptok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e2")
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--gen-tokens", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    records = []
    feasible = []

    # Phase 1: context ladder (prefill+short gen)
    print("Phase1 context ladder", flush=True)
    for ctx in [x for x in LADDER if x <= args.max_len]:
        print(f"probe ctx={ctx}", flush=True)
        try:
            llm = load_llm(1, ctx)
            prompt = FILLER * (ctx // len(FILLER))
            text, ntok, dt, ptok = run_gen(llm, prompt, 4)
            records.append({"phase":"ladder","context_len":ctx,"ok":True,"gen_tokens":ntok,"dt":dt})
            feasible.append(ctx)
            print(f"  OK gen={ntok} dt={dt:.2f}s", flush=True)
            del llm; gc.collect()
        except Exception as e:
            print(f"  FAIL: {str(e).splitlines()[0][:200]}", flush=True)
            records.append({"phase":"ladder","context_len":ctx,"ok":False,"error":str(e)[:500]})
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print("OOM boundary; stopping", flush=True); break
            try: del llm
            except Exception: pass
            gc.collect()

    # Phase 2: NIAH at feasible lengths
    print("Phase2 NIAH", flush=True)
    for loops in sorted(set([1, args.loops])):
        print(f"-- loops={loops} --", flush=True)
        for ctx in feasible:
            for depth in DEPTHS:
                secret = f"DORY-{random.randint(100000,999999)}"
                prompt = build_niah(None, ctx, depth, secret)
                try:
                    llm = load_llm(loops, ctx)
                    text, ntok, dt, ptok = run_gen(llm, prompt, args.gen_tokens)
                    ok = secret.lower() in text.lower()
                    records.append({"phase":"niah","loops":loops,"context_len":ctx,"depth":depth,
                                    "secret":secret,"generated":text[:120],"recall_correct":ok,"dt":dt})
                    print(f"  loops={loops} ctx={ctx} depth={depth:.2f} recall={ok}", flush=True)
                    del llm; gc.collect()
                except Exception as e:
                    print(f"  FAIL loops={loops} ctx={ctx} depth={depth:.2f}: {str(e).splitlines()[0][:150]}", flush=True)
                    records.append({"phase":"niah","loops":loops,"context_len":ctx,"depth":depth,"ok":False,"error":str(e)[:500]})
                    if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                        break

    payload = {"meta":{"model":MODEL_ID,"max_len":args.max_len,"loops_tested":sorted(set([1,args.loops]))},
               "feasible_lengths":feasible, "results":records}
    with open(args.out+"/results.json" if False else str(Path(args.out)/"results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {Path(args.out)/'results.json'}; feasible={feasible}")

if __name__ == "__main__":
    main()
