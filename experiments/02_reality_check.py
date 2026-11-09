#!/usr/bin/env python3
"""E2: 256K context feasibility on 2xT4 for Dory."""
# Full implementation: see team subagent draft (Iceman). Reproduced below.
import argparse, gc, json, math, os, random, sys, time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
# Kaggle transformers (>=4.5x) rejects Dory's 'mlp' entries in layer_types; allow them.
try:
    import transformers.configuration_utils as _cu
    if "mlp" not in getattr(_cu, "ALLOWED_ATTN_LAYER_TYPES", ()):
        _cu.ALLOWED_ATTN_LAYER_TYPES = _cu.ALLOWED_ATTN_LAYER_TYPES + ("mlp",)
except Exception:
    pass
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

LADDER = [8192, 16384, 32768, 65536, 131072, 262144]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
FILLER = "the forest river stone cloud light path "
QUESTION = "\n\nQuestion: What is the secret code?\nAnswer:"

@dataclass
class RunRecord:
    phase: str; context_len: int; loops: Optional[int]=None; depth: Optional[float]=None
    ok: bool=True; oom: bool=False; error: Optional[str]=None
    peak_vram_gb: Optional[float]=None; prefill_tok_s: Optional[float]=None
    prefill_s: Optional[float]=None; ttft_s: Optional[float]=None
    recall_correct: Optional[bool]=None; generated: Optional[str]=None; secret: Optional[str]=None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="experiments/e2_reality_check.json")
    p.add_argument("--max-len", type=int, default=262144)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-tokens", type=int, default=16)
    return p.parse_args()

def bytes_to_gb(b): return b/(1024**3)
def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
def is_oom(e):
    m=str(e).lower(); return isinstance(e,torch.cuda.OutOfMemoryError) or "out of memory" in m or "cuda oom" in m

def load_model(model_id, loops, device):
    dtype=torch.float16
    config=AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if loops is not None and hasattr(config,"n_recurrent_loops"):
        config.n_recurrent_loops=int(loops)
    tok=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device); model.eval()
    return model, tok, config, dtype

def print_budget(model, config, dtype):
    param_bytes=sum(p.numel()*p.element_size() for p in model.parameters())+sum(b.numel()*b.element_size() for b in model.buffers())
    print(f"Weights: {param_bytes/1e9:.3f} GB ({dtype})")
    n_layers=getattr(config,"num_hidden_layers",None); hidden=getattr(config,"hidden_size",None)
    n_heads=getattr(config,"num_attention_heads",None); n_kv=getattr(config,"num_key_value_heads",n_heads)
    if all(v is not None for v in [n_layers,hidden,n_heads,n_kv]) and n_heads:
        hd=hidden//n_heads; bp=2 if dtype in (torch.float16,torch.bfloat16) else 4
        kvpt=2*int(n_layers)*int(n_kv)*int(hd)*bp
        print(f"KV/token ~{kvpt/1024**2:.3f} MiB")
        for c in [32768,65536,131072,262144]:
            print(f"  KV@{c}: {kvpt*c/1e9:.3f} GB")

def make_prompt_ids(ctx, device):
    return torch.randint(10, 50000, (1,ctx), device=device, dtype=torch.long)

def chunked_prefill(model, ids, device, chunk=4096):
    """Prefill in chunks to avoid materializing the full seq x seq attention matrix."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    t0=time.perf_counter()
    past=None; last=None
    L=ids.shape[1]
    for st in range(0, L, chunk):
        sl=ids[:, st:min(st+chunk, L)]
        with torch.inference_mode():
            out=model(input_ids=sl, past_key_values=past, use_cache=True)
        past=getattr(out, "past_key_values", past)
        last=out
    if device.startswith("cuda"): torch.cuda.synchronize(device)
    dt=max(time.perf_counter()-t0,1e-9)
    peak=bytes_to_gb(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0.0
    return dt, L/dt, peak, past, last

def prefill_probe(model, ids, device):
    dt,tps,peak,past,last=chunked_prefill(model, ids, device)
    return dt, tps, peak

def build_niah(tok, ctx, depth):
    random.seed(42); secret=f"DORY-{random.randint(100000,999999)}"
    needle=f"\nThe secret code is {secret}.\n"
    fid=tok(FILLER,add_special_tokens=False).input_ids
    nid=tok(needle,add_special_tokens=False).input_ids
    qid=tok(QUESTION,add_special_tokens=False).input_ids
    budget=max(32, ctx-len(nid)-len(qid)-8)
    reps=max(1, (budget+len(fid)-1)//max(1,len(fid)))
    hay=(fid*reps)[:budget]
    ins=min(len(hay), max(0,int(len(hay)*depth)))
    full=hay[:ins]+nid+hay[ins:]+qid
    if len(full)>ctx:
        full=full[:ctx-len(qid)]+qid
    return tok.decode(full, skip_special_tokens=True), secret, len(full)

@torch.no_grad()
def niah_run(model, tok, text, secret, device, gentok):
    enc=tok(text, return_tensors="pt", add_special_tokens=True)
    ids=enc["input_ids"].to(device)
    plen=ids.shape[1]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    t0=time.perf_counter()
    dt,tps,peak,past,last=chunked_prefill(model, ids, device)
    t1=time.perf_counter(); prefill_s=max(t1-t0,1e-9)
    cur=torch.argmax(last.logits[:,-1,:],dim=-1,keepdim=True)
    gens=[cur]
    if device.startswith("cuda"): torch.cuda.synchronize(device)
    t_first=time.perf_counter()
    for _ in range(max(0,gentok-1)):
        step=model(input_ids=cur, past_key_values=past, use_cache=True)
        past=getattr(step,"past_key_values",past)
        cur=torch.argmax(step.logits[:,-1,:],dim=-1,keepdim=True)
        gens.append(cur)
        if tok.eos_token_id is not None and int(cur.item())==int(tok.eos_token_id): break
    if device.startswith("cuda"): torch.cuda.synchronize(device)
    gen_text=tok.decode(torch.cat(gens,dim=1)[0], skip_special_tokens=True)
    peak=peak if device.startswith("cuda") else 0.0
    return {"prefill_s":prefill_s,"prefill_tok_s":plen/prefill_s,"ttft_s":max(t_first-t0,1e-9),
            "peak_vram_gb":peak,"generated":gen_text.strip(),"recall_correct":secret.lower() in gen_text.lower(),
            "prompt_len":plen}

def main():
    args=parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr); sys.exit(2)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    print(f"Loading {args.model} (fp16) on {args.device}")
    model,tok,config,dtype=load_model(args.model, 1, args.device)
    print_budget(model, config, dtype)
    records=[]; feasible=[]
    print("\n=== Phase1 prefill OOM ladder ===")
    for ctx in [x for x in LADDER if x<=args.max_len]:
        cuda_cleanup()
        print(f"probe ctx={ctx}", flush=True)
        try:
            ids=make_prompt_ids(ctx, args.device)
            ps,tps,peak=prefill_probe(model,ids,args.device)
            records.append(asdict(RunRecord(phase="prefill_ladder",context_len=ctx,ok=True,peak_vram_gb=round(peak,4),prefill_tok_s=round(tps,2),prefill_s=round(ps,4))))
            feasible.append(ctx); print(f"  OK peak={peak:.3f}GB {tps:.1f}tok/s")
            del ids
        except RuntimeError as e:
            cuda_cleanup()
            oom=is_oom(e)
            records.append(asdict(RunRecord(phase="prefill_ladder",context_len=ctx,ok=False,oom=oom,error=str(e)[:400])))
            print(f"  {'OOM' if oom else 'ERR'}: {str(e).splitlines()[0][:150]}")
            if oom:
                print(f"OOM boundary at {ctx}, stopping"); break
    del model; cuda_cleanup()
    print("\n=== Phase2 NIAH ===")
    for loops in sorted(set([1,args.loops])):
        print(f"\n-- loops={loops} --")
        try:
            model,tok,config,dtype=load_model(args.model, loops, args.device)
        except RuntimeError as e:
            cuda_cleanup()
            records.append(asdict(RunRecord(phase="niah_load",context_len=0,loops=loops,ok=False,oom=is_oom(e),error=str(e)[:400])))
            if is_oom(e): break
            continue
        for ctx in feasible:
            for depth in DEPTHS:
                cuda_cleanup()
                print(f"  loops={loops} ctx={ctx} depth={depth:.2f}", flush=True)
                try:
                    text,secret,_=build_niah(tok,ctx,depth)
                    m=niah_run(model,tok,text,secret,args.device,args.gen_tokens)
                    rec=asdict(RunRecord(phase="niah",context_len=ctx,loops=loops,depth=depth,ok=True,
                        peak_vram_gb=round(m["peak_vram_gb"],4),prefill_tok_s=round(m["prefill_tok_s"],2),
                        prefill_s=round(m["prefill_s"],4),ttft_s=round(m["ttft_s"],4),
                        recall_correct=bool(m["recall_correct"]),generated=m["generated"][:120],secret=secret))
                    print(f"    recall={rec['recall_correct']} peak={rec['peak_vram_gb']}GB prefill={rec['prefill_tok_s']}tok/s ttft={rec['ttft_s']}s")
                except RuntimeError as e:
                    cuda_cleanup()
                    rec=asdict(RunRecord(phase="niah",context_len=ctx,loops=loops,depth=depth,ok=False,oom=is_oom(e),error=str(e)[:400]))
                    print(f"    {'OOM' if rec['oom'] else 'ERR'}: {str(e).splitlines()[0][:120]}")
                records.append(rec)
                if rec.get("oom"):
                    break
            if records and records[-1].get("phase")=="niah" and records[-1].get("oom") and records[-1].get("context_len")==ctx:
                print(f"  stopping NIAH loops={loops} at ctx={ctx} (OOM)"); break
        del model; cuda_cleanup()
    payload={"meta":{"model":args.model,"device":args.device,"dtype":"float16","max_len":args.max_len,
            "loops_tested":sorted(set([1,args.loops])),"ladder":[x for x in LADDER if x<=args.max_len],"depths":DEPTHS},
            "feasible_prefill_lengths":feasible,"results":records}
    with open(args.out,"w") as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {args.out}\nFeasible: {feasible}")

if __name__=="__main__": main()
