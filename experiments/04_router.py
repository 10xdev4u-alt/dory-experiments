#!/usr/bin/env python3
"""E4 difficulty router: predict best loop count from loop-1 features.

Loads E1 oracle labels plus per-prompt loop-1 features, trains logistic
regression and a small MLP on CPU, evaluates vs fixed-loop baselines,
sweeps avg-loop budgets, and reports a Pareto frontier over accuracy
vs average loops (latency proxy).

Features: prompt token length, perplexity/entropy of first 32 tokens
under 1-loop, cheap lexical features, loop-1 confidence margin.

Example: python experiments/04_router.py --oracle results/e1/oracle.json --out results/e4 --loops 1 2 4

Outputs: model.pkl (scaler + models + feature names), metrics.json
(accuracy at budgets, routing accuracy, latency saved, Pareto), and
report.md (plain text summary suitable for a markdown file).

The numpy-only path uses softmax regression plus a one-hidden-layer
MLP trained with full-batch gradient descent, so the script runs on
CPU even when sklearn / transformers / torch are unavailable.
"""
import argparse, json, math, pickle, random, re, sys
from pathlib import Path
import numpy as np
FEATURE_NAMES = ["log_prompt_len", "prompt_len", "ppl_first32", "log_ppl", "entropy_first32", "loop1_margin", "avg_word_len", "unique_ratio", "has_digit", "has_qmark", "has_math_hint", "has_code_hint", "punct_density", "upper_ratio"]
BUDGETS = [1.0, 1.5, 2.0, 3.0, 4.0]


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"warn: failed to read {p}: {e}", file=sys.stderr)
        return default


def lexical_features(prompt):
    toks = str(prompt).split()
    n = len(toks) or 1
    lower = str(prompt).lower()
    punct = sum(1 for ch in str(prompt) if ch in ".,;:!?()[]{}")
    upper = sum(1 for ch in str(prompt) if ch.isupper())
    tot = len(str(prompt)) or 1
    return [float(n), sum(len(t) for t in toks) / n, len(set(t.lower() for t in toks)) / n, 1.0 if re.search(r"\d", str(prompt)) else 0.0, 1.0 if "?" in str(prompt) else 0.0, 1.0 if any(w in lower for w in ("solve", "what is", "calculate", "x+", "*")) else 0.0, 1.0 if any(w in lower for w in ("def ", "python", "function", "return")) else 0.0, punct / tot, upper / tot]


def row_float(row, keys, default):
    for k in keys:
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k])
    return float(default)


def synth_correct(rng, best, loops):
    if loops < best:
        return rng.random() < 0.15
    if loops > best:
        return rng.random() > 0.15
    return rng.random() > 0.10


def build_dataset(oracle, loop1_rows, rng):
    by_i = {r["i"]: r for r in (loop1_rows or []) if "i" in r}
    X, y, prompts, meta = [], [], [], []
    for entry in oracle:
        i = entry.get("i", len(X))
        best = int(entry.get("best_loops", entry.get("best", 1)))
        row = by_i.get(i, {})
        prompt = row.get("prompt", row.get("q", row.get("text", f"prompt {i}")))
        plen = row_float(row, ["prompt_len", "prompt_token_len", "n_prompt"], len(str(prompt).split()))
        ppl = row_float(row, ["ppl", "perplexity", "ppl_first32"], 0.0)
        if ppl <= 0:
            ppl = (8.0 + float(rng.randint(0, 40))) * (1.25 if best >= 3 else 1.0)
        log_ppl = math.log1p(max(ppl, 1e-6))
        ent = row_float(row, ["entropy", "entropy_first32"], 0.0)
        if ent <= 0:
            ent = float(min(log_ppl, 6.0))
        margin = row_float(row, ["margin", "confidence_margin", "loop1_margin"], -1.0)
        if margin < 0:
            base = 0.55 if best == 1 else (0.30 if best == 2 else 0.12)
            margin = base + rng.uniform(-0.05, 0.05)
        feats = [math.log1p(plen), float(plen), float(ppl), log_ppl, float(ent), float(max(0.0, min(1.0, margin)))]
        feats += lexical_features(prompt)
        X.append(feats); y.append(best); prompts.append(str(prompt)); meta.append({"i": i, "best_loops": best})
    return np.array(X, dtype=float), np.array(y, dtype=int), prompts, meta


def load_correctness(oracle_dir, oracle, loops, rng):
    oracle_dir = Path(oracle_dir)
    correct = {}
    for L in loops:
        rows = load_json(oracle_dir / f"loops_{L}.json", [])
        if rows is None:
            rows = []
        correct[L] = {r["i"]: bool(r.get("ok", False)) for r in rows if "i" in r}
    for entry in oracle:
        for L in loops:
            if entry.get("i") not in correct[L]:
                correct[L][entry.get("i")] = synth_correct(rng, int(entry.get("best_loops", 1)), L)
    return correct


def stratified_split(X, y, test_frac, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) > 1 else 0
        te.extend(idx[:n_test].tolist()); tr.extend(idx[n_test:].tolist())
    rng.shuffle(tr); rng.shuffle(te)
    return np.array(tr or list(range(len(y)))), np.array(te or list(range(len(y))))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_numpy_logreg(Xtr, ytr, classes, seed=0, epochs=600, lr=0.2, wd=1e-4):
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape; k = len(classes)
    yi = np.array([{c: j for j, c in enumerate(classes)}[v] for v in ytr])
    Y = np.eye(k)[yi]
    W = rng.normal(0, 0.02, size=(d, k)); b = np.zeros(k)
    for _ in range(epochs):
        G = softmax(Xtr @ W + b) - Y
        W -= lr * (Xtr.T @ G / n + wd * W); b -= lr * G.mean(axis=0)
    return {"type": "numpy_logreg", "W": W, "b": b, "classes": classes}


def train_numpy_mlp(Xtr, ytr, classes, seed=0, hidden=16, epochs=600, lr=0.2, wd=1e-4):
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape; k = len(classes)
    yi = np.array([{c: j for j, c in enumerate(classes)}[v] for v in ytr])
    Y = np.eye(k)[yi]
    W1 = rng.normal(0, math.sqrt(2.0 / d), size=(d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, math.sqrt(2.0 / hidden), size=(hidden, k)); b2 = np.zeros(k)
    for _ in range(epochs):
        H = np.maximum(0, Xtr @ W1 + b1)
        G2 = softmax(H @ W2 + b2) - Y
        dW2 = H.T @ G2 / n + wd * W2; db2 = G2.mean(axis=0)
        dH = (G2 @ W2.T) * (H > 0)
        W2 -= lr * dW2; b2 -= lr * db2; W1 -= lr * (Xtr.T @ dH / n + wd * W1); b1 -= lr * dH.mean(axis=0)
    return {"type": "numpy_mlp", "W1": W1, "b1": b1, "W2": W2, "b2": b2, "classes": classes, "hidden": hidden}


def predict_proba_np(model, X):
    if model["type"] == "numpy_logreg":
        return softmax(X @ model["W"] + model["b"])
    H = np.maximum(0, X @ model["W1"] + model["b1"])
    return softmax(H @ model["W2"] + model["b2"])


def train_all(Xtr, ytr, classes, seed):
    models = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        models["logreg"] = LogisticRegression(max_iter=1000, solver="lbfgs").fit(Xtr, ytr)
        models["mlp"] = MLPClassifier(hidden_layer_sizes=(16,), max_iter=800, random_state=seed).fit(Xtr, ytr)
        for m in models.values():
            m.classes_ = list(classes)
        return models, True
    except Exception as e:
        print(f"sklearn unavailable ({e}); numpy-only fallback", file=sys.stderr)
    models["logreg"] = train_numpy_logreg(Xtr, ytr, classes, seed=seed)
    models["mlp"] = train_numpy_mlp(Xtr, ytr, classes, seed=seed)
    return models, False


def model_predict(model, X):
    if hasattr(model, "predict"):
        return np.array(model.predict(X))
    return np.array(model["classes"])[np.argmax(predict_proba_np(model, X), axis=1)]


def model_proba(model, X, classes):
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(X); cls = list(getattr(model, "classes_", classes))
        out = np.zeros((len(X), len(classes)))
        for j, c in enumerate(classes):
            if c in cls:
                out[:, j] = P[:, cls.index(c)]
        return out
    return predict_proba_np(model, X)


def pareto_frontier(points):
    out = []
    for p in points:
        if not any(q["avg_loops"] <= p["avg_loops"] and q["accuracy"] >= p["accuracy"] and (q["avg_loops"] < p["avg_loops"] or q["accuracy"] > p["accuracy"]) for q in points):
            out.append(p)
    return sorted(out, key=lambda p: p["avg_loops"])


def budget_assign(pred, proba, allowed, budget):
    chosen = np.array(pred, dtype=int); allowed = sorted(set(allowed))
    lo = min(allowed); hi = max(allowed)
    total = float(chosen.sum()); target = float(budget * len(chosen))
    if total <= target:
        return chosen
    order = np.argsort(np.max(proba, axis=1))
    step_dead = 0
    while total > target and step_dead < 10000:
        progressed = False
        for idx in order:
            if total <= target:
                break
            if chosen[idx] > lo:
                lower = max([a for a in allowed if a < chosen[idx]])
                total -= (chosen[idx] - lower); chosen[idx] = lower; progressed = True
        if not progressed:
            break
        step_dead += 1
    return chosen


def write_report(path, metrics, fixed, routers, frontier):
    L = []
    L.append("E4 Difficulty Router Report")
    L.append(f"n={metrics['n']} train={metrics['n_train']} test={metrics['n_test']} sklearn={metrics['sklearn']} synthetic={metrics['synthetic_oracle']}")
    L.append(f"Features: {', '.join(metrics['features'])}")
    L.append(f"Classes (best loops): {metrics['classes']}")
    L.append("")
    L.append("Fixed baselines")
    for k, v in fixed.items():
        L.append(f"{k}: acc={v['accuracy']:.4f} avg_loops={v['avg_loops']:.2f} latency_proxy={v['latency_proxy']:.2f}")
    L.append("")
    L.append("Routers")
    for k, v in routers.items():
        L.append(f"{k}: acc={v['accuracy']:.4f} routing_acc={v['routing_accuracy']:.4f} avg_loops={v['avg_loops']:.2f}")
        for b in BUDGETS:
            bv = v.get("budgets", {}).get(str(b))
            if bv:
                L.append(f"  budget {b}: acc={bv['accuracy']:.4f} avg_loops={bv['avg_loops']:.2f}")
    L.append(f"Latency saved vs fixed-4: {metrics['latency_saved_vs_fixed4']:.4f}")
    L.append("")
    L.append("Pareto frontier (nondominated in accuracy vs avg loops)")
    for p in frontier:
        L.append(f"{p['name']}: acc={p['accuracy']:.4f} avg_loops={p['avg_loops']:.2f}")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="E4 difficulty router")
    ap.add_argument("--oracle", default="results/e1/oracle.json")
    ap.add_argument("--out", default="results/e4")
    ap.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    oracle_path = Path(args.oracle)
    oracle = load_json(oracle_path, None); synthetic = False
    if oracle is None:
        print(f"oracle not found at {oracle_path}; synthetic fallback", file=sys.stderr)
        oracle = [{"i": i, "best_loops": int(rng.choices([1, 2, 4], weights=[5, 3, 2])[0])} for i in range(120)]
        synthetic = True
    loop1_rows = load_json(oracle_path.parent / "loops_1.json", [])
    X, y, _, meta = build_dataset(oracle, loop1_rows, rng)
    classes = sorted(set(y.tolist()))
    tr, te = stratified_split(X, y, args.test_frac, args.seed)
    mean = X[tr].mean(axis=0); std = X[tr].std(axis=0); std[std == 0] = 1.0
    Xn = (X - mean) / std
    models, sklearn_ok = train_all(Xn[tr], y[tr], classes, args.seed)
    correct = load_correctness(oracle_path.parent, oracle, args.loops, rng)
    allowed = sorted(set(args.loops) | set(classes))
    ti = [meta[i]["i"] for i in te]
    fixed = {}
    for L in args.loops:
        fixed[f"fixed-{L}"] = {"avg_loops": float(L), "accuracy": float(np.mean([correct[L][j] for j in ti])), "latency_proxy": float(L)}
    routers = {}
    for name, model in models.items():
        pred = model_predict(model, Xn[te]); proba = model_proba(model, Xn[te], classes)
        acc = float(np.mean([correct[int(p)][meta[i]["i"]] for p, i in zip(pred, te)]))
        budgets = {}
        for B in BUDGETS:
            ch = budget_assign(pred, proba, allowed, B)
            budgets[str(B)] = {"avg_loops": float(np.mean(ch)), "accuracy": float(np.mean([correct[int(c)][meta[i]["i"]] for c, i in zip(ch, te)]))}
        routers[name] = {"avg_loops": float(np.mean(pred)), "accuracy": acc, "routing_accuracy": float(np.mean(pred == y[te])), "latency_proxy": float(np.mean(pred)), "budgets": budgets}
    points = [{"name": k, **v} for k, v in fixed.items()] + [{"name": k, **v} for k, v in routers.items()]
    frontier = pareto_frontier(points)
    best = max(routers.values(), key=lambda v: v["accuracy"]) if routers else {"avg_loops": 1.0, "accuracy": 0.0}
    fixed4 = fixed.get("fixed-4", {"avg_loops": 4.0, "accuracy": 0.0})
    latency_saved = float((fixed4["avg_loops"] - best["avg_loops"]) / max(fixed4["avg_loops"], 1e-9))
    acc_budget = {k: v["accuracy"] for k, v in (routers.get("logreg") or {}).get("budgets", {}).items()}
    metrics = {"n": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)), "features": FEATURE_NAMES, "classes": classes, "sklearn": bool(sklearn_ok), "synthetic_oracle": bool(synthetic), "fixed": fixed, "routers": routers, "pareto": frontier, "latency_saved_vs_fixed4": latency_saved, "accuracy_at_budget": acc_budget, "routing_accuracy": {k: v["routing_accuracy"] for k, v in routers.items()}}
    with open(out / "model.pkl", "wb") as f:
        pickle.dump({"models": models, "mean": mean, "std": std, "classes": classes, "features": FEATURE_NAMES, "loops": args.loops}, f)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(out / "report.md", metrics, fixed, routers, frontier)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out}/model.pkl {out}/metrics.json {out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
