#!/usr/bin/env python3
"""E4 difficulty router: predict escalation (loop4 correct, loop2 wrong)."""
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = ("gsm8k", "math", "bbh")


def load_checkpoints(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_family(row):
    ds = str(row.get("dataset", "")).lower().strip()
    if ds in FAMILIES:
        return ds
    q = str(row.get("question", ""))
    low = q.lower()
    if "\\" in q or "$" in q or "matrix" in low:
        return "math"
    if "passage" in low or "read the following" in low:
        return "bbh"
    idx = row.get("idx", 0)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = 0
    if idx < 100:
        return "gsm8k"
    if idx < 200:
        return "math"
    return "bbh"


def features(question, family):
    q = str(question)
    toks = q.split()
    has_equal = 1.0 if "=" in q else 0.0
    has_plus = 1.0 if "+" in q else 0.0
    has_digit = 1.0 if any(c.isdigit() for c in q) else 0.0
    math_marker = 1.0 if (has_equal or has_plus or has_digit) else 0.0
    onehot = [1.0 if family == fam else 0.0 for fam in FAMILIES]
    return [
        float(len(q)),
        float(len(toks)),
        has_equal,
        has_plus,
        has_digit,
        math_marker,
    ] + onehot


FEATURE_NAMES = [
    "prompt_len_chars",
    "token_count_est",
    "has_equal",
    "has_plus",
    "has_digit",
    "has_math_marker",
    "family_gsm8k",
    "family_math",
    "family_bbh",
]


def try_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401

        return True
    except Exception:
        return False


SKLEARN_AVAILABLE = try_sklearn()


def standardize_fit(X):
    n = len(X)
    d = len(X[0]) if n else 0
    mean = [sum(row[j] for row in X) / n for j in range(d)] if n else [0.0] * d
    var = [sum((row[j] - mean[j]) ** 2 for row in X) / n for j in range(d)] if n else [1.0] * d
    std = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    return mean, std


def standardize_transform(X, mean, std):
    return [[(row[j] - mean[j]) / std[j] for j in range(len(row))] for row in X]


def train_logistic_pure(X, y, epochs=2000, lr=0.5, l2=1e-3):
    n = len(X)
    d = len(X[0]) if n else 0
    npos = sum(1 for v in y if v == 1)
    nneg = n - npos
    wpos = n / (2.0 * npos) if npos > 0 else 1.0
    wneg = n / (2.0 * nneg) if nneg > 0 else 1.0
    weights = [0.0] * (d + 1)
    for _ in range(epochs):
        grad = [0.0] * (d + 1)
        for i in range(n):
            xi = list(X[i]) + [1.0]
            z = sum(weights[j] * xi[j] for j in range(d + 1))
            z = max(-30.0, min(30.0, z))
            pred = 1.0 / (1.0 + math.exp(-z))
            diff = (pred - y[i]) * (wpos if y[i] == 1 else wneg)
            for j in range(d + 1):
                grad[j] += diff * xi[j]
        for j in range(d):
            grad[j] += l2 * weights[j]
        for j in range(d + 1):
            weights[j] -= lr * grad[j] / max(1, n)
    return weights


def predict_pure(weights, X):
    out = []
    for row in X:
        xi = list(row) + [1.0]
        z = sum(weights[j] * xi[j] for j in range(len(weights)))
        z = max(-30.0, min(30.0, z))
        p = 1.0 / (1.0 + math.exp(-z))
        out.append(1 if p >= 0.5 else 0)
    return out


class RouterModel:
    def __init__(self):
        self.backend = "sklearn" if SKLEARN_AVAILABLE else "pure_python"
        self.scaler = None
        self.clf = None
        self.mean = None
        self.std = None
        self.weights = None

    def fit(self, X, y):
        if SKLEARN_AVAILABLE:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
            self.clf = LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="lbfgs"
            )
            self.clf.fit(Xs, y)
        else:
            self.mean, self.std = standardize_fit(X)
            Xs = standardize_transform(X, self.mean, self.std)
            self.weights = train_logistic_pure(Xs, y)

    def predict(self, X):
        if SKLEARN_AVAILABLE:
            Xs = self.scaler.transform(X)
            return [int(v) for v in self.clf.predict(Xs)]
        Xs = standardize_transform(X, self.mean, self.std)
        return predict_pure(self.weights, Xs)

    def coefficients(self):
        if SKLEARN_AVAILABLE:
            try:
                return [float(v) for v in self.clf.coef_[0].tolist()]
            except Exception:
                return None
        if self.weights is None:
            return None
        return [float(v) for v in self.weights[:-1]]


def stratified_folds(labels, k=5, seed=42):
    rng = random.Random(seed)
    pos = [i for i, v in enumerate(labels) if v == 1]
    neg = [i for i, v in enumerate(labels) if v == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(k)]
    for j, i in enumerate(pos):
        folds[j % k].append(i)
    for j, i in enumerate(neg):
        folds[j % k].append(i)
    return folds


def build_samples(ckpt_path):
    rows = load_checkpoints(ckpt_path)
    by_idx = defaultdict(dict)
    for r in rows:
        try:
            loop = int(r.get("loop"))
        except (TypeError, ValueError):
            continue
        by_idx[r.get("idx")][loop] = r

    samples = []
    for idx, d in by_idx.items():
        r2 = d.get(2)
        r4 = d.get(4)
        if r2 is None or r4 is None:
            continue
        fam = task_family(r2)
        l2c = bool(r2.get("correct", False))
        l4c = bool(r4.get("correct", False))
        esc = l4c and (not l2c)
        q = r2.get("question", "")
        samples.append(
            {
                "idx": idx,
                "dataset": r2.get("dataset"),
                "family": fam,
                "question": q,
                "loop2_correct": l2c,
                "loop4_correct": l4c,
                "escalation": esc,
                "features": features(q, fam),
            }
        )
    return samples


def render_markdown(result):
    s = result["summary"]
    lines = []
    lines.append("# E4: Difficulty router (escalation predictor)")
    lines.append("")
    lines.append(f"- Checkpoint: `{result['checkpoint']}`")
    lines.append(f"- Backend: `{result['backend']}`")
    lines.append(f"- Samples: `{s['n']}`")
    lines.append(f"- Escalation positives: `{s['n_escalation']}` ({s['escalation_rate']:.3%})")
    lines.append("")
    lines.append("## Metrics (out-of-fold, 5-fold CV)")
    lines.append("")
    lines.append(f"- Classifier accuracy: **{s['classifier_accuracy']:.4f}**")
    lines.append(f"- Misroute rate: **{s['misroute_rate']:.4f}**")
    lines.append(f"- Net accuracy if router applied: **{s['net_accuracy_router']:.4f}**")
    lines.append(f"- Oracle capture %: **{s['oracle_capture_pct']:.2f}%**")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append(f"- Fixed loop2 accuracy: {s['baseline_loop2_accuracy']:.4f}")
    lines.append(f"- Fixed loop4 accuracy: {s['baseline_loop4_accuracy']:.4f}")
    lines.append(f"- Oracle (best of 2/4) accuracy: {s['oracle_accuracy']:.4f}")
    lines.append("")
    lines.append("## Confusion matrix (escalation)")
    lines.append("")
    cm = s["confusion"]
    lines.append(f"- TP: {cm['tp']}  FP: {cm['fp']}")
    lines.append(f"- FN: {cm['fn']}  TN: {cm['tn']}")
    lines.append("")
    lines.append("## Feature coefficients")
    lines.append("")
    coefs = result.get("coefficients") or {}
    for name in FEATURE_NAMES:
        if name in coefs:
            lines.append(f"- {name}: {coefs[name]:.4f}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="E4 difficulty router")
    ap.add_argument("--out", default="results/e4", help="Output directory")
    ap.add_argument("--ckpt", default="results/e5/checkpoints.jsonl", help="E5 checkpoints.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(args.ckpt)
    if not samples:
        raise SystemExit(f"No samples built from {args.ckpt}")

    X = [s["features"] for s in samples]
    y = [1 if s["escalation"] else 0 for s in samples]

    folds = stratified_folds(y, k=5, seed=args.seed)
    oof = [0] * len(samples)
    for fold in folds:
        test_set = set(fold)
        train_idx = [i for i in range(len(samples)) if i not in test_set]
        test_idx = list(fold)
        model = RouterModel()
        model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        preds = model.predict([X[i] for i in test_idx])
        for i, p in zip(test_idx, preds):
            oof[i] = int(p)

    final_model = RouterModel()
    final_model.fit(X, y)
    coefs = final_model.coefficients()
    coef_map = {}
    if coefs is not None:
        for name, val in zip(FEATURE_NAMES, coefs):
            coef_map[name] = float(val)

    tp = sum(1 for i in range(len(samples)) if y[i] == 1 and oof[i] == 1)
    tn = sum(1 for i in range(len(samples)) if y[i] == 0 and oof[i] == 0)
    fp = sum(1 for i in range(len(samples)) if y[i] == 0 and oof[i] == 1)
    fn = sum(1 for i in range(len(samples)) if y[i] == 1 and oof[i] == 0)

    n = len(samples)
    n_esc = sum(y)
    classifier_accuracy = (tp + tn) / n if n else 0.0
    misroute_rate = (fp + fn) / n if n else 0.0

    correct = 0
    for i, s in enumerate(samples):
        if oof[i] == 1:
            if s["loop4_correct"]:
                correct += 1
        else:
            if s["loop2_correct"]:
                correct += 1
    net_acc = correct / n if n else 0.0

    baseline2 = sum(1 for s in samples if s["loop2_correct"]) / n
    baseline4 = sum(1 for s in samples if s["loop4_correct"]) / n
    oracle = sum(1 for s in samples if (s["loop2_correct"] or s["loop4_correct"])) / n
    capture = (tp / n_esc * 100.0) if n_esc > 0 else 0.0

    result = {
        "checkpoint": str(args.ckpt),
        "backend": final_model.backend,
        "features": FEATURE_NAMES,
        "summary": {
            "n": n,
            "n_escalation": n_esc,
            "escalation_rate": n_esc / n if n else 0.0,
            "classifier_accuracy": classifier_accuracy,
            "misroute_rate": misroute_rate,
            "false_escalation_rate": fp / n if n else 0.0,
            "missed_escalation_rate": fn / n if n else 0.0,
            "net_accuracy_router": net_acc,
            "oracle_capture_pct": capture,
            "baseline_loop2_accuracy": baseline2,
            "baseline_loop4_accuracy": baseline4,
            "oracle_accuracy": oracle,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "cv_folds": 5,
        },
        "coefficients": coef_map,
        "family_counts": dict(Counter(s["family"] for s in samples)),
    }

    json_path = out_dir / "results.json"
    md_path = out_dir / "summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))

    with open(out_dir / "e4_router.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(out_dir / "e4_router.md", "w", encoding="utf-8") as f:
        f.write(render_markdown(result))

    print(f"Samples: {n}  escalation: {n_esc}")
    print(f"Classifier accuracy: {classifier_accuracy:.4f}")
    print(f"Misroute rate: {misroute_rate:.4f}")
    print(f"Net accuracy router: {net_acc:.4f}")
    print(f"Oracle capture %: {capture:.2f}%")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
