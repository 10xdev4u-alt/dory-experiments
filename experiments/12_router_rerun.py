#!/usr/bin/env python3
"""E12: router rerun on position-aware (E11) re-extracted labels.

Loads E5 checkpoints, re-extracts answers with the position-aware method from
E11, computes per-prompt correctness at loop2 and loop4 under re-extraction,
trains a logistic-regression router on the 5 frozen features from the E4 spec,
and evaluates accuracy / misroute / net accuracy vs static / oracle capture.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# E11 position-aware extraction (reused)
# ---------------------------------------------------------------------------

DOUBT_MARKERS = [
    "let me double-check",
    "let me verify",
    "let me re-read",
    "let me check",
    "wait,",
    "actually,",
    "double-check",
    "reconsider",
]


def normalize_number(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_boxed_balanced(text, first=True):
    results = []
    for m in __import__("re").finditer(r"\\boxed\s*\{", text or ""):
        i = m.end() - 1
        depth = 0
        start = i + 1
        j = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[start:j])
                    break
            j += 1
    if not results:
        return ""
    return results[0] if first else results[-1]


import re


def _first_block(text):
    if "</think>" in (text or ""):
        return (text or "").split("</think>", 1)[0]
    return text or ""


def _trim_doubt(block):
    pos = len(block)
    low = (block or "").lower()
    for marker in DOUBT_MARKERS:
        i = low.find(marker)
        if i != -1 and i < pos:
            pos = i
    return (block or "")[:pos]


def _clean_gsm(candidate):
    c = (candidate or "").strip()
    m = re.search(r"-?\d[\d,\.]*%", c)
    if m:
        return m.group(0)
    m = re.search(r"-?\d[\d,\.]*", c)
    if m:
        return m.group(0).replace(",", "")
    return c.rstrip(".").strip()


def extract_gsm(text):
    block = _trim_doubt(_first_block(text or ""))
    cands = []
    for m in re.finditer(r"Answer:\s*([^\n<>]+)", block, re.I):
        cands.append((m.start(), m.group(1).strip()))
    for m in re.finditer(r"answer is\s*([^\n<>]+)", block, re.I):
        cands.append((m.start(), m.group(1).strip()))
    for m in re.finditer(r"final answer[:\s]*([^\n<>]+)", block, re.I):
        cands.append((m.start(), m.group(1).strip()))
    for m in re.finditer(r"=", block):
        line_start = block.rfind("\n", 0, m.start()) + 1
        line_end = block.find("\n", m.start())
        if line_end == -1:
            line_end = len(block)
        line = block[line_start:line_end]
        seg = line.rsplit("=", 1)[-1].strip()
        cands.append((m.start(), seg))
    for m in re.finditer(r"\bis\s+(-?[\d][\d,\.]*%?)", block, re.I):
        cands.append((m.start(), m.group(1).strip()))
    if not cands:
        nums = re.findall(r"-?\d+(?:\.\d+)?", block)
        return nums[-1] if nums else ""
    cands.sort(key=lambda x: x[0])
    return _clean_gsm(cands[-1][1])


def extract_math(text):
    got = extract_boxed_balanced(text or "", first=True)
    if got:
        return got
    block = _trim_doubt(_first_block(text or ""))
    cands = []
    for m in re.finditer(r"Answer:\s*([^\n<>]+)", block, re.I):
        cands.append((m.start(), m.group(1).strip()))
    for m in re.finditer(r"answer is\s*([^\n<>]+)", block, re.I):
        cands.append((m.start(), m.group(1).strip()))
    for m in re.finditer(r"=", block):
        line_start = block.rfind("\n", 0, m.start()) + 1
        line_end = block.find("\n", m.start())
        if line_end == -1:
            line_end = len(block)
        line = block[line_start:line_end]
        seg = line.rsplit("=", 1)[-1].strip()
        cands.append((m.start(), seg))
    if not cands:
        nums = re.findall(r"-?\d+(?:\.\d+)?", block)
        return nums[-1] if nums else ""
    cands.sort(key=lambda x: x[0])
    best = cands[-1][1]
    token = best.split()[0] if best.split() else best
    token = token.rstrip(".").rstrip(",").rstrip("\\").rstrip(")").strip()
    return token if token else best


def grade(item, got):
    ds = item["dataset"]
    target = str(item["target"]).strip()
    if ds == "gsm8k":
        a, b = normalize_number(got), normalize_number(target)
        if a is not None and b is not None:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return str(got).strip() == target
    if ds == "math":
        a, b = normalize_number(got), normalize_number(target)
        if a is not None and b is not None:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        norm = lambda x: re.sub(r"\s+", "", str(x).strip().lower())
        return norm(got) == norm(target)
    return False


def reextract(record):
    if record["dataset"] == "gsm8k":
        return extract_gsm(record["output"])
    if record["dataset"] == "math":
        return extract_math(record["output"])
    return ""


# ---------------------------------------------------------------------------
# Router (E4-style, pure python; 5 frozen features)
# ---------------------------------------------------------------------------

FAMILIES = ("gsm8k", "math", "bbh")

FEATURE_NAMES = [
    "log_len",
    "digit_density",
    "math_symbol",
    "question_marks",
    "task_type",
]


def load_checkpoints(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def features(question):
    q = str(question)
    n = len(q)
    n_digits = sum(1 for c in q if c.isdigit())
    log_len = math.log1p(n) if n > 0 else 0.0
    digit_density = (n_digits / n) if n > 0 else 0.0
    math_symbols = set("=+-×÷$\\")
    math_symbol = 1.0 if (any(c in math_symbols for c in q) or "\\frac" in q) else 0.0
    question_marks = float(q.count("?"))
    low = q.lower()
    word_patterns = ["how many", "how much", "total", "each", "sells", "buys"]
    task_type = 1.0 if (n_digits > 0 and any(p in low for p in word_patterns)) else 0.0
    return [log_len, digit_density, math_symbol, question_marks, task_type]


def try_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401

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
        # re-extract with E11 position-aware method
        g2 = reextract(r2)
        g4 = reextract(r4)
        l2c = bool(grade(r2, g2))
        l4c = bool(grade(r4, g4))
        esc = l4c and (not l2c)
        q = r2.get("question", "")
        samples.append(
            {
                "idx": idx,
                "dataset": r2.get("dataset"),
                "question": q,
                "loop2_correct": l2c,
                "loop4_correct": l4c,
                "reextracted_loop2": g2,
                "reextracted_loop4": g4,
                "escalation": esc,
                "features": features(q),
            }
        )
    return samples


def render_markdown(result):
    s = result["summary"]
    lines = []
    lines.append("# E12: Router rerun on position-aware (E11) re-extracted labels")
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
    lines.append("## Baselines (re-extracted labels)")
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
    ap = argparse.ArgumentParser(description="E12 router rerun on re-extracted labels")
    ap.add_argument("--out", default="results/e12", help="Output directory")
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
        "family_counts": dict(Counter(s["dataset"] for s in samples)),
    }

    json_path = out_dir / "results.json"
    md_path = out_dir / "summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
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
