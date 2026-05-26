"""
Orchestrates train / evaluate / infer for ML classifiers.
"""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.model_ml.base import BaseMLClassifier
from src.model_llm.evaluator import compute_metrics

if TYPE_CHECKING:
    pass

ML_MODEL_DIR = Path("data/ml_model")


# ── Classifier registry ──────────────────────────────────────────────────────

def get_classifier(name: str) -> BaseMLClassifier:
    if name == "svm":
        from src.model_ml.svm.model import SVMClassifier
        return SVMClassifier()
    if name == "logistic_regression":
        from src.model_ml.logistic_regression.model import LogisticRegressionClassifier
        return LogisticRegressionClassifier()
    if name == "naive_bayes":
        from src.model_ml.naive_bayes.model import NaiveBayesClassifier
        return NaiveBayesClassifier()
    print(f"[ERROR] Unknown ML model: '{name}'", file=sys.stderr)
    sys.exit(1)


def load_classifier(name: str) -> BaseMLClassifier:
    model_dir = ML_MODEL_DIR / name
    pkl = model_dir / "model.pkl"
    if not pkl.exists():
        print(f"[ERROR] Model artifact not found: {pkl}. Run --action train first.", file=sys.stderr)
        sys.exit(1)
    clf = get_classifier(name)
    return clf.__class__.load(model_dir)


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_dataset(path: Path) -> tuple[list[str], dict[str, list]]:
    """
    Load a dataset JSONL where each line has text + labels.
    Returns:
        texts: list[str]
        labels: {axis_en: [value_per_story, ...]}
                Multi-label axes → list[list[str]]
    """
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] Dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"[ERROR] Dataset is empty: {path}", file=sys.stderr)
        sys.exit(1)

    texts: list[str] = [r["text"] for r in records]
    axis_keys = [k for k in records[0] if k not in ("id", "text")]
    labels: dict[str, list] = {k: [r[k] for r in records] for k in axis_keys}
    return texts, labels


def load_labeled_stories(path: Path) -> tuple[list[dict], dict]:
    """
    Load dataset JSONL → (stories list, ground_truth dict) for evaluator.
    stories: [{"id": ..., "text": ...}]
    gt:      {story_id: {axis_en: value}}
    """
    path = Path(path)
    stories, gt = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stories.append({"id": obj["id"], "text": obj["text"]})
            gt[obj["id"]] = {k: v for k, v in obj.items() if k not in ("id", "text")}
    return stories, gt


# ── Train ────────────────────────────────────────────────────────────────────

def train_model(clf: BaseMLClassifier, train_path: Path, axes: list[dict]) -> None:
    print(f"[INFO] Loading training data from {train_path} ...", file=sys.stderr)
    texts, labels = load_dataset(train_path)
    print(f"[INFO] Training {clf.name} on {len(texts)} samples ...", file=sys.stderr)
    clf.train(texts, axes, labels)

    model_dir = ML_MODEL_DIR / clf.name
    clf.save(model_dir)
    print(f"[INFO] Model saved → {model_dir / 'model.pkl'}", file=sys.stderr)

    # Quick train-set accuracy
    print("\n── Train-set accuracy (sanity check) ──")
    correct = {a["metadata_en"]: 0 for a in axes}
    for text, sid in zip(texts, range(len(texts))):
        pred = clf.predict_one(text, axes)
        for axis in axes:
            aen = axis["metadata_en"]
            is_multi = axis["loai_nhan"] == "Đa nhãn"
            gt_val  = labels[aen][sid]
            pr_val  = pred[aen]
            if is_multi:
                if set(gt_val) == set(pr_val):
                    correct[aen] += 1
            else:
                if gt_val == pr_val:
                    correct[aen] += 1
    n = len(texts)
    for aen, cnt in correct.items():
        print(f"  {aen:<15}: {cnt}/{n} = {cnt/n:.2%}")


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate_model(clf: BaseMLClassifier, test_path: Path, axes: list[dict]) -> dict:
    print(f"[INFO] Evaluating {clf.name} on {test_path} ...", file=sys.stderr)
    stories, gt = load_labeled_stories(test_path)

    model_preds: dict[str, dict] = {}
    for s in stories:
        model_preds[s["id"]] = clf.predict_one(s["text"], axes)

    metrics = compute_metrics(model_preds, gt, axes)
    return metrics


def print_metrics(metrics: dict, axes: list[dict]) -> None:
    axis_labels = [a["metadata_en"] for a in axes] + ["Overall"]
    col_w = 10
    header = f"{'Axis':<18}" + "".join(f"{'Acc':>{col_w}}{'Prec':>{col_w}}{'Rec':>{col_w}}{'F1':>{col_w}}")
    print("\n── Evaluation Metrics ──")
    print(header)
    print("─" * len(header))
    for aen in axis_labels:
        m = metrics.get(aen, {})
        if aen == "Overall":
            print(f"{'Overall':<18}" + " " * (col_w * 3) + f"{m.get('f1', 0):{col_w}.3f}")
        else:
            print(
                f"{aen:<18}"
                f"{m.get('accuracy', 0):{col_w}.3f}"
                f"{m.get('precision', 0):{col_w}.3f}"
                f"{m.get('recall', 0):{col_w}.3f}"
                f"{m.get('f1', 0):{col_w}.3f}"
            )


# ── Infer ────────────────────────────────────────────────────────────────────

def infer_one(clf: BaseMLClassifier, text: str, axes: list[dict]) -> dict:
    return clf.predict_one(text, axes)
