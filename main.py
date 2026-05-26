import argparse
import json
import sys
from typing import List

from src.label_model.config import INPUT_JSONL, METADATA_PATH, MODELS, MODEL_NAME, PREDICT_DIR
from src.label_model.metadata_loader import load_axes
from src.label_model.text_loader import load_stories


# ── Shared helpers ───────────────────────────────────────────────────────────

def _results_to_pred_dict(results) -> dict:
    out = {}
    for r in results:
        axis_en = r.axis["metadata_en"]
        if r.axis["loai_nhan"] == "Đa nhãn":
            out[axis_en] = [lbl for lbl, _ in r.predictions]
        else:
            out[axis_en] = r.predictions[0][0] if r.predictions else ""
    return out


def _model_short_name(model_id: str) -> str:
    for cfg in MODELS:
        if cfg["id"] == model_id:
            return cfg["short_name"]
    return model_id.split("/")[-1]


def _render_ml_result(pred: dict, axes: list[dict]) -> None:
    """Print ML prediction as a simple table."""
    from tabulate import tabulate
    rows = []
    for axis in axes:
        aen = axis["metadata_en"]
        val = pred.get(aen, "")
        display = ", ".join(val) if isinstance(val, list) else val
        rows.append([axis["metadata_vi"], axis["loai_nhan"], display])
    print(tabulate(rows, headers=["Trục (VI)", "Loại nhãn", "Giá trị dự đoán"], tablefmt="github"))


# ── LLM mode ─────────────────────────────────────────────────────────────────

def run_llm(args, axes) -> None:
    from src.label_model.classifier import TextClassifier
    from src.label_model.formatter import render
    from src.label_model.validator import validate

    model_id   = args.model or MODEL_NAME
    short_name = _model_short_name(model_id)
    stories    = load_stories(INPUT_JSONL)

    print(f"[INFO] Model: {model_id}  ({short_name})", file=sys.stderr)
    print(f"[INFO] Processing {len(stories)} stories ...", file=sys.stderr)

    clf = TextClassifier(model_name=model_id)

    out_dir  = PREDICT_DIR / short_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "user_input_predict.jsonl"

    for story in stories:
        sid  = story["id"]
        text = story["text"]

        results = clf.predict(text, axes)
        results = validate(results)

        print(f"\n── {sid} ──")
        print(render(results))

        record = {"id": sid, "model": short_name}
        record.update(_results_to_pred_dict(results))
        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Predictions saved to {out_file}", file=sys.stderr)


# ── ML mode ──────────────────────────────────────────────────────────────────

def run_ml(args, axes) -> None:
    from src.ml_model.trainer import (
        get_classifier, load_classifier,
        train_model, evaluate_model, print_metrics, infer_one,
    )
    from src.label_model.config import TEST_JSONL

    action   = args.action
    ml_name  = args.ml_model

    if action is None:
        print("[ERROR] --action required for --mode ml  (train / evaluate / infer)", file=sys.stderr)
        sys.exit(1)

    if action == "train":
        from src.label_model.config import ML_MODEL_DIR
        from pathlib import Path
        train_path = Path("data/dataset/train.jsonl")
        clf = get_classifier(ml_name)
        train_model(clf, train_path, axes)

    elif action == "evaluate":
        clf = load_classifier(ml_name)
        metrics = evaluate_model(clf, TEST_JSONL, axes)
        print_metrics(metrics, axes)

    elif action == "infer":
        if not args.story:
            print("[ERROR] --story <story_id> required for --action infer", file=sys.stderr)
            sys.exit(1)
        stories = load_stories(INPUT_JSONL)
        match   = next((s for s in stories if s["id"] == args.story), None)
        if match is None:
            ids = [s["id"] for s in stories]
            print(f"[ERROR] Story '{args.story}' not found. Available: {ids}", file=sys.stderr)
            sys.exit(1)

        clf  = load_classifier(ml_name)
        pred = infer_one(clf, match["text"], axes)

        print(f"\n── {match['id']} (ML: {ml_name}) ──")
        _render_ml_result(pred, axes)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vietnamese TTS annotation classifier — LLM or ML mode"
    )
    parser.add_argument(
        "--mode", choices=["llm", "ml"], default="llm",
        help="Inference backend (default: llm)",
    )
    # LLM args
    parser.add_argument(
        "--model", default=None,
        help=f"HuggingFace model ID for LLM mode (default: {MODEL_NAME})",
    )
    # ML args
    parser.add_argument(
        "--action", choices=["train", "evaluate", "infer"], default=None,
        help="ML action: train | evaluate | infer",
    )
    parser.add_argument(
        "--ml-model", choices=["svm", "logistic_regression", "naive_bayes"], default="svm",
        help="ML classifier to use (default: svm)",
    )
    parser.add_argument(
        "--story", default=None,
        help="Story ID for ML infer mode (e.g. story_03)",
    )
    args = parser.parse_args()

    axes = load_axes(METADATA_PATH)

    if args.mode == "llm":
        run_llm(args, axes)
    else:
        run_ml(args, axes)


if __name__ == "__main__":
    main()
