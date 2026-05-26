import argparse
import json
import sys
from typing import List

from src.label_model.classifier import PredResult, TextClassifier
from src.label_model.config import INPUT_JSONL, METADATA_PATH, MODELS, MODEL_NAME, PREDICT_DIR
from src.label_model.formatter import render
from src.label_model.metadata_loader import load_axes
from src.label_model.text_loader import load_stories
from src.label_model.validator import validate


def _results_to_pred_dict(results: List[PredResult]) -> dict:
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


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Vietnamese TTS annotation classifier (batch)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"HuggingFace model name (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    model_id = args.model or MODEL_NAME
    short_name = _model_short_name(model_id)

    axes    = load_axes(METADATA_PATH)
    stories = load_stories(INPUT_JSONL)

    print(f"[INFO] Model: {model_id}  ({short_name})", file=sys.stderr)
    print(f"[INFO] Processing {len(stories)} stories ...", file=sys.stderr)

    clf = TextClassifier(model_name=model_id)

    out_dir  = PREDICT_DIR / short_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "user_input_predict.jsonl"

    for story in stories:
        sid  = story["id"]
        text = story["text"]

        results  = clf.predict(text, axes)
        results  = validate(results)

        print(f"\n── {sid} ──")
        print(render(results))

        record = {"id": sid, "model": short_name}
        record.update(_results_to_pred_dict(results))
        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Predictions saved to {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
