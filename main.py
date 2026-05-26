import argparse
import sys
from pathlib import Path

from src.classifier import TextClassifier
from src.config import METADATA_PATH, OUTPUT_FILE
from src.formatter import render, to_jsonl_lines
from src.metadata_loader import load_axes
from src.text_loader import load_text
from src.validator import validate


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Vietnamese TTS annotation classifier"
    )
    parser.add_argument("input_file", help="Path to Vietnamese text file (.txt)")
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name (default: joeddav/xlm-roberta-large-xnli)",
    )
    args = parser.parse_args()

    axes = load_axes(METADATA_PATH)
    text = load_text(args.input_file)

    if not text:
        print("[ERROR] No text to classify.", file=sys.stderr)
        sys.exit(1)

    clf_kwargs = {}
    if args.model:
        clf_kwargs["model_name"] = args.model

    clf = TextClassifier(**clf_kwargs)
    results = clf.predict(text, axes)
    results = validate(results)

    print(render(results))

    source = Path(args.input_file).name        # "input1.txt"
    jsonl = to_jsonl_lines(results, source)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(jsonl + "\n")
    print(f"[INFO] Appended to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
