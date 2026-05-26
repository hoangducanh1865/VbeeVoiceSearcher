import json
import sys
from pathlib import Path
from typing import List

from src.label_model.config import MAX_LENGTH


def load_stories(path: Path) -> List[dict]:
    """Load user_input.jsonl → list of {"id": ..., "text": ...} dicts."""
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] Input JSONL not found: {path}", file=sys.stderr)
        sys.exit(1)
    stories = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" not in obj or "text" not in obj:
                    print(f"[WARNING] Line {lineno} missing 'id' or 'text', skipping.", file=sys.stderr)
                    continue
                stories.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Malformed JSON at line {lineno}: {e}", file=sys.stderr)
    if not stories:
        print("[ERROR] No valid stories loaded.", file=sys.stderr)
        sys.exit(1)
    return stories


def load_ground_truth(path: Path) -> dict:
    """Load user_input_label.jsonl → {story_id: {axis_en: value}} dict."""
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] Label JSONL not found: {path}", file=sys.stderr)
        sys.exit(1)
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story_id = obj.pop("id")
            gt[story_id] = obj
    return gt


def load_text(path: str | Path) -> str:
    """Read a UTF-8 text file and return its content as a single string."""
    path = Path(path)
    if not path.exists():
        print(f"[WARNING] Input file not found: {path}", file=sys.stderr)
        return ""

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        print(f"[WARNING] Input file is empty: {path}", file=sys.stderr)
        return ""

    # Rough char estimate — tokenizer will truncate at token level
    if len(text) > MAX_LENGTH * 4:
        print(
            f"[INFO] Text is long ({len(text)} chars). "
            f"Tokenizer will truncate to {MAX_LENGTH} tokens.",
            file=sys.stderr,
        )

    return text
