import json
import sys
from pathlib import Path
from typing import List


def load_axes(path: Path) -> List[dict]:
    """Parse meta_data.jsonl and return list of axis definitions."""
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] Metadata file not found: {path}", file=sys.stderr)
        sys.exit(1)

    axes = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                axes.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping malformed JSON at line {lineno}: {e}", file=sys.stderr)

    if not axes:
        print("[ERROR] No axes loaded from metadata file.", file=sys.stderr)
        sys.exit(1)

    return axes
