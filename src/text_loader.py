import sys
from pathlib import Path

from src.config import MAX_LENGTH


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
