"""
Generate a synthetic Vietnamese TTS annotation dataset using Gemini API.

Usage:
    python -m src.dataset.generator
    python -m src.dataset.generator --n 300 --out data/dataset

Outputs:
    data/dataset/train.jsonl  (N * 0.8 samples)
    data/dataset/test.jsonl   (N * 0.2 samples)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

# ── Constants ────────────────────────────────────────────────────────────────
N_SAMPLES    = 300
TRAIN_RATIO  = 0.8
GEMINI_MODEL = "gemini-1.5-flash"
RETRY_LIMIT  = 3
RETRY_DELAY  = 5   # seconds between retries

METADATA_PATH = Path("data/meta_data/meta_data.jsonl")
DATASET_DIR   = Path("data/dataset")


# ── Gemini setup ─────────────────────────────────────────────────────────────

def _init_gemini() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        print("[ERROR] GEMINI_API not found in environment / .env", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


# ── Metadata ─────────────────────────────────────────────────────────────────

def _load_axes(path: Path) -> list[dict]:
    axes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                axes.append(json.loads(line))
    return axes


# ── Label combination builder ────────────────────────────────────────────────

# Semantic affinity map: which Style labels fit which (Valence, Energy) combos
_STYLE_AFFINITY: dict[tuple, list[str]] = {
    # (Valence_short, Energy_short) → preferred Style labels
    ("Tiêu cực", "Nhẹ nhàng"): ["Sâu lắng", "Buồn bã", "Ôn hoà"],
    ("Tiêu cực", "Trung bình"): ["Buồn bã", "Ôn hoà", "Truyền cảm hứng"],
    ("Tiêu cực", "Mạnh mẽ"):   ["Dồn dập", "Gấp gáp", "Tự tin"],
    ("Trung tính", "Nhẹ nhàng"):["Sâu lắng", "Ôn hoà", "Ngọt ngào"],
    ("Trung tính", "Trung bình"):["Ôn hoà", "Tự tin", "Sâu lắng"],
    ("Trung tính", "Mạnh mẽ"):  ["Tự tin", "Dồn dập", "Gấp gáp"],
    ("Tích cực", "Nhẹ nhàng"):  ["Ngọt ngào", "Ôn hoà", "Sâu lắng"],
    ("Tích cực", "Trung bình"): ["Vui tươi", "Ngọt ngào", "Sôi nổi"],
    ("Tích cực", "Mạnh mẽ"):    ["Sôi nổi", "Truyền cảm hứng", "Vui tươi"],
}

def _valence_short(v: str) -> str:
    if "Tích cực" in v:
        return "Tích cực"
    if "Tiêu cực" in v:
        return "Tiêu cực"
    return "Trung tính"

def _energy_short(e: str) -> str:
    if "low" in e or "Nhẹ" in e:
        return "Nhẹ nhàng"
    if "high" in e or "Mạnh" in e:
        return "Mạnh mẽ"
    return "Trung bình"


def build_label_combinations(axes: list[dict], n: int) -> list[dict]:
    """
    Build n label combos systematically covering all axis values,
    using semantic affinity for Style label selection.
    """
    intonation_vals = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Intonation")
    valence_vals    = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Valence")
    energy_vals     = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Energy")
    temporal_vals   = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Temporal")
    style_vals      = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Style")

    # Build exhaustive base (135 combos = 5×3×3×3)
    base: list[dict] = []
    for inton in intonation_vals:
        for val in valence_vals:
            for eng in energy_vals:
                for temp in temporal_vals:
                    vk = _valence_short(val)
                    ek = _energy_short(eng)
                    key = (vk, ek)
                    preferred = _STYLE_AFFINITY.get(key, style_vals)
                    # primary style from affinity, then pad from full list
                    pool = preferred + [s for s in style_vals if s not in preferred]
                    styles = pool[:3]
                    base.append({
                        "Intonation": inton,
                        "Valence": val,
                        "Energy": eng,
                        "Temporal": temp,
                        "Style": styles,
                    })

    random.shuffle(base)

    # Repeat + randomise Style to reach n
    combos: list[dict] = []
    while len(combos) < n:
        for c in base:
            if len(combos) >= n:
                break
            vk = _valence_short(c["Valence"])
            ek = _energy_short(c["Energy"])
            key = (vk, ek)
            preferred = _STYLE_AFFINITY.get(key, style_vals)
            pool = preferred + [s for s in style_vals if s not in preferred]
            # Introduce variation: occasionally pick different Style combo
            if len(combos) < len(base):
                styles = pool[:3]
            else:
                extra = [s for s in style_vals if s not in preferred[:2]]
                styles = preferred[:1] + random.sample(extra, min(2, len(extra)))
                if len(styles) < 3:
                    styles = (styles + pool)[:3]
            combos.append({**c, "Style": styles})

    random.shuffle(combos)
    return combos[:n]


# ── Gemini story generation ───────────────────────────────────────────────────

def _build_prompt(combo: dict) -> str:
    styles = ", ".join(combo["Style"])
    return (
        "Viết một đoạn văn bản tiếng Việt khoảng 3-5 câu, "
        "phù hợp để đọc với thông số TTS sau:\n"
        f"- Ngữ điệu: {combo['Intonation']}\n"
        f"- Cảm xúc: {combo['Valence']}\n"
        f"- Năng lượng: {combo['Energy']}\n"
        f"- Tốc độ đọc: {combo['Temporal']}\n"
        f"- Phong cách: {styles}\n\n"
        "Đoạn văn có thể là trích truyện, kịch bản, tin tức, phát biểu, thơ, v.v. "
        "Chỉ trả về nội dung đoạn văn, không giải thích."
    )


def generate_story(model: genai.GenerativeModel, combo: dict) -> str | None:
    prompt = _build_prompt(combo)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            if text:
                return text
        except Exception as e:
            print(f"  [WARN] Attempt {attempt}/{RETRY_LIMIT} failed: {e}", file=sys.stderr)
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    return None


# ── Main generation loop ─────────────────────────────────────────────────────

def generate_dataset(n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = _load_axes(METADATA_PATH)
    model = _init_gemini()

    combos = build_label_combinations(axes, n)
    print(f"[INFO] Generating {n} stories with Gemini ({GEMINI_MODEL}) ...", file=sys.stderr)

    records: list[dict] = []
    raw_path = out_dir / "raw.jsonl"
    raw_path.unlink(missing_ok=True)

    for i, combo in enumerate(combos, 1):
        text = generate_story(model, combo)
        if text is None:
            print(f"  [SKIP] story_{i:03d} — Gemini returned nothing", file=sys.stderr)
            continue

        record = {"id": f"gen_{i:03d}", "text": text, **combo}
        records.append(record)
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if i % 10 == 0 or i == n:
            print(f"[INFO] Generated {i}/{n} ...", file=sys.stderr)

        # Polite rate-limit: ~0.3 s between calls (free tier = 15 RPM = 1/4s)
        time.sleep(0.35)

    # Shuffle and split
    random.shuffle(records)
    split = int(len(records) * TRAIN_RATIO)
    train_records = records[:split]
    test_records  = records[split:]

    for path, data in [(out_dir / "train.jsonl", train_records),
                       (out_dir / "test.jsonl",  test_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[INFO] Done. {len(train_records)} train / {len(test_records)} test "
        f"→ {out_dir}/",
        file=sys.stderr,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Vietnamese TTS annotation dataset via Gemini")
    parser.add_argument("--n",   type=int, default=N_SAMPLES, help=f"Number of samples (default: {N_SAMPLES})")
    parser.add_argument("--out", type=str, default=str(DATASET_DIR), help="Output directory")
    args = parser.parse_args()
    generate_dataset(args.n, Path(args.out))


if __name__ == "__main__":
    main()
