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
GEMINI_MODEL = "gemini-2.5-flash"
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
    if "Nhẹ nhàng" in e:
        return "Nhẹ nhàng"
    if "Mạnh mẽ" in e:
        return "Mạnh mẽ"
    return "Trung bình"


def build_label_combinations(axes: list[dict], n: int) -> list[dict]:
    """
    Build n label combos systematically covering all axis values,
    using semantic affinity for Style label selection.
    Increased randomization to avoid repetitive Style assignments.
    """
    intonation_vals = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Intonation")
    valence_vals    = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Valence")
    energy_vals     = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Energy")
    temporal_vals   = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Temporal")
    style_vals      = next(a["cac_gia_tri"] for a in axes if a["metadata_en"] == "Style")

    # Build base combos: 5 Intonation × 3 Valence × 3 Energy × 4 Temporal = 180 combos
    base: list[dict] = []
    for inton in intonation_vals:
        for val in valence_vals:
            for eng in energy_vals:
                for temp in temporal_vals:
                    vk = _valence_short(val)
                    ek = _energy_short(eng)
                    key = (vk, ek)
                    
                    preferred = _STYLE_AFFINITY.get(key, style_vals)
                    
                    # INCREASED RANDOMIZATION:
                    # Sample 2 from preferred styles, then pick 1 from remaining
                    chosen_styles = random.sample(preferred, min(2, len(preferred)))
                    remaining = [s for s in style_vals if s not in chosen_styles]
                    chosen_styles.append(random.choice(remaining))
                    
                    base.append({
                        "Intonation": inton,
                        "Valence": val,
                        "Energy": eng,
                        "Temporal": temp,
                        "Style": chosen_styles,
                    })

    random.shuffle(base)

    # Repeat base to reach n combos, randomizing Style each time
    combos: list[dict] = []
    while len(combos) < n:
        for c in base:
            if len(combos) >= n:
                break
            
            vk = _valence_short(c["Valence"])
            ek = _energy_short(c["Energy"])
            key = (vk, ek)
            preferred = _STYLE_AFFINITY.get(key, style_vals)
            
            # Create different Style variations for diversity
            styles = random.sample(preferred, min(2, len(preferred)))
            extra = [s for s in style_vals if s not in styles]
            styles.append(random.choice(extra))
            
            combos.append({**c, "Style": styles})

    random.shuffle(combos)
    return combos[:n]


# ── Gemini story generation ───────────────────────────────────────────────────

def _build_prompt(combo: dict) -> str:
    """Build a prompt optimized for TTS data generation."""
    styles = ", ".join(combo["Style"])
    return (
        "Bạn là một chuyên gia biên kịch và ngôn ngữ học tiếng Việt.\n"
        "Hãy viết một đoạn văn bản tiếng Việt ngắn từ 3 đến 5 câu (tuyệt đối KHÔNG viết thơ, KHÔNG viết kịch bản có tên nhân vật phía trước).\n\n"
        "Văn bản này phải được viết bằng từ ngữ, cấu trúc câu và ngữ cảnh sao cho người nghe có thể cảm nhận rõ ràng các yếu tố sau khi đọc lên:\n"
        f"- Cảm xúc chủ đạo: {combo['Valence']}\n"
        f"- Năng lượng/Sắc thái: {combo['Energy']}\n"
        f"- Phong cách diễn đạt: {styles}\n"
        f"- Gợi ý nhịp điệu khi đọc: {combo['Temporal']} và có tông giọng {combo['Intonation']}\n\n"
        "QUY ĐỊNH NGHIÊM NGẶT VỀ ĐỊNH DẠNG:\n"
        "1. CHỈ trả về duy nhất nội dung đoạn văn bằng tiếng Việt chuẩn. Không có lời mở đầu, không giải thích, không ghi chú.\n"
        "2. KHÔNG sử dụng các ký tự đặc biệt như: *, #, _, ~, [], (), <>, -, tránh lạm dụng dấu ba chấm (...).\n"
        "3. Các câu phải có độ dài vừa phải, rõ ràng cấu trúc Chủ ngữ - Vị ngữ để người đọc dễ ngắt nghỉ tự nhiên."
    )


def _is_valid_text(text: str) -> bool:
    """Validate generated text for TTS suitability."""
    if not text or len(text) < 20:
        return False
    
    # Avoid special characters that break TTS
    forbidden_chars = ['*', '#', '_', '~', '[', ']', '<', '>', '«', '»']
    for char in forbidden_chars:
        if char in text:
            return False
    
    # Avoid excessive ellipsis or dashes
    if text.count('...') > 2 or text.count('---') > 0:
        return False
    
    # Avoid poetry-like structures (lines ending with common rhyme markers)
    lines = text.strip().split('\n')
    if len(lines) > 2 and all(len(line) < 15 for line in lines if line):
        # Likely verse/poetry format
        return False
    
    # Check for minimal sentence structure (Chủ ngữ - Vị ngữ)
    sentence_count = len([s for s in text.split('.') if s.strip()])
    if sentence_count < 2:
        return False
    
    return True


def generate_story(model: genai.GenerativeModel, combo: dict) -> str | None:
    prompt = _build_prompt(combo)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            if text and _is_valid_text(text):
                return text
            elif text:
                print(f"  [WARN] Text failed validation (attempt {attempt}/{RETRY_LIMIT})", file=sys.stderr)
        except Exception as e:
            print(f"  [WARN] Attempt {attempt}/{RETRY_LIMIT} failed: {e}", file=sys.stderr)
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    return None


def _save_train_test_split(records: dict[str, dict], out_dir: Path) -> None:
    """Shuffle all records and save to train/test files."""
    records_list = list(records.values())
    random.shuffle(records_list)
    split = int(len(records_list) * TRAIN_RATIO)
    train_records = records_list[:split]
    test_records  = records_list[split:]

    for path, data in [(out_dir / "train.jsonl", train_records),
                       (out_dir / "test.jsonl",  test_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Main generation loop ─────────────────────────────────────────────────────

def generate_dataset(n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = _load_axes(METADATA_PATH)
    model = _init_gemini()

    combos = build_label_combinations(axes, n)
    raw_path = out_dir / "raw.jsonl"

    # ── Load existing records (resume support) ──
    existing_records: dict[str, dict] = {}  # {id: record}
    if raw_path.exists():
        print(f"[INFO] Loading existing records from {raw_path} ...", file=sys.stderr)
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_records[rec["id"]] = rec
        print(f"[INFO] Found {len(existing_records)} existing records", file=sys.stderr)
    
    print(f"[INFO] Generating {n} stories with Gemini ({GEMINI_MODEL}) ...", file=sys.stderr)
    print(f"[INFO] Target: {n} / Already have: {len(existing_records)}", file=sys.stderr)

    # ── Generate missing stories ──
    start_idx = len(existing_records)
    for i, combo in enumerate(combos, 1):
        story_id = f"gen_{i:03d}"
        
        # Skip if already generated
        if story_id in existing_records:
            if i <= start_idx:
                continue
        
        text = generate_story(model, combo)
        if text is None:
            print(f"  [SKIP] {story_id} — Gemini returned nothing", file=sys.stderr)
            continue

        record = {"id": story_id, "text": text, **combo}
        existing_records[story_id] = record
        
        # Append to raw.jsonl
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Update train/test files every 10 records
        if len(existing_records) % 10 == 0 or i == n:
            _save_train_test_split(existing_records, out_dir)
            split = int(len(existing_records) * TRAIN_RATIO)
            print(f"[INFO] Generated {len(existing_records)}/{n} → {split} train / {len(existing_records) - split} test", file=sys.stderr)

        # Polite rate-limit: ~0.3 s between calls (free tier = 15 RPM = 1/4s)
        time.sleep(0.35)

    # Final split (in case not multiple of 10)
    if len(existing_records) % 10 != 0:
        _save_train_test_split(existing_records, out_dir)
    
    records_list = list(existing_records.values())
    split = int(len(records_list) * TRAIN_RATIO)
    print(
        f"[INFO] Done. {split} train / {len(records_list) - split} test "
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
