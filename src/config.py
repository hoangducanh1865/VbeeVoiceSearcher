import os
import torch
from pathlib import Path

# ── HF_TOKEN: auto-load from Google Colab secrets if available ──────────────
try:
    from google.colab import userdata
    _token = userdata.get('HF_TOKEN')
    if _token:
        os.environ['HF_TOKEN'] = _token
except Exception:
    pass  # Not in Colab or secret not set — falls back to env var / cached login

# ── Model for single-run (main.py) ──────────────────────────────────────────
# facebook/xlm-roberta-large is the base MLM — no NLI head.
# joeddav/xlm-roberta-large-xnli IS the same architecture but fine-tuned on XNLI,
# enabling true zero-shot across 15+ languages including Vietnamese.
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

# ── Models for multi-model comparison (experiment/compare_models.py) ─────────
MODELS = [
    {
        "id": "joeddav/xlm-roberta-large-xnli",
        "short_name": "XLM-R-Large",
        "note": "XLM-RoBERTa-Large fine-tuned on XNLI, ~1.1GB",
    },
    {
        "id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "short_name": "mDeBERTa-v3",
        "note": "Multilingual DeBERTa-v3 Base, MNLI+XNLI, ~280MB",
    },
    {
        "id": "typeform/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "short_name": "mDeBERTa-2M7",
        "note": "mDeBERTa-v3 Base fine-tuned on 2.7M multilingual NLI pairs, ~280MB",
    },
]

# ── Paths ────────────────────────────────────────────────────────────────────
METADATA_PATH = Path("data/meta_data/meta_data.jsonl")
OUTPUT_FILE = Path("data/input_label.jsonl")

# ── NLI hypothesis template (Vietnamese) ─────────────────────────────────────
HYPOTHESIS_TEMPLATE = "Đoạn văn này thể hiện {}."

# ── Thresholds ───────────────────────────────────────────────────────────────
MULTI_LABEL_THRESHOLD = 0.50   # confidence cutoff for Style (multi-label)
MULTI_LABEL_TOP_K = 3          # fallback: max labels if none clear threshold

MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
