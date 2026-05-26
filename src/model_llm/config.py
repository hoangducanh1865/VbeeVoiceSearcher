import os
import torch
from pathlib import Path

# ── HF_TOKEN: auto-load from Google Colab secrets if available ──────────────
# Must call huggingface_hub.login() — setting os.environ alone is not enough
# because huggingface_hub registers auth state internally at session level.
try:
    from google.colab import userdata
    import huggingface_hub
    _token = userdata.get('HF_TOKEN')
    if _token:
        huggingface_hub.login(token=_token, add_to_git_credential=False)
except Exception:
    pass  # Not in Colab or secret not set — falls back to cached ~/.cache/huggingface/token

# ── Model for single-run (main.py) ──────────────────────────────────────────
# facebook/xlm-roberta-large is the base MLM — no NLI head.
# joeddav/xlm-roberta-large-xnli IS the same architecture but fine-tuned on XNLI,
# enabling true zero-shot across 15+ languages including Vietnamese.
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

# ── Models for multi-model comparison (experiment/compare_models.py) ─────────
MODELS = [
    {
        "id": "joeddav/xlm-roberta-large-xnli",
        "short_name": "XLM-R-L",
        "note": "XLM-RoBERTa-Large + XNLI, multilingual, ~1.1GB",
    },
    {
        "id": "symanto/xlm-roberta-base-snli-mnli-anli-xnli",
        "short_name": "XLM-R-B",
        "note": "XLM-RoBERTa-Base, SNLI+MNLI+ANLI+XNLI, multilingual, ~500MB",
    },
    {
        "id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "short_name": "mDeBERTa-v3",
        "note": "Multilingual DeBERTa-v3 Base, MNLI+XNLI, ~280MB",
    },
    {
        "id": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "short_name": "mDeBERTa-2M7",
        "note": "mDeBERTa-v3 Base, 2.7M multilingual NLI pairs, ~280MB",
    },
    {
        "id": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        "short_name": "MiniLM-L6",
        "note": "Multilingual MiniLM-v2 L6, MNLI+XNLI, ~120MB",
    },
    {
        "id": "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli",
        "short_name": "MiniLM-L12",
        "note": "Multilingual MiniLM-v2 L12, MNLI+XNLI, ~230MB",
    },
    {
        "id": "cross-encoder/nli-deberta-v3-large",
        "short_name": "DeBERTa-v3-L",
        "note": "DeBERTa-v3-Large NLI, English-focused, ~900MB",
    },
]

# ── Paths ────────────────────────────────────────────────────────────────────
METADATA_PATH = Path("data/meta_data/meta_data.jsonl")
OUTPUT_FILE   = Path("data/input_label.jsonl")   # legacy single-run output
INPUT_JSONL   = Path("data/user_input/user_input.jsonl")
PREDICT_DIR   = Path("data/user_input_predict")
TEST_JSONL    = Path("data/dataset/test.jsonl")
ML_MODEL_DIR  = Path("data/ml_model")

# ── NLI hypothesis template (Vietnamese) ─────────────────────────────────────
HYPOTHESIS_TEMPLATE = "Đoạn văn này thể hiện {}."

# ── Thresholds ───────────────────────────────────────────────────────────────
MULTI_LABEL_THRESHOLD = 0.50   # confidence cutoff for Style (multi-label)
MULTI_LABEL_TOP_K = 3          # fallback: max labels if none clear threshold

MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
