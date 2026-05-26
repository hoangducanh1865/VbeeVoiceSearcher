import torch
from pathlib import Path

# Model: XLM-RoBERTa-Large fine-tuned on XNLI for zero-shot classification
# facebook/xlm-roberta-large is the base MLM — cannot do NLI without a classification head.
# joeddav/xlm-roberta-large-xnli IS the same architecture but fine-tuned on XNLI,
# enabling true zero-shot across 15+ languages including Vietnamese.
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

METADATA_PATH = Path("data/meta_data/meta_data.jsonl")
OUTPUT_FILE = Path("data/input_label.jsonl")

# NLI hypothesis template in Vietnamese — "{}" is replaced by each candidate label
HYPOTHESIS_TEMPLATE = "Đoạn văn này thể hiện {}."

# Confidence threshold for multi-label (Style) axis
MULTI_LABEL_THRESHOLD = 0.50

# Fallback: max labels to keep if none exceed threshold
MULTI_LABEL_TOP_K = 3

MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
