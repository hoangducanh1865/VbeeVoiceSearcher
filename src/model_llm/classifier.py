import gc
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Import config first so huggingface_hub.login() runs before transformers is loaded
from src.model_llm.config import (
    DEVICE,
    HYPOTHESIS_TEMPLATE,
    MODEL_NAME,
    MULTI_LABEL_THRESHOLD,
    MULTI_LABEL_TOP_K,
)

import torch
from transformers import pipeline


@dataclass
class PredResult:
    axis: dict
    predictions: List[Tuple[str, float]] = field(default_factory=list)
    all_scores: Dict[str, float] = field(default_factory=dict)  # full score map for all candidates
    has_warning: bool = False


class TextClassifier:
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        device_id = 0 if device == "cuda" else -1
        print(f"[INFO] Loading model '{model_name}' on {device} ...", file=sys.stderr)
        try:
            self.pipe = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device_id,
            )
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
            sys.exit(1)
        print("[INFO] Model loaded.", file=sys.stderr)

    def clear(self) -> None:
        """Remove model from GPU memory and clear CUDA cache."""
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] Model cleared from memory.", file=sys.stderr)

    def predict(self, text: str, axes: List[dict]) -> List[PredResult]:
        if not text:
            print("[WARNING] Empty text — returning empty predictions.", file=sys.stderr)
            return [PredResult(axis=ax, predictions=[("", 0.0)]) for ax in axes]

        results = []
        for axis in axes:
            labels: List[str] = axis["cac_gia_tri"]
            is_multi: bool = axis["loai_nhan"] == "Đa nhãn"

            out = self.pipe(
                text,
                candidate_labels=labels,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                multi_label=is_multi,
            )
            scored = list(zip(out["labels"], out["scores"]))
            all_scores = {label: score for label, score in scored}

            if is_multi:
                picked = [(l, s) for l, s in scored if s >= MULTI_LABEL_THRESHOLD]
                if not picked:
                    picked = scored[:MULTI_LABEL_TOP_K]
            else:
                picked = [scored[0]]

            results.append(PredResult(axis=axis, predictions=picked, all_scores=all_scores))

        return results
