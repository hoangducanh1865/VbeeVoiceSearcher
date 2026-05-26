import sys
from dataclasses import dataclass, field
from typing import List, Tuple

from transformers import pipeline

from src.config import (
    DEVICE,
    HYPOTHESIS_TEMPLATE,
    MODEL_NAME,
    MULTI_LABEL_THRESHOLD,
    MULTI_LABEL_TOP_K,
)


@dataclass
class PredResult:
    axis: dict
    predictions: List[Tuple[str, float]] = field(default_factory=list)
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
            # pipeline returns labels sorted by score descending
            scored = list(zip(out["labels"], out["scores"]))

            if is_multi:
                picked = [(l, s) for l, s in scored if s >= MULTI_LABEL_THRESHOLD]
                # Fallback: if nothing clears threshold, take top-k
                if not picked:
                    picked = scored[:MULTI_LABEL_TOP_K]
            else:
                picked = [scored[0]]

            results.append(PredResult(axis=axis, predictions=picked))

        return results
