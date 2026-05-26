import json
from typing import List

from tabulate import tabulate

from src.label_model.classifier import PredResult


def render(results: List[PredResult]) -> str:
    """Format prediction results into a Vietnamese-labelled table string."""
    rows = []
    for result in results:
        axis_vi = result.axis["metadata_vi"]
        loai_nhan = result.axis["loai_nhan"]
        labels = ", ".join(label for label, _ in result.predictions)
        confidences = ", ".join(f"{score:.2f}" for _, score in result.predictions)
        warning = " ⚠️" if result.has_warning else ""
        rows.append([axis_vi + warning, loai_nhan, labels, confidences])

    headers = ["Trục (VI)", "Loại nhãn", "Giá trị dự đoán", "Confidence"]
    return tabulate(rows, headers=headers, tablefmt="github")


def to_jsonl_lines(results: List[PredResult], source: str) -> str:
    """Serialize predictions as JSONL — one line per axis."""
    lines = []
    for result in results:
        is_multi = result.axis["loai_nhan"] == "Đa nhãn"
        if is_multi:
            gia_tri = [l for l, _ in result.predictions]
            confidence = [round(s, 4) for _, s in result.predictions]
        else:
            gia_tri = result.predictions[0][0]
            confidence = round(result.predictions[0][1], 4)

        record = {
            "source": source,
            "axis_en": result.axis["metadata_en"],
            "axis_vi": result.axis["metadata_vi"],
            "loai_nhan": result.axis["loai_nhan"],
            "gia_tri": gia_tri,
            "confidence": confidence,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)
