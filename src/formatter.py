from typing import List

from tabulate import tabulate

from src.classifier import PredResult


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
