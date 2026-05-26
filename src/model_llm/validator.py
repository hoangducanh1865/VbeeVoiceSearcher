import sys
from typing import List

from src.model_llm.classifier import PredResult


def validate(results: List[PredResult]) -> List[PredResult]:
    """Check each predicted label exists in the axis's allowed values. Flag unknowns."""
    for result in results:
        allowed = set(result.axis["cac_gia_tri"])
        axis_vi = result.axis["metadata_vi"]
        for label, _ in result.predictions:
            if label and label not in allowed:
                print(
                    f"⚠️  WARNING: \"{label}\" not in allowed values for [{axis_vi}]",
                    file=sys.stderr,
                )
                result.has_warning = True
    return results
