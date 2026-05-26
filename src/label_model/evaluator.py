"""
Compute accuracy / precision / recall / F1 for each axis and overall,
comparing model predictions against ground truth labels.

Single-label axes: sklearn accuracy + macro-F1.
Multi-label axis (Style): binary-vector macro precision/recall/F1.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def _extract_single(
    preds_per_story: Dict[str, str],
    gt_per_story: Dict[str, str],
    story_ids: List[str],
) -> Tuple[List, List]:
    y_true = [gt_per_story.get(sid, "") for sid in story_ids]
    y_pred = [preds_per_story.get(sid, "") for sid in story_ids]
    return y_true, y_pred


def _extract_multi(
    preds_per_story: Dict[str, List[str]],
    gt_per_story: Dict[str, List[str]],
    story_ids: List[str],
    all_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    label_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    n = len(story_ids)
    k = len(all_labels)
    y_true = np.zeros((n, k), dtype=int)
    y_pred = np.zeros((n, k), dtype=int)
    for i, sid in enumerate(story_ids):
        for lbl in gt_per_story.get(sid, []):
            if lbl in label_idx:
                y_true[i, label_idx[lbl]] = 1
        for lbl in preds_per_story.get(sid, []):
            if lbl in label_idx:
                y_pred[i, label_idx[lbl]] = 1
    return y_true, y_pred


def compute_metrics(
    model_preds: Dict[str, Dict],
    ground_truth: Dict[str, Dict],
    axes: List[dict],
) -> Dict[str, Dict]:
    """
    Args:
        model_preds: {story_id: {axis_en: predicted_value_or_list}}
        ground_truth: {story_id: {axis_en: gt_value_or_list}}
        axes: list of axis defs from metadata

    Returns:
        {axis_en: {accuracy, precision, recall, f1}, ..., "Overall": {f1}}
    """
    story_ids = sorted(ground_truth.keys())
    metrics: Dict[str, Dict] = {}
    f1_scores = []

    for axis in axes:
        axis_en = axis["metadata_en"]
        is_multi = axis["loai_nhan"] == "Đa nhãn"

        if is_multi:
            all_labels = axis["cac_gia_tri"]
            gt_map  = {sid: ground_truth[sid].get(axis_en, []) for sid in story_ids}
            pred_map = {sid: model_preds.get(sid, {}).get(axis_en, []) for sid in story_ids}
            y_true, y_pred = _extract_multi(pred_map, gt_map, story_ids, all_labels)

            # Per-story exact match accuracy
            exact = float(np.mean([
                set(ground_truth[sid].get(axis_en, [])) == set(model_preds.get(sid, {}).get(axis_en, []))
                for sid in story_ids
            ]))
            prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        else:
            gt_map   = {sid: ground_truth[sid].get(axis_en, "") for sid in story_ids}
            pred_map = {sid: model_preds.get(sid, {}).get(axis_en, "") for sid in story_ids}
            y_true, y_pred = _extract_single(pred_map, gt_map, story_ids)

            exact = float(accuracy_score(y_true, y_pred))
            prec  = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            rec   = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            f1    = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        metrics[axis_en] = {"accuracy": exact, "precision": prec, "recall": rec, "f1": f1}
        f1_scores.append(f1)

    metrics["Overall"] = {"f1": float(np.mean(f1_scores))}
    return metrics
