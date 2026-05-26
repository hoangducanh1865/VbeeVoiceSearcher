"""
Compare multiple zero-shot models on the same Vietnamese input text.
Outputs a PNG chart to ./experiment/compare_<input_stem>.png

Usage:
    python experiment/compare_models.py ./data/user_input/input1.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import PredResult, TextClassifier
from src.config import DEVICE, METADATA_PATH, MODELS
from src.metadata_loader import load_axes
from src.text_loader import load_text


# ── Model runner ─────────────────────────────────────────────────────────────

def run_all_models(
    text: str, axes: List[dict]
) -> Dict[str, List[PredResult]]:
    """Load each model, predict, then clear GPU before loading the next."""
    all_results: Dict[str, List[PredResult]] = {}

    for cfg in MODELS:
        short = cfg["short_name"]
        clf = TextClassifier(model_name=cfg["id"], device=DEVICE)
        all_results[short] = clf.predict(text, axes)
        clf.clear()

    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

_LABEL_COLORS = plt.cm.Set2.colors  # type: ignore


def _bar_chart_single(ax, axis_def: dict, model_results: Dict[str, List[PredResult]], axis_idx: int):
    """Bar chart for one single-label axis: one bar per model."""
    model_names = list(model_results.keys())
    allowed = axis_def["cac_gia_tri"]
    label_to_color = {lbl: _LABEL_COLORS[i % len(_LABEL_COLORS)] for i, lbl in enumerate(allowed)}

    confidences, colors, pred_labels = [], [], []
    for model in model_names:
        result = model_results[model][axis_idx]
        label, conf = result.predictions[0]
        confidences.append(conf)
        colors.append(label_to_color.get(label, "gray"))
        pred_labels.append(label)

    x = np.arange(len(model_names))
    bar_w = min(0.55, 0.85 / len(model_names))
    bars = ax.bar(x, confidences, color=colors, width=bar_w, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=8, rotation=10, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Confidence", fontsize=8)
    ax.set_title(axis_def["metadata_vi"], fontsize=9, fontweight="bold", pad=6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.tick_params(axis="y", labelsize=7)

    for bar, label, conf in zip(bars, pred_labels, confidences):
        # Truncate long labels to fit
        short_label = label if len(label) <= 12 else label[:11] + "…"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            short_label,
            ha="center", va="bottom", fontsize=6.5, rotation=0,
        )

    # Legend mapping color → label
    patches = [
        plt.Rectangle((0, 0), 1, 1, color=label_to_color[lbl])  # type: ignore
        for lbl in allowed
    ]
    ax.legend(patches, allowed, fontsize=6, loc="upper right", framealpha=0.7)


def _heatmap_multi(ax, axis_def: dict, model_results: Dict[str, List[PredResult]], axis_idx: int):
    """Heatmap for multi-label Style axis: all labels × models."""
    model_names = list(model_results.keys())
    all_labels = axis_def["cac_gia_tri"]

    # Build matrix rows=labels, cols=models
    matrix = np.zeros((len(all_labels), len(model_names)))
    for m_idx, model in enumerate(model_names):
        result = model_results[model][axis_idx]
        for l_idx, label in enumerate(all_labels):
            matrix[l_idx, m_idx] = result.all_scores.get(label, 0.0)

    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=max(6, 9 - len(model_names)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_title(f"{axis_def['metadata_vi']}  ({axis_def['loai_nhan']})", fontsize=9, fontweight="bold", pad=6)

    # Annotate cells
    threshold = 0.50
    for i in range(len(all_labels)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            marker = "★ " if val >= threshold else ""
            ax.text(
                j, i,
                f"{marker}{val:.2f}",
                ha="center", va="center",
                fontsize=7.5,
                color="black" if val < 0.7 else "white",
                fontweight="bold" if val >= threshold else "normal",
            )

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Confidence")
    ax.text(
        1.12, -0.04, "★ ≥ 0.50",
        transform=ax.transAxes, fontsize=7, va="top", color="#c0392b",
    )


def plot_comparison(
    all_results: Dict[str, List[PredResult]],
    axes: List[dict],
    source: str,
    output_path: Path,
):
    single_axes = [ax for ax in axes if ax["loai_nhan"] == "Đơn nhãn"]
    multi_axes  = [ax for ax in axes if ax["loai_nhan"] == "Đa nhãn"]
    single_idxs = [axes.index(ax) for ax in single_axes]
    multi_idxs  = [axes.index(ax) for ax in multi_axes]

    n_single = len(single_axes)
    n_models = len(all_results)
    fig_w = max(5 * n_single, 2.5 * n_models * n_single // 2)
    fig_h = 10 + max(0, n_models - 4)   # taller heatmap cell labels if many models
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, n_single, height_ratios=[1, 1.6], hspace=0.55, wspace=0.35)

    # Top row: single-label bar charts
    for col, (axis_def, axis_idx) in enumerate(zip(single_axes, single_idxs)):
        ax = fig.add_subplot(gs[0, col])
        _bar_chart_single(ax, axis_def, all_results, axis_idx)

    # Bottom row: multi-label heatmap spanning full width
    if multi_axes:
        ax_bottom = fig.add_subplot(gs[1, :])
        _heatmap_multi(ax_bottom, multi_axes[0], all_results, multi_idxs[0])

    model_list = " | ".join(
        f"{cfg['short_name']} ({cfg['note']})" for cfg in MODELS
    )
    fig.suptitle(
        f"Model Comparison — Vietnamese TTS Annotation\nInput: {source}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.text(0.5, -0.01, model_list, ha="center", fontsize=6.5, color="gray")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved → {output_path}", file=sys.stderr)
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare zero-shot models on Vietnamese text")
    parser.add_argument("input_file", help="Path to Vietnamese text file (.txt)")
    args = parser.parse_args()

    axes = load_axes(METADATA_PATH)
    text = load_text(args.input_file)
    if not text:
        print("[ERROR] Empty text.", file=sys.stderr)
        sys.exit(1)

    all_results = run_all_models(text, axes)

    out_dir = Path("experiment")
    out_dir.mkdir(exist_ok=True)
    stem = Path(args.input_file).stem
    output_path = out_dir / f"compare_{stem}.png"

    plot_comparison(all_results, axes, Path(args.input_file).name, output_path)


if __name__ == "__main__":
    main()
