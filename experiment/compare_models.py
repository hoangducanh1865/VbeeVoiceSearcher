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


def _grouped_bar_multi(ax, axis_def: dict, model_results: Dict[str, List[PredResult]], axis_idx: int):
    """Grouped bar chart for multi-label axis: x=style labels, one bar series per model."""
    model_names = list(model_results.keys())
    all_labels = axis_def["cac_gia_tri"]
    n_models = len(model_names)
    n_labels = len(all_labels)

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))  # type: ignore
    bar_w = 0.8 / n_models
    x = np.arange(n_labels)

    for m_idx, (model_name, color) in enumerate(zip(model_names, colors)):
        result = model_results[model_name][axis_idx]
        scores = [result.all_scores.get(lbl, 0.0) for lbl in all_labels]
        offset = (m_idx - n_models / 2 + 0.5) * bar_w
        ax.bar(
            x + offset, scores,
            width=bar_w * 0.92, color=color,
            label=model_name, edgecolor="white", linewidth=0.4, alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Confidence", fontsize=9)
    ax.set_title(
        f"{axis_def['metadata_vi']}  ({axis_def['loai_nhan']})",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.axhline(0.50, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7, label="threshold = 0.50")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85, ncol=2)


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
    n_style_labels = len(multi_axes[0]["cac_gia_tri"]) if multi_axes else 0
    fig_w = max(5 * n_single, 2.2 * n_style_labels + n_models)
    fig_h = 11
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, n_single, height_ratios=[1, 1.2], hspace=0.6, wspace=0.35)

    # Top row: single-label bar charts
    for col, (axis_def, axis_idx) in enumerate(zip(single_axes, single_idxs)):
        ax = fig.add_subplot(gs[0, col])
        _bar_chart_single(ax, axis_def, all_results, axis_idx)

    # Bottom row: multi-label grouped bar chart spanning full width
    if multi_axes:
        ax_bottom = fig.add_subplot(gs[1, :])
        _grouped_bar_multi(ax_bottom, multi_axes[0], all_results, multi_idxs[0])

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
