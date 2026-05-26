"""
Compare LLM and/or ML models on the test set.

Usage:
    python experiment/compare_models.py                  # run both (default)
    python experiment/compare_models.py --mode llm       # → compare_input_llm.png
    python experiment/compare_models.py --mode ml        # → compare_input_ml.png
    python experiment/compare_models.py --sample gen_005 # LLM chart: specific story
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_llm.classifier import PredResult, TextClassifier
from src.model_llm.config import DEVICE, METADATA_PATH, MODELS, PREDICT_DIR, TEST_JSONL
from src.model_llm.evaluator import compute_metrics
from src.model_llm.metadata_loader import load_axes


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_test_set(path: Path) -> tuple:
    """
    Load dataset JSONL (text + labels in same file).
    Returns (stories, gt) where:
        stories = [{"id": ..., "text": ...}]
        gt      = {story_id: {axis_en: value}}
    """
    stories, gt = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stories.append({"id": obj["id"], "text": obj["text"]})
            gt[obj["id"]] = {k: v for k, v in obj.items() if k not in ("id", "text")}
    return stories, gt


def _results_to_pred_dict(results: List[PredResult]) -> Dict[str, object]:
    """Convert List[PredResult] → {axis_en: value_or_list} for one story."""
    out = {}
    for r in results:
        axis_en = r.axis["metadata_en"]
        if r.axis["loai_nhan"] == "Đa nhãn":
            out[axis_en] = [lbl for lbl, _ in r.predictions]
        else:
            out[axis_en] = r.predictions[0][0] if r.predictions else ""
    return out


def save_predictions(
    story_id: str,
    model_short: str,
    results: List[PredResult],
    predict_dir: Path,
) -> None:
    out_dir = predict_dir / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"id": story_id, "model": model_short}
    record.update(_results_to_pred_dict(results))
    out_file = out_dir / "user_input_predict.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Model runner ──────────────────────────────────────────────────────────────

def run_all_models(
    stories: List[dict],
    axes: List[dict],
) -> Dict[str, Dict[str, List[PredResult]]]:
    """
    Returns: {model_short_name: {story_id: List[PredResult]}}
    Also writes each prediction to PREDICT_DIR/<model_short>/user_input_predict.jsonl.
    """
    # Clear existing per-model predict files for a fresh run
    for cfg in MODELS:
        predict_file = PREDICT_DIR / cfg["short_name"] / "user_input_predict.jsonl"
        predict_file.unlink(missing_ok=True)

    all_results: Dict[str, Dict[str, List[PredResult]]] = {}

    for cfg in MODELS:
        short = cfg["short_name"]
        clf = TextClassifier(model_name=cfg["id"], device=DEVICE)
        model_results: Dict[str, List[PredResult]] = {}

        for story in stories:
            sid = story["id"]
            results = clf.predict(story["text"], axes)
            model_results[sid] = results
            save_predictions(sid, short, results, PREDICT_DIR)

        clf.clear()
        all_results[short] = model_results

    return all_results


# ── Plotting helpers ──────────────────────────────────────────────────────────

_LABEL_COLORS = plt.cm.Set2.colors  # type: ignore


def _bar_chart_single(
    ax, axis_def: dict,
    model_results: Dict[str, Dict[str, List[PredResult]]],
    sample_id: str, axis_idx: int,
):
    model_names = list(model_results.keys())
    allowed = axis_def["cac_gia_tri"]
    label_to_color = {lbl: _LABEL_COLORS[i % len(_LABEL_COLORS)] for i, lbl in enumerate(allowed)}

    confidences, colors, pred_labels = [], [], []
    for model in model_names:
        result = model_results[model][sample_id][axis_idx]
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

    for bar, label in zip(bars, pred_labels):
        short_label = label if len(label) <= 12 else label[:11] + "…"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            short_label, ha="center", va="bottom", fontsize=6.5,
        )
    patches = [plt.Rectangle((0, 0), 1, 1, color=label_to_color[lbl]) for lbl in allowed]  # type: ignore
    ax.legend(patches, allowed, fontsize=6, loc="upper right", framealpha=0.7)


def _grouped_bar_multi(
    ax, axis_def: dict,
    model_results: Dict[str, Dict[str, List[PredResult]]],
    sample_id: str, axis_idx: int,
):
    model_names = list(model_results.keys())
    all_labels = axis_def["cac_gia_tri"]
    n_models = len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))  # type: ignore
    bar_w = 0.8 / n_models
    x = np.arange(len(all_labels))

    for m_idx, (model_name, color) in enumerate(zip(model_names, colors)):
        result = model_results[model_name][sample_id][axis_idx]
        scores = [result.all_scores.get(lbl, 0.0) for lbl in all_labels]
        offset = (m_idx - n_models / 2 + 0.5) * bar_w
        ax.bar(x + offset, scores, width=bar_w * 0.92, color=color,
               label=model_name, edgecolor="white", linewidth=0.4, alpha=0.88)

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


def _top3_per_model_row(
    axes_row: list, axis_def: dict,
    model_results: Dict[str, Dict[str, List[PredResult]]],
    sample_id: str, axis_idx: int,
):
    model_names = list(model_results.keys())
    for ax, model_name in zip(axes_row, model_names):
        result = model_results[model_name][sample_id][axis_idx]
        top3 = sorted(result.all_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        labels = [item[0] for item in top3]
        scores = [item[1] for item in top3]

        bar_colors = plt.cm.RdYlGn(np.array(scores))  # type: ignore
        bars = ax.bar(range(3), scores, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=18, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(model_name, fontsize=8.5, fontweight="bold", pad=5)
        ax.axhline(0.50, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5, zorder=0)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                    f"{score:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")


def _metrics_heatmap(
    ax,
    metrics_all_models: Dict[str, Dict[str, Dict]],
    axes: List[dict],
):
    """Heatmap: rows=models, cols=axes+Overall, values=F1. Below: accuracy bar."""
    model_names = list(metrics_all_models.keys())
    axis_labels_en = [a["metadata_en"] for a in axes] + ["Overall"]
    axis_labels_vi = [a["metadata_vi"] for a in axes] + ["Tổng quát"]

    n_models = len(model_names)
    n_cols = len(axis_labels_en)

    matrix_f1  = np.zeros((n_models, n_cols))
    matrix_acc = np.zeros((n_models, n_cols))

    for m_i, model in enumerate(model_names):
        for c_i, aen in enumerate(axis_labels_en):
            m = metrics_all_models[model].get(aen, {})
            matrix_f1[m_i, c_i]  = m.get("f1", 0.0)
            matrix_acc[m_i, c_i] = m.get("accuracy", m.get("f1", 0.0))

    im = ax.imshow(matrix_f1, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(axis_labels_vi, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("Macro-F1 per model per axis  (test set)", fontsize=10, fontweight="bold", pad=8)

    for m_i in range(n_models):
        for c_i in range(n_cols):
            f1  = matrix_f1[m_i, c_i]
            acc = matrix_acc[m_i, c_i]
            aen = axis_labels_en[c_i]
            show_acc = aen != "Overall"
            cell_text = f"F1={f1:.2f}\nAcc={acc:.2f}" if show_acc else f"F1={f1:.2f}"
            ax.text(
                c_i, m_i, cell_text,
                ha="center", va="center", fontsize=7,
                color="black" if f1 < 0.65 else "white",
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="F1 score")

    # Vertical separator before Overall column
    ax.axvline(n_cols - 1.5, color="white", linewidth=2)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_comparison(
    all_results: Dict[str, Dict[str, List[PredResult]]],
    metrics_all_models: Dict[str, Dict[str, Dict]],
    axes: List[dict],
    sample_id: str,
    source_label: str,
    output_path: Path,
):
    single_axes = [a for a in axes if a["loai_nhan"] == "Đơn nhãn"]
    multi_axes  = [a for a in axes if a["loai_nhan"] == "Đa nhãn"]
    single_idxs = [axes.index(a) for a in single_axes]
    multi_idxs  = [axes.index(a) for a in multi_axes]

    n_single  = len(single_axes)
    n_models  = len(all_results)
    n_style   = len(multi_axes[0]["cac_gia_tri"]) if multi_axes else 0

    fig_w = max(5 * n_single, 2.2 * n_style + n_models)
    fig   = plt.figure(figsize=(fig_w, 22))

    outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.2, 0.85, 1.1], hspace=0.7)

    # ── Section 1: single-label bar charts ───────────────────────────────────
    gs_top = outer[0].subgridspec(1, n_single, wspace=0.35)
    for col, (adef, aidx) in enumerate(zip(single_axes, single_idxs)):
        ax = fig.add_subplot(gs_top[0, col])
        _bar_chart_single(ax, adef, all_results, sample_id, aidx)

    # ── Section 2: Style grouped bar chart ───────────────────────────────────
    if multi_axes:
        ax_mid = fig.add_subplot(outer[1])
        _grouped_bar_multi(ax_mid, multi_axes[0], all_results, sample_id, multi_idxs[0])

    # ── Section 3: top-3 Style labels per model ──────────────────────────────
    if multi_axes:
        gs_bot = outer[2].subgridspec(1, n_models, wspace=0.45)
        axes_bot = [fig.add_subplot(gs_bot[0, j]) for j in range(n_models)]
        for ax in axes_bot[1:]:
            ax.sharey(axes_bot[0])
            ax.tick_params(labelleft=False)
        axes_bot[0].set_ylabel("Confidence", fontsize=8)
        _top3_per_model_row(axes_bot, multi_axes[0], all_results, sample_id, multi_idxs[0])
        fig.text(
            0.5, outer[2].get_position(fig).y1 + 0.004,
            f"Top-3 {multi_axes[0]['metadata_vi']} labels per model  [{sample_id}]",
            ha="center", fontsize=9, fontweight="bold", color="#333333",
        )

    # ── Section 4: metrics heatmap (all 20 stories) ──────────────────────────
    ax_metrics = fig.add_subplot(outer[3])
    _metrics_heatmap(ax_metrics, metrics_all_models, axes)

    model_list = " | ".join(f"{cfg['short_name']} ({cfg['note']})" for cfg in MODELS)
    fig.suptitle(
        f"Model Comparison — Vietnamese TTS Annotation\n"
        f"Sections 1–3: {source_label}  |  Section 4: 20 stories",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.text(0.5, -0.005, model_list, ha="center", fontsize=6.5, color="gray")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved → {output_path}", file=sys.stderr)
    plt.close()


# ── ML comparison ─────────────────────────────────────────────────────────────

_ML_NAMES = ["svm", "logistic_regression", "naive_bayes"]


def _plot_ml_heatmap(
    metrics_all: Dict[str, Dict[str, Dict]],
    axes: List[dict],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.9 * len(metrics_all) + 1.5)))
    _metrics_heatmap(ax, metrics_all, axes)
    ax.set_title("ML Models — Macro-F1 per axis  (test set)", fontsize=10, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] ML chart saved → {output_path}", file=sys.stderr)
    plt.close()


def run_ml_comparison(axes: List[dict], test_path: Path, out_dir: Path) -> None:
    from src.model_ml.trainer import load_classifier, evaluate_model

    metrics_all: Dict[str, Dict] = {}
    for name in _ML_NAMES:
        try:
            clf = load_classifier(name)
            metrics_all[name] = evaluate_model(clf, test_path, axes)
            print(f"[INFO] Evaluated ML model: {name}", file=sys.stderr)
        except SystemExit:
            print(f"[WARN] {name} not trained yet — skipping. "
                  f"Run: python main.py --mode ml --action train --ml-model {name}", file=sys.stderr)

    if not metrics_all:
        print("[ERROR] No trained ML models found.", file=sys.stderr)
        return

    # Print summary
    print("\n=== ML Metrics Summary (F1) ===")
    axis_labels = [a["metadata_en"] for a in axes] + ["Overall"]
    header = f"{'Model':<22}" + "".join(f"{a:<14}" for a in axis_labels)
    print(header)
    print("-" * len(header))
    for model, m in metrics_all.items():
        row = f"{model:<22}"
        for aen in axis_labels:
            row += f"{m.get(aen, {}).get('f1', 0.0):<14.3f}"
        print(row)

    _plot_ml_heatmap(metrics_all, axes, out_dir / "compare_input_ml.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare LLM and/or ML models on test set")
    parser.add_argument("--mode", choices=["llm", "ml", "all"], default="all",
                        help="Which comparison to run (default: all)")
    parser.add_argument("--sample", default=None,
                        help="Story ID to display in LLM chart sections 1-3 (default: first)")
    parser.add_argument("--test", default=str(TEST_JSONL),
                        help=f"Path to labeled test JSONL (default: {TEST_JSONL})")
    args = parser.parse_args()

    axes     = load_axes(METADATA_PATH)
    out_dir  = Path("experiment")
    out_dir.mkdir(exist_ok=True)

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"[ERROR] Test set not found: {test_path}. Run: python -m src.dataset.generator",
              file=sys.stderr)
        sys.exit(1)

    if args.mode in ("ml", "all"):
        run_ml_comparison(axes, test_path, out_dir)

    if args.mode in ("llm", "all"):
        stories, gt = load_test_set(test_path)
        print(f"[INFO] Loaded {len(stories)} test stories.", file=sys.stderr)

        sample_id = args.sample or stories[0]["id"]
        if sample_id not in {s["id"] for s in stories}:
            print(f"[WARNING] --sample '{sample_id}' not found; using first story.", file=sys.stderr)
            sample_id = stories[0]["id"]

        all_results = run_all_models(stories, axes)

        metrics_all_models: Dict[str, Dict] = {}
        for model_short, story_results in all_results.items():
            model_preds = {
                sid: _results_to_pred_dict(results)
                for sid, results in story_results.items()
            }
            metrics_all_models[model_short] = compute_metrics(model_preds, gt, axes)

        print("\n=== LLM Metrics Summary (F1) ===")
        axis_labels = [a["metadata_en"] for a in axes] + ["Overall"]
        header = f"{'Model':<15}" + "".join(f"{a:<14}" for a in axis_labels)
        print(header)
        print("-" * len(header))
        for model, m in metrics_all_models.items():
            row = f"{model:<15}"
            for aen in axis_labels:
                row += f"{m.get(aen, {}).get('f1', 0.0):<14.3f}"
            print(row)

        plot_comparison(all_results, metrics_all_models, axes, sample_id, sample_id,
                        out_dir / "compare_input_llm.png")


if __name__ == "__main__":
    main()
