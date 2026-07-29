"""Classification metrics and evaluation figures for a rare-event target.

Accuracy is computed and reported for completeness but is never used to rank or
select a model: with a ~2% positive rate, predicting "no failure" everywhere
already scores 98%.  The metrics that drive decisions here are PR-AUC, recall,
F2 (which weights recall four times as heavily as precision) and the
false-negative rate.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.utils.paths import ensure_parent

logger = get_logger(__name__)

#: Metric names where a larger value is better. Used by the comparison table.
HIGHER_IS_BETTER = frozenset(
    {"precision", "recall", "f1", "f2", "pr_auc", "roc_auc", "accuracy", "specificity"}
)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute the full metric set at a given decision threshold.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities of the positive class.
        threshold: Decision threshold applied to ``y_prob``.

    Returns:
        A dictionary of scalar metrics, including the raw confusion-matrix
        counts so downstream code never has to recompute them.

    Raises:
        ValueError: If the inputs have mismatched lengths.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_positive = tp + fn
    n_negative = tn + fp
    has_both_classes = n_positive > 0 and n_negative > 0

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if has_both_classes
        else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "false_negative_rate": float(fn / n_positive) if n_positive else 0.0,
        "false_positive_rate": float(fp / n_negative) if n_negative else 0.0,
        "specificity": float(tn / n_negative) if n_negative else 0.0,
        "positive_prediction_rate": float(y_pred.mean()),
        "base_rate": float(y_true.mean()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "n_samples": int(y_true.size),
    }


def measure_inference_latency(model: Any, x: pd.DataFrame, n_repeats: int = 3) -> dict[str, float]:
    """Measure single-row and batch scoring latency.

    Args:
        model: Fitted estimator exposing ``predict_proba``.
        x: Feature frame to score.
        n_repeats: Number of timing repetitions; the minimum is reported.

    Returns:
        ``single_row_ms`` and ``per_row_batch_ms`` timings.
    """
    single = x.iloc[[0]]
    single_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict_proba(single)
        single_times.append((time.perf_counter() - start) * 1000.0)

    batch = x.iloc[: min(len(x), 5000)]
    batch_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict_proba(batch)
        batch_times.append((time.perf_counter() - start) * 1000.0 / len(batch))

    return {
        "single_row_ms": round(float(min(single_times)), 4),
        "per_row_batch_ms": round(float(min(batch_times)), 6),
    }


def build_comparison_table(results: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Flatten per-model, per-split metrics into a comparison table.

    Args:
        results: ``{model_name: {split_name: metrics}}``.

    Returns:
        A tidy dataframe with one row per (model, split), sorted by validation
        PR-AUC descending.
    """
    rows = [
        {"model": model_name, "split": split_name, **metrics}
        for model_name, splits in results.items()
        for split_name, metrics in splits.items()
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    ranking = (
        table.loc[table["split"] == "valid", ["model", "pr_auc"]]
        .set_index("model")["pr_auc"]
        .to_dict()
    )
    table["_rank"] = table["model"].map(ranking).fillna(-1.0)
    return (
        table.sort_values(["_rank", "model", "split"], ascending=[False, True, True])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _new_axes(figsize: tuple[float, float] = (6.4, 4.8)):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt, *plt.subplots(figsize=figsize)


def plot_confusion_matrix(metrics: dict[str, float], path: str | Path, title: str) -> Path:
    """Render a 2x2 confusion matrix.

    Args:
        metrics: Output of :func:`classification_metrics`.
        path: Destination image path.
        title: Figure title.

    Returns:
        The written path.
    """
    plt, fig, ax = _new_axes((5.2, 4.4))
    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )
    ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(
            column,
            row,
            f"{int(value):,}",
            ha="center",
            va="center",
            color="white" if value > matrix.max() / 2 else "black",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(f"{title}\n(threshold = {metrics['threshold']:.3f})")
    fig.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: str | Path,
    title: str,
    threshold: float | None = None,
) -> Path:
    """Render a precision-recall curve with the baseline and operating point.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        path: Destination image path.
        title: Figure title.
        threshold: Operating threshold to mark, if any.

    Returns:
        The written path.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    plt, fig, ax = _new_axes()
    ax.plot(
        recall, precision, lw=2, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.3f}"
    )
    base_rate = float(np.mean(y_true))
    ax.axhline(base_rate, ls="--", c="grey", lw=1, label=f"Base rate = {base_rate:.3f}")
    if threshold is not None and thresholds.size:
        index = int(np.searchsorted(thresholds, threshold))
        index = min(index, precision.size - 1)
        ax.plot(
            recall[index], precision[index], "ro", ms=8, label=f"Operating point @ {threshold:.3f}"
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path, title: str) -> Path:
    """Render a ROC curve.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        path: Destination image path.
        title: Figure title.

    Returns:
        The written path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt, fig, ax = _new_axes()
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def plot_model_comparison(table: pd.DataFrame, path: str | Path, split: str = "valid") -> Path:
    """Render a grouped bar chart comparing candidates on one split.

    Args:
        table: Output of :func:`build_comparison_table`.
        path: Destination image path.
        split: Which split to chart.

    Returns:
        The written path.
    """
    subset = table.loc[table["split"] == split].set_index("model")
    metrics = [m for m in ("pr_auc", "recall", "precision", "f2", "roc_auc") if m in subset.columns]
    plt, fig, ax = _new_axes((8.0, 4.6))
    x = np.arange(len(subset))
    width = 0.8 / max(len(metrics), 1)
    for offset, metric in enumerate(metrics):
        ax.bar(x + offset * width, subset[metric].to_numpy(), width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, subset.index, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison ({split} split)")
    ax.legend(fontsize=8, ncol=len(metrics))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target
