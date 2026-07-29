"""Evaluation figures for the health-score regression.

The Failure Prediction Module's figures (PR curve, confusion matrix, threshold
sweep) describe a thresholded binary classifier and say nothing useful about a
0-100 condition score, so this module provides the regression equivalents:

``plot_health_model_comparison``
    Candidate MAE / MAE-on-degraded / RMSE side by side on the validation split.
``plot_predicted_vs_actual``
    The honest scatter: where the score is right, where it is wrong, and in
    which direction.  The class bands are drawn on so the operationally
    important errors - the ones that cross a boundary - are visible rather than
    averaged away.
``plot_error_by_band``
    Absolute error grouped by true health band, which is the plot that shows
    whether good overall MAE is being carried by the healthy majority.
``plot_class_confusion``
    Predicted against true health class: the decision-level view of the score.

Matplotlib is imported lazily behind the ``Agg`` backend, exactly as the failure
module does it, so importing this module in a headless process (or in the API)
never initialises a GUI toolkit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import HEALTH_CLASS_ORDER
from wind_turbine_pm.health.health_class import ClassBands, classify_health_series, severity_rank
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.utils.paths import ensure_parent

logger = get_logger(__name__)

#: Colour per health class, used consistently across figures and the dashboard.
CLASS_COLOURS: dict[str, str] = {
    "healthy": "#2f855a",
    "monitor": "#b7791f",
    "degraded": "#c05621",
    "critical": "#c53030",
}


def _new_axes(figsize: tuple[float, float] = (6.4, 4.8)):
    """Create a figure/axes pair on the non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt, *plt.subplots(figsize=figsize)


def _save(plt, fig, path: str | Path) -> Path:
    """Write a figure and close it."""
    target = ensure_parent(path)
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def plot_health_model_comparison(
    table: pd.DataFrame, path: str | Path, split: str = "valid"
) -> Path:
    """Chart candidate error metrics on one split.

    Args:
        table: Output of
            :func:`~wind_turbine_pm.health.health_score.comparison_table`.
        path: Destination image path.
        split: Which split to chart.

    Returns:
        The written path.
    """
    subset = table.loc[table["split"] == split].set_index("model")
    metrics = [m for m in ("mae", "mae_degraded", "rmse") if m in subset.columns]
    plt, fig, ax = _new_axes((8.0, 4.6))
    x = np.arange(len(subset))
    width = 0.8 / max(len(metrics), 1)
    for offset, metric in enumerate(metrics):
        ax.bar(x + offset * width, subset[metric].to_numpy(), width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, subset.index, rotation=15, ha="right")
    ax.set_ylabel("Error (health-score points)")
    ax.set_title(f"Health candidate comparison ({split} split) - lower is better")
    ax.legend(fontsize=8, ncol=len(metrics))
    ax.grid(axis="y", alpha=0.3)
    return _save(plt, fig, path)


def plot_predicted_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, cfg: Config, path: str | Path, title: str
) -> Path:
    """Scatter predicted against true health score, with the class bands drawn.

    Args:
        y_true: Ground-truth health scores.
        y_pred: Predicted health scores.
        cfg: Merged configuration, for the class boundaries.
        path: Destination image path.
        title: Figure title.

    Returns:
        The written path.
    """
    bands = ClassBands.from_config(cfg)
    plt, fig, ax = _new_axes((6.0, 5.6))

    ax.scatter(y_true, y_pred, s=5, alpha=0.25, color="#2b6cb0", edgecolors="none")
    ax.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1, label="perfect")

    for boundary, label in (
        (bands.healthy_min, "healthy"),
        (bands.monitor_min, "monitor"),
        (bands.degraded_min, "degraded"),
    ):
        ax.axvline(boundary, color="#718096", linewidth=0.8, alpha=0.6)
        ax.axhline(boundary, color="#718096", linewidth=0.8, alpha=0.6)
        ax.text(boundary + 0.6, 2, label, fontsize=7, rotation=90, color="#4a5568")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("True health score")
    ax.set_ylabel("Predicted health score")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    return _save(plt, fig, path)


def plot_error_by_band(
    y_true: np.ndarray, y_pred: np.ndarray, cfg: Config, path: str | Path
) -> Path:
    """Chart absolute error grouped by the observation's true health band.

    Args:
        y_true: Ground-truth health scores.
        y_pred: Predicted health scores.
        cfg: Merged configuration.
        path: Destination image path.

    Returns:
        The written path.
    """
    classes = classify_health_series(pd.Series(np.asarray(y_true, dtype=float)), cfg)
    error = pd.Series(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))

    labels = [str(name) for name in HEALTH_CLASS_ORDER]
    grouped = [error[classes.to_numpy() == label].to_numpy() for label in labels]
    counts = [len(values) for values in grouped]

    plt, fig, ax = _new_axes((7.0, 4.4))
    positions = np.arange(len(labels))
    populated = [index for index, values in enumerate(grouped) if len(values)]
    if populated:
        ax.boxplot(
            [grouped[index] for index in populated],
            positions=positions[populated],
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#bee3f8", "edgecolor": "#2b6cb0"},
            medianprops={"color": "#c53030"},
        )
    ax.set_xticks(positions, [f"{label}\n(n={count})" for label, count in zip(labels, counts)])
    ax.set_ylabel("Absolute error (points)")
    ax.set_title("Health-score error by true band - the degraded bands are the ones that matter")
    ax.grid(axis="y", alpha=0.3)
    return _save(plt, fig, path)


def plot_class_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, cfg: Config, path: str | Path
) -> Path:
    """Render the predicted-versus-true health-class matrix.

    Args:
        y_true: Ground-truth health scores.
        y_pred: Predicted health scores.
        cfg: Merged configuration.
        path: Destination image path.

    Returns:
        The written path.
    """
    labels = [str(name) for name in HEALTH_CLASS_ORDER]
    true_classes = classify_health_series(pd.Series(np.asarray(y_true, dtype=float)), cfg)
    predicted_classes = classify_health_series(pd.Series(np.asarray(y_pred, dtype=float)), cfg)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(true_classes, predicted_classes):
        matrix[labels.index(actual), labels.index(predicted)] += 1

    plt, fig, ax = _new_axes((5.8, 5.0))
    ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(
            column,
            row,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if value > matrix.max() / 2 else "black",
        )
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Health class agreement")
    return _save(plt, fig, path)


def band_error_table(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Summarise error and optimism per true health band.

    ``optimistic_rate`` - the share of observations the model placed in a
    *healthier* band than the truth - is reported per band because it is the
    dangerous direction: calling a degraded machine healthy costs more than the
    reverse, and an aggregate error figure hides it entirely.

    Args:
        y_true: Ground-truth health scores.
        y_pred: Predicted health scores.
        cfg: Merged configuration.

    Returns:
        One row per health band, in canonical severity order.
    """
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    true_classes = classify_health_series(pd.Series(true_values), cfg)
    predicted_classes = classify_health_series(pd.Series(predicted_values), cfg)
    difference = predicted_classes.map(severity_rank) - true_classes.map(severity_rank)
    error = np.abs(predicted_values - true_values)

    rows: list[dict[str, float | str | int]] = []
    for name in HEALTH_CLASS_ORDER:
        mask = (true_classes == str(name)).to_numpy()
        count = int(mask.sum())
        rows.append(
            {
                "health_class": str(name),
                "n": count,
                "mae": round(float(error[mask].mean()), 4) if count else 0.0,
                "class_agreement": round(float((difference[mask] == 0).mean()), 4)
                if count
                else 0.0,
                "optimistic_rate": round(float((difference[mask] < 0).mean()), 4) if count else 0.0,
            }
        )
    return pd.DataFrame(rows)
