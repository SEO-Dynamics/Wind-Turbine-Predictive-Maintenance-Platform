"""Evaluation figures for anomaly detection and maintenance decision support.

The Stage 3 figures keep two different kinds of evidence separate:

``plot_anomaly_model_comparison``
    Validation performance for every novelty-model candidate. Test metrics are
    deliberately excluded because they did not participate in selection.
``plot_anomaly_calibration``
    The empirical healthy-validation score mapping and its measured warning /
    alarm rates.
``plot_maintenance_policy``
    The configured component weights and deterministic action windows. This is
    a policy visual, not a claim of measured operational effectiveness.

Matplotlib is imported lazily behind the non-interactive ``Agg`` backend so API
imports never initialise a GUI toolkit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.contracts.anomaly import AnomalyCalibration
from wind_turbine_pm.utils.paths import ensure_parent

INK = "#1f2937"
MUTED = "#64748b"
GRID = "#dbe3ec"
BLUE = "#2563eb"
GOLD = "#d97706"
ORANGE = "#ea580c"
OLIVE = "#6b7f2a"
LIGHT_BLUE = "#dbeafe"


def _new_axes(figsize: tuple[float, float] = (8.6, 5.0)):
    """Create a figure/axes pair on a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt, *plt.subplots(figsize=figsize)


def _save(plt, fig, path: str | Path) -> Path:
    """Write a figure with consistent report-friendly styling."""
    target = ensure_parent(path)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def _humanize(value: str) -> str:
    return value.replace("_", " ").title()


def plot_anomaly_model_comparison(comparison: pd.DataFrame, path: str | Path) -> Path:
    """Chart candidate validation metrics used in model review.

    Args:
        comparison: Candidate table written by the anomaly training pipeline.
        path: Destination image path.

    Returns:
        The written path.
    """
    required = {"model", "selected", "pr_auc", "recall", "precision", "f2"}
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError(f"Anomaly comparison is missing columns: {missing}")
    if comparison.empty:
        raise ValueError("Anomaly comparison must contain at least one candidate")

    metrics = ("pr_auc", "recall", "precision", "f2")
    colours = (BLUE, GOLD, ORANGE, OLIVE)
    labels = [
        f"{_humanize(str(row.model))}\n(selected)"
        if bool(row.selected)
        else _humanize(str(row.model))
        for row in comparison.itertuples()
    ]

    plt, fig, ax = _new_axes((9.4, 5.4))
    x = np.arange(len(comparison), dtype=float)
    width = 0.18
    for index, (metric, colour) in enumerate(zip(metrics, colours, strict=True)):
        positions = x + (index - 1.5) * width
        values = comparison[metric].to_numpy(dtype=float)
        bars = ax.bar(
            positions,
            values,
            width,
            label=metric.replace("_", "-").upper(),
            color=colour,
            edgecolor=INK,
            linewidth=0.35,
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7)

    ax.set_xticks(x, labels)
    for tick, selected in zip(ax.get_xticklabels(), comparison["selected"], strict=True):
        tick.set_fontweight("bold" if bool(selected) else "normal")
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Score (0–1)")
    ax.set_title(
        "Anomaly candidate performance",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
        pad=24,
    )
    ax.text(
        0.0,
        1.02,
        "Validation split · warning threshold calibrated on healthy observations · higher is better",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
        va="bottom",
    )
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    return _save(plt, fig, path)


def plot_anomaly_calibration(calibration: AnomalyCalibration, path: str | Path) -> Path:
    """Visualise the empirical healthy-reference calibration contract.

    Args:
        calibration: Selected model's healthy-validation calibration.
        path: Destination image path.

    Returns:
        The written path.
    """
    scores = np.asarray(calibration.reference_scores, dtype=float)
    scores = np.sort(scores[np.isfinite(scores)])
    if scores.size < 2:
        raise ValueError("Calibration figure requires at least two finite reference scores")

    plt, fig, _ = _new_axes((10.2, 4.8))
    fig.clear()
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1.0]})

    percentiles = np.arange(1, scores.size + 1, dtype=float) / scores.size
    display_max = float(np.quantile(scores, 0.995))
    left.plot(scores, percentiles, color=BLUE, linewidth=2.0, label="Healthy empirical CDF")
    left.axvline(
        calibration.warning_raw,
        color=GOLD,
        linestyle="--",
        linewidth=1.6,
        label="Warning threshold (95th percentile)",
    )
    left.axvline(
        calibration.alarm_raw,
        color=ORANGE,
        linestyle=":",
        linewidth=2.0,
        label="Alarm threshold (99th percentile)",
    )
    left.set_xlim(float(scores.min()), display_max)
    left.set_ylim(0.0, 1.01)
    left.set_xlabel("Raw novelty score (higher = more unusual)")
    left.set_ylabel("Healthy-reference percentile")
    left.grid(color=GRID, linewidth=0.8, alpha=0.8)
    left.set_axisbelow(True)
    left.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    left.text(
        0.99,
        0.03,
        "x-axis capped at healthy 99.5th percentile",
        transform=left.transAxes,
        ha="right",
        color=MUTED,
        fontsize=7,
    )

    target = np.array([1.0 - calibration.warning_percentile, 1.0 - calibration.alarm_percentile])
    measured = np.array([calibration.achieved_warning_rate, calibration.achieved_alarm_rate])
    positions = np.arange(2, dtype=float)
    width = 0.34
    target_bars = right.bar(
        positions - width / 2,
        target,
        width,
        label="Target",
        color="white",
        edgecolor=INK,
        linewidth=1.2,
    )
    measured_bars = right.bar(
        positions + width / 2,
        measured,
        width,
        label="Measured",
        color=LIGHT_BLUE,
        edgecolor=BLUE,
        linewidth=1.0,
    )
    right.bar_label(target_bars, labels=[f"{value:.1%}" for value in target], padding=3, fontsize=8)
    right.bar_label(
        measured_bars, labels=[f"{value:.2%}" for value in measured], padding=3, fontsize=8
    )
    right.set_xticks(positions, ["Warning", "Alarm"])
    right.set_ylim(0.0, max(0.065, float(max(target.max(), measured.max()) * 1.3)))
    right.set_ylabel("Healthy observations alerted")
    right.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    right.set_axisbelow(True)
    right.spines[["top", "right"]].set_visible(False)
    right.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "Healthy-reference anomaly calibration",
        x=0.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.92,
        f"Validation reference · n={scores.size:,} healthy observations · synthetic data",
        color=MUTED,
        fontsize=9,
    )
    return _save(plt, fig, path)


def plot_maintenance_policy(cfg: Config, path: str | Path) -> Path:
    """Chart configured risk weights and deterministic review windows.

    Args:
        cfg: Stage 3 configuration containing the ``maintenance`` namespace.
        path: Destination image path.

    Returns:
        The written path.
    """
    weights = cfg.require("maintenance.weights").to_dict()
    actions = cfg.require("maintenance.actions")
    action_hours = {
        "Same-shift review": float(actions.require("immediate_review_hours")),
        "Urgent review": float(actions.require("urgent_review_hours")),
        "Planned inspection": float(actions.require("planned_inspection_hours")),
    }

    plt, fig, _ = _new_axes((10.2, 4.8))
    fig.clear()
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.35]})

    weight_labels = [_humanize(name) for name in weights]
    weight_values = np.array([float(value) for value in weights.values()])
    weight_positions = np.arange(len(weight_labels))
    bars = left.barh(
        weight_positions,
        weight_values,
        color=[BLUE, GOLD, OLIVE],
        edgecolor=INK,
        linewidth=0.4,
    )
    left.bar_label(bars, labels=[f"{value:.0%}" for value in weight_values], padding=4, fontsize=9)
    left.set_yticks(weight_positions, weight_labels)
    left.invert_yaxis()
    left.set_xlim(0.0, max(0.6, float(weight_values.max() * 1.2)))
    left.set_xlabel("Share of unified risk")
    left.set_title("Evidence weights", loc="left", fontsize=11, fontweight="bold", color=INK)
    left.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    left.set_axisbelow(True)
    left.spines[["top", "right"]].set_visible(False)

    action_labels = list(action_hours)
    hours = np.array(list(action_hours.values()))
    action_positions = np.arange(len(action_labels))
    action_bars = right.barh(
        action_positions,
        hours,
        color=[ORANGE, GOLD, LIGHT_BLUE],
        edgecolor=[ORANGE, GOLD, BLUE],
        linewidth=1.0,
    )
    right.bar_label(
        action_bars,
        labels=[f"{value:.0f} h" for value in hours],
        padding=4,
        fontsize=9,
    )
    right.set_yticks(action_positions, action_labels)
    right.invert_yaxis()
    right.set_xlim(0.0, float(hours.max() * 1.18))
    right.set_xlabel("Recommended review window (hours)")
    right.set_title(
        "Deterministic action windows", loc="left", fontsize=11, fontweight="bold", color=INK
    )
    right.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    right.set_axisbelow(True)
    right.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Maintenance decision policy",
        x=0.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.92,
        "Configured weights and action windows · guardrails can raise severity · advisory only",
        color=MUTED,
        fontsize=9,
    )
    return _save(plt, fig, path)
