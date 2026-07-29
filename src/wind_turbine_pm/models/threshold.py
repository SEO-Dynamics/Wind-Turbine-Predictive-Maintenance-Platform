"""Decision-threshold optimisation.

A threshold of 0.50 is meaningless for a rare-event classifier trained with
class weighting: it is an artefact of the loss function, not an operating
decision.  This module searches a grid on the **validation split only** and
selects a threshold under an explicit objective.

Two methods are implemented:

``f2``
    Maximise the F-beta score with ``beta = 2``, which weights recall four times
    as heavily as precision.  Appropriate when the cost of a missed failure
    dominates but has not been quantified.

``cost``
    Minimise an expected cost built from an explicit, **hypothetical** cost
    matrix (see ``configs/failure_model.yaml``).  The default figures - missed
    failure 10, false alarm 2, acted-on true positive 1 - are illustrative units
    chosen to express "a missed failure is roughly five times worse than an
    unnecessary inspection".  They are not calibrated against any real
    operator's economics and must be re-estimated before operational use.

The test split is never passed to any function in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.models.evaluation import classification_metrics
from wind_turbine_pm.utils.paths import ensure_parent

logger = get_logger(__name__)


@dataclass(frozen=True)
class CostMatrix:
    """Illustrative cost of each confusion-matrix outcome, in arbitrary units."""

    false_negative: float = 10.0
    false_positive: float = 2.0
    true_positive: float = 1.0
    true_negative: float = 0.0

    @classmethod
    def from_config(cls, cfg: Config) -> CostMatrix:
        """Build a cost matrix from configuration.

        Args:
            cfg: Merged configuration.

        Returns:
            The configured :class:`CostMatrix`.
        """
        costs = cfg.get("threshold.costs") or {}
        return cls(
            false_negative=float(costs.get("false_negative", 10.0)),
            false_positive=float(costs.get("false_positive", 2.0)),
            true_positive=float(costs.get("true_positive", 1.0)),
            true_negative=float(costs.get("true_negative", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable representation."""
        return {
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
        }

    def total(self, tn: int, fp: int, fn: int, tp: int) -> float:
        """Compute total expected cost for a confusion matrix.

        Args:
            tn: True negatives.
            fp: False positives.
            fn: False negatives.
            tp: True positives.

        Returns:
            The total cost in illustrative units.
        """
        return (
            tn * self.true_negative
            + fp * self.false_positive
            + fn * self.false_negative
            + tp * self.true_positive
        )


@dataclass
class ThresholdResult:
    """Outcome of a threshold search."""

    threshold: float
    method: str
    selected_on: str
    costs: dict[str, float]
    metrics: dict[str, float]
    alternatives: dict[str, float] = field(default_factory=dict)
    curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation (excluding the curve)."""
        return {
            "threshold": float(self.threshold),
            "method": self.method,
            "selected_on": self.selected_on,
            "costs": self.costs,
            "metrics": self.metrics,
            "alternatives": self.alternatives,
        }


def threshold_grid(cfg: Config) -> np.ndarray:
    """Build the threshold search grid from configuration.

    Args:
        cfg: Merged configuration.

    Returns:
        A 1-D array of candidate thresholds.
    """
    return np.linspace(
        float(cfg.get("threshold.grid_start", 0.01)),
        float(cfg.get("threshold.grid_stop", 0.99)),
        int(cfg.get("threshold.grid_steps", 197)),
    )


def threshold_curve(
    y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray, costs: CostMatrix
) -> pd.DataFrame:
    """Evaluate every threshold on the grid.

    Args:
        y_true: Validation ground-truth labels.
        y_prob: Validation predicted probabilities.
        grid: Candidate thresholds.
        costs: Illustrative cost matrix.

    Returns:
        One row per threshold with metrics and total cost.
    """
    rows = []
    for threshold in grid:
        metrics = classification_metrics(y_true, y_prob, float(threshold))
        metrics["total_cost"] = costs.total(
            metrics["true_negatives"],
            metrics["false_positives"],
            metrics["false_negatives"],
            metrics["true_positives"],
        )
        metrics["cost_per_sample"] = metrics["total_cost"] / max(metrics["n_samples"], 1)
        rows.append(metrics)
    return pd.DataFrame(rows)


def optimise_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: Config,
    method: str | None = None,
    split_name: str = "valid",
) -> ThresholdResult:
    """Select a decision threshold on the validation split.

    Args:
        y_true: Validation ground-truth labels.
        y_prob: Validation predicted probabilities.
        cfg: Merged configuration.
        method: Override for ``threshold.method`` (``"f2"`` or ``"cost"``).
        split_name: Name recorded in the artifact; must not be ``"test"``.

    Returns:
        The :class:`ThresholdResult`, including the full curve and the
        thresholds the *other* method would have chosen.

    Raises:
        ValueError: If ``method`` is unknown or ``split_name`` is ``"test"``.
    """
    if split_name == "test":
        raise ValueError(
            "Threshold selection must never use the test split; "
            "pass validation predictions instead."
        )
    chosen_method = str(method or cfg.require("threshold.method"))
    if chosen_method not in {"f2", "cost"}:
        raise ValueError(f"Unknown threshold method {chosen_method!r}; expected 'f2' or 'cost'")

    costs = CostMatrix.from_config(cfg)
    grid = threshold_grid(cfg)
    curve = threshold_curve(np.asarray(y_true), np.asarray(y_prob), grid, costs)

    f2_threshold = float(curve.loc[curve["f2"].idxmax(), "threshold"])
    cost_threshold = float(curve.loc[curve["total_cost"].idxmin(), "threshold"])
    selected = f2_threshold if chosen_method == "f2" else cost_threshold

    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_prob), selected)
    metrics["total_cost"] = costs.total(
        metrics["true_negatives"],
        metrics["false_positives"],
        metrics["false_negatives"],
        metrics["true_positives"],
    )
    metrics["cost_per_sample"] = metrics["total_cost"] / max(metrics["n_samples"], 1)

    result = ThresholdResult(
        threshold=selected,
        method=chosen_method,
        selected_on=split_name,
        costs=costs.to_dict(),
        metrics=metrics,
        alternatives={"f2": f2_threshold, "cost": cost_threshold, "default": 0.5},
        curve=curve,
    )
    logger.info(
        "Selected decision threshold",
        extra={
            "method": chosen_method,
            "threshold": round(selected, 4),
            "recall": round(metrics["recall"], 4),
            "precision": round(metrics["precision"], 4),
            "f2": round(metrics["f2"], 4),
            "alternatives": result.alternatives,
        },
    )
    return result


def plot_threshold_curve(result: ThresholdResult, path: str | Path) -> Path:
    """Render metric and cost curves against the threshold.

    Args:
        result: A completed threshold search.
        path: Destination image path.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    curve = result.curve
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True)

    for metric in ("precision", "recall", "f1", "f2"):
        top.plot(curve["threshold"], curve[metric], lw=1.7, label=metric)
    top.axvline(
        result.threshold, c="red", ls="--", lw=1.4, label=f"selected = {result.threshold:.3f}"
    )
    top.axvline(0.5, c="grey", ls=":", lw=1.2, label="naive 0.50")
    top.set_ylabel("Score")
    top.set_title(
        f"Threshold selection on the {result.selected_on} split (method: {result.method})"
    )
    top.legend(fontsize=8, ncol=3)
    top.grid(alpha=0.3)

    bottom.plot(
        curve["threshold"],
        curve["cost_per_sample"],
        lw=1.8,
        c="darkorange",
        label="cost per sample",
    )
    bottom.axvline(result.threshold, c="red", ls="--", lw=1.4)
    bottom.axvline(0.5, c="grey", ls=":", lw=1.2)
    bottom.set_xlabel("Decision threshold")
    bottom.set_ylabel("Illustrative cost / sample")
    bottom.legend(fontsize=8)
    bottom.grid(alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "Cost units are hypothetical and not calibrated against real operator economics.",
        ha="center",
        fontsize=7.5,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    target = ensure_parent(path)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def map_risk_level(probability: float, low_max: float, medium_max: float) -> str:
    """Map a probability to its configured risk band.

    Args:
        probability: Failure probability in ``[0, 1]``.
        low_max: Upper bound (exclusive) of the low band.
        medium_max: Upper bound (exclusive) of the medium band.

    Returns:
        ``"low"``, ``"medium"`` or ``"high"``.

    Raises:
        ValueError: If the bands are not strictly increasing.
    """
    if not 0.0 < low_max < medium_max < 1.0:
        raise ValueError(
            f"Risk bands must satisfy 0 < low_max < medium_max < 1, got {low_max}, {medium_max}"
        )
    if probability < low_max:
        return "low"
    if probability < medium_max:
        return "medium"
    return "high"
