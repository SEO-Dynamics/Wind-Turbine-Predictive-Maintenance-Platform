"""Health-score target, candidate training, selection and metrics.

The health score is a **regression** onto the machine's condition, not a
classification of an event.  ``health_score_target = 100 * (1 - degradation)``,
so 100 is as-new and 0 is the worst state the record contains.

Where the ground truth comes from
---------------------------------
In this repository the degradation state comes from the synthetic generator's
``degradation_level`` column.  That is a legitimate label and an illegitimate
feature, which is why it is listed in ``health.features.exclude_columns`` and
asserted absent from the feature matrix by the test suite.

On a real fleet the label must come from a source **independent of the SCADA
channels the features are built from** - inspection reports, oil analysis,
borescope findings, or a certified condition-monitoring index.  Deriving the
label from the same signals that produce the features would make the model
learn its own input and report a flattering error that means nothing.
:func:`build_health_target` raises with that instruction when the column is
missing, rather than silently inventing a target.

Information flow is the same one-directional discipline the failure module
uses: train fits, valid ranks and selects, test is scored once and never fed
back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import OPERATIONAL_STATUS, TURBINE_ID
from wind_turbine_pm.health.health_class import ClassBands
from wind_turbine_pm.logging_config import get_logger

logger = get_logger(__name__)

#: Score below which an observation counts as "the part that has to be right".
#: Read from configuration at call time; this is only the fallback.
_DEFAULT_DEGRADED_BOUNDARY = 60.0


class HealthTargetError(ValueError):
    """Raised when a usable health ground truth cannot be constructed."""


@dataclass
class SplitData:
    """Feature matrix and target for one chronological split."""

    x: pd.DataFrame
    y: np.ndarray
    name: str

    @property
    def n_degraded(self) -> int:
        """Observations whose true score is below the Monitor boundary."""
        return int((self.y < _DEFAULT_DEGRADED_BOUNDARY).sum())


@dataclass
class HealthCandidate:
    """One trained candidate and everything measured about it."""

    name: str
    algorithm: str
    estimator: Any
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    valid_predictions: np.ndarray = field(default_factory=lambda: np.empty(0))
    train_seconds: float = 0.0
    rejected_reason: str = ""

    @property
    def accepted(self) -> bool:
        """Whether the candidate passed the guard rails."""
        return not self.rejected_reason


def build_health_target(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach the ``health_score_target`` column.

    Args:
        frame: Preprocessed frame carrying the ground-truth degradation column.
        cfg: Merged configuration carrying the ``health`` namespace.

    Returns:
        A copy of ``frame`` with ``health_score_target`` (0-100, higher is
        healthier) and ``health_label_eligible`` added.  Rows recorded in an
        excluded operating state are marked ineligible rather than dropped, so
        the caller keeps the full history for feature building.

    Raises:
        HealthTargetError: If the configured source column is absent or is not
            in the documented ``[0, 1]`` range.
    """
    source = str(cfg.require("health.target.source_column"))
    name = str(cfg.require("health.target.name"))
    scale = float(cfg.get("health.target.scale", 100.0))

    if source not in frame.columns:
        raise HealthTargetError(
            f"Health ground truth column {source!r} is absent. On synthetic data it is produced "
            f"by scripts/generate_synthetic_data.py. On a real fleet it must be supplied from a "
            f"source independent of the SCADA feature channels - inspection reports, oil "
            f"analysis, or a certified condition-monitoring index - and joined onto the frame "
            f"as {source!r} in [0, 1], where 0 is as-new."
        )

    degradation = pd.to_numeric(frame[source], errors="coerce")
    observed_max = float(degradation.max(skipna=True)) if degradation.notna().any() else 0.0
    if observed_max > 1.0 + 1e-9 or float(degradation.min(skipna=True) or 0.0) < -1e-9:
        raise HealthTargetError(
            f"Health ground truth {source!r} must lie in [0, 1]; observed range "
            f"[{degradation.min():.3f}, {degradation.max():.3f}]. Normalise it before use."
        )

    out = frame.copy()
    out[name] = (scale * (1.0 - degradation.fillna(0.0))).clip(0.0, scale).astype(float)

    excluded = {str(state) for state in cfg.get("health.target.exclude_states", [])}
    if excluded and OPERATIONAL_STATUS in out.columns:
        eligible = ~out[OPERATIONAL_STATUS].astype(str).isin(excluded)
    else:
        eligible = pd.Series(True, index=out.index)
    out["health_label_eligible"] = eligible & degradation.notna()

    logger.info(
        "Created health target",
        extra={
            "target": name,
            "source": source,
            "eligible_rows": int(out["health_label_eligible"].sum()),
            "mean_score": round(float(out.loc[out["health_label_eligible"], name].mean()), 3),
            "degraded_rows": int(
                (out.loc[out["health_label_eligible"], name] < _DEFAULT_DEGRADED_BOUNDARY).sum()
            ),
        },
    )
    return out


def apply_label_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a usable health label.

    Args:
        frame: Frame produced by :func:`build_health_target`.

    Returns:
        The filtered frame with the helper flag removed.

    Raises:
        KeyError: If :func:`build_health_target` was not run first.
    """
    if "health_label_eligible" not in frame.columns:
        raise KeyError("apply_label_filter requires build_health_target to have been run")
    return frame.loc[frame["health_label_eligible"]].drop(columns=["health_label_eligible"]).copy()


def to_split_data(
    frame: pd.DataFrame, features: pd.DataFrame, target_column: str, name: str
) -> SplitData:
    """Slice aligned features and target for one split.

    Args:
        frame: Metadata frame carrying the target column.
        features: Feature matrix sharing ``frame``'s index.
        target_column: Name of the regression target.
        name: Split name, used in log lines and metric keys.

    Returns:
        The :class:`SplitData`.

    Raises:
        ValueError: If the target column is missing or the split is empty.
    """
    if target_column not in frame.columns:
        raise ValueError(f"Split {name!r} has no target column {target_column!r}")
    if frame.empty:
        raise ValueError(f"Split {name!r} is empty")
    return SplitData(
        x=features.loc[frame.index],
        y=pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float),
        name=name,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, degraded_boundary: float
) -> dict[str, float]:
    """Compute the health-score metric set.

    ``mae_degraded`` - error restricted to genuinely degraded observations - is
    reported alongside overall MAE because the two disagree in the way that
    matters.  Most observations are healthy, so overall MAE is dominated by the
    easy majority; a model can win it while being useless exactly where the
    score has to be trusted.

    Args:
        y_true: Ground-truth health scores.
        y_pred: Predicted health scores.
        degraded_boundary: Score below which an observation counts as degraded.

    Returns:
        A dictionary of scalar metrics.

    Raises:
        ValueError: If the inputs have mismatched lengths.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    residual = y_pred - y_true
    total_variance = float(np.var(y_true))
    degraded = y_true < degraded_boundary

    spearman = float("nan")
    if y_true.size > 2 and np.ptp(y_true) > 0 and np.ptp(y_pred) > 0:
        ranks_true = pd.Series(y_true).rank().to_numpy()
        ranks_pred = pd.Series(y_pred).rank().to_numpy()
        spearman = float(np.corrcoef(ranks_true, ranks_pred)[0, 1])

    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "r2": float(1.0 - np.mean(residual**2) / total_variance) if total_variance > 0 else 0.0,
        "spearman": spearman,
        "bias": float(np.mean(residual)),
        "mae_degraded": float(np.mean(np.abs(residual[degraded]))) if degraded.any() else 0.0,
        "n_degraded": int(degraded.sum()),
        "n_samples": int(y_true.size),
        "within_5_points": float(np.mean(np.abs(residual) <= 5.0)),
        "within_10_points": float(np.mean(np.abs(residual) <= 10.0)),
    }


def class_agreement(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> dict[str, float]:
    """Measure how often the predicted score lands in the true health class.

    A regression metric alone does not say whether the *decision* is right: an
    error of 3 points matters if it straddles a class boundary and not
    otherwise.  This is the operationally meaningful companion metric.

    Args:
        y_true: Ground-truth health scores.
        y_pred: Predicted health scores.
        cfg: Merged configuration.

    Returns:
        Exact-class agreement, the share within one class, and the share of
        observations whose true class is worse than the predicted one (the
        dangerous direction).
    """
    from wind_turbine_pm.health.health_class import classify_health_series, severity_rank

    bands = ClassBands.from_config(cfg)
    true_classes = classify_health_series(pd.Series(y_true), cfg, bands).map(severity_rank)
    predicted_classes = classify_health_series(pd.Series(y_pred), cfg, bands).map(severity_rank)
    difference = predicted_classes - true_classes
    return {
        "class_agreement": float((difference == 0).mean()),
        "class_within_one": float((difference.abs() <= 1).mean()),
        "class_optimistic_rate": float((difference < 0).mean()),
    }


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def _build_estimator(name: str, params: dict[str, Any], seed: int) -> tuple[Any, str]:
    """Build one candidate estimator and report its algorithm name.

    Every candidate is wrapped in a pipeline that imputes missing values, so a
    feature whose trailing window is not yet defined never crashes scoring.
    Tree ensembles keep the imputer for the same reason; ``Ridge`` also gets a
    scaler because a penalised linear model is not scale-invariant.

    Args:
        name: Candidate key from configuration.
        params: Estimator hyper-parameters.
        seed: Global random seed.

    Returns:
        ``(estimator, algorithm_name)``.

    Raises:
        ValueError: If the candidate key is unknown.
    """
    imputer = SimpleImputer(strategy="median")
    if name == "mean_baseline":
        model: Any = DummyRegressor(**params)
    elif name == "ridge":
        model = Ridge(random_state=None, **params)
        return (
            Pipeline([("impute", imputer), ("scale", StandardScaler()), ("model", model)]),
            "Ridge",
        )
    elif name == "random_forest":
        model = RandomForestRegressor(random_state=seed, **params)
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(random_state=seed, **params)
    else:
        raise ValueError(f"Unknown health candidate {name!r}")
    return Pipeline([("impute", imputer), ("model", model)]), type(model).__name__


def train_health_candidates(
    train: SplitData, valid: SplitData, cfg: Config
) -> list[HealthCandidate]:
    """Train every enabled candidate and measure it on train and validation.

    Args:
        train: Training split.
        valid: Validation split.
        cfg: Merged configuration.

    Returns:
        One :class:`HealthCandidate` per enabled candidate, in configuration
        order.

    Raises:
        ValueError: If no candidate is enabled.
    """
    configured = dict(cfg.require("health.training.candidates"))
    seed = int(cfg.get("random_seed", 42))
    boundary = float(cfg.get("health.classes.monitor_min", _DEFAULT_DEGRADED_BOUNDARY))

    candidates: list[HealthCandidate] = []
    for name, body in configured.items():
        settings = dict(body)
        if not bool(settings.get("enabled", True)):
            logger.info("Skipping disabled health candidate", extra={"candidate": name})
            continue

        estimator, algorithm = _build_estimator(name, dict(settings.get("params", {})), seed)
        started = time.perf_counter()
        estimator.fit(train.x, train.y)
        elapsed = time.perf_counter() - started

        train_predictions = np.clip(estimator.predict(train.x), 0.0, 100.0)
        valid_predictions = np.clip(estimator.predict(valid.x), 0.0, 100.0)
        metrics = {
            "train": regression_metrics(train.y, train_predictions, boundary),
            "valid": {
                **regression_metrics(valid.y, valid_predictions, boundary),
                **class_agreement(valid.y, valid_predictions, cfg),
            },
        }
        candidates.append(
            HealthCandidate(
                name=name,
                algorithm=algorithm,
                estimator=estimator,
                metrics=metrics,
                valid_predictions=valid_predictions,
                train_seconds=elapsed,
            )
        )
        logger.info(
            "Trained health candidate",
            extra={
                "candidate": name,
                "seconds": round(elapsed, 2),
                "valid_mae": round(metrics["valid"]["mae"], 3),
                "valid_mae_degraded": round(metrics["valid"]["mae_degraded"], 3),
                "valid_spearman": round(metrics["valid"]["spearman"], 3),
            },
        )

    if not candidates:
        raise ValueError("No health candidates are enabled in health.training.candidates")
    return candidates


def select_best_health_candidate(
    candidates: list[HealthCandidate], cfg: Config
) -> tuple[HealthCandidate, str]:
    """Rank candidates on the validation split and pick a winner.

    Guard rails, applied before ranking:

    * a candidate whose rank correlation with the true state is below
      ``min_spearman`` is rejected - a score that does not order turbines
      correctly cannot prioritise maintenance, whatever its MAE;
    * a candidate no better than ``max_mae_ratio_vs_baseline`` times the
      mean-prediction baseline is rejected as not worth deploying.

    Survivors are ranked on ``mae_degraded`` with a tolerance band, so a
    sub-percent difference - noise at this dataset size - does not decide the
    winner; ties are broken by the configured tie-breakers.

    Args:
        candidates: Trained candidates.
        cfg: Merged configuration.

    Returns:
        ``(winner, rationale)``.

    Raises:
        ValueError: If every candidate was rejected.
    """
    primary = str(cfg.get("health.training.selection.primary_metric", "mae_degraded"))
    tie_breakers = list(cfg.get("health.training.selection.tie_breakers", ["mae", "rmse"]))
    tolerance = float(cfg.get("health.training.selection.primary_tolerance", 0.05))
    min_spearman = float(cfg.get("health.training.selection.min_spearman", 0.5))
    max_ratio = float(cfg.get("health.training.selection.max_mae_ratio_vs_baseline", 0.85))

    baseline = next((c for c in candidates if c.name == "mean_baseline"), None)
    baseline_mae = float(baseline.metrics["valid"]["mae"]) if baseline else float("inf")

    eligible: list[HealthCandidate] = []
    for candidate in candidates:
        if candidate.name == "mean_baseline":
            candidate.rejected_reason = "reference baseline, never published"
            continue
        valid = candidate.metrics["valid"]
        spearman = float(valid.get("spearman", float("nan")))
        if not np.isfinite(spearman) or spearman < min_spearman:
            candidate.rejected_reason = (
                f"validation Spearman {spearman:.3f} is below the {min_spearman:.2f} floor"
            )
            continue
        if np.isfinite(baseline_mae) and valid["mae"] > max_ratio * baseline_mae:
            candidate.rejected_reason = (
                f"validation MAE {valid['mae']:.3f} is not better than {max_ratio:.0%} of the "
                f"mean-prediction baseline ({baseline_mae:.3f})"
            )
            continue
        eligible.append(candidate)

    if not eligible:
        reasons = "; ".join(f"{c.name}: {c.rejected_reason}" for c in candidates)
        raise ValueError(f"Every health candidate was rejected. Reasons - {reasons}")

    def primary_value(candidate: HealthCandidate) -> float:
        return float(candidate.metrics["valid"][primary])

    best_value = min(primary_value(candidate) for candidate in eligible)
    band = best_value * (1.0 + tolerance) if best_value > 0 else tolerance
    tied = [candidate for candidate in eligible if primary_value(candidate) <= band]

    def sort_key(candidate: HealthCandidate) -> tuple[float, ...]:
        valid = candidate.metrics["valid"]
        keys: list[float] = []
        for metric in tie_breakers:
            value = float(valid.get(metric, float("nan")))
            # Higher is better for correlation-style metrics, lower for errors.
            keys.append(-value if metric in {"spearman", "r2", "class_agreement"} else value)
        return tuple(keys)

    winner = sorted(tied, key=sort_key)[0]
    rationale = (
        f"Selected {winner.name} ({winner.algorithm}) on the validation split: {primary}="
        f"{primary_value(winner):.3f}, MAE={winner.metrics['valid']['mae']:.3f}, "
        f"Spearman={winner.metrics['valid']['spearman']:.3f}, class agreement="
        f"{winner.metrics['valid'].get('class_agreement', float('nan')):.3f}. "
        f"{len(tied)} candidate(s) fell inside the {tolerance:.0%} tolerance band on {primary} "
        f"and were separated by {tie_breakers}. Rejected: "
        + (
            "; ".join(f"{c.name} ({c.rejected_reason})" for c in candidates if c.rejected_reason)
            or "none"
        )
    )
    logger.info("Health model selection complete", extra={"winner": winner.name})
    return winner, rationale


def evaluate_on_test(candidate: HealthCandidate, test: SplitData, cfg: Config) -> dict[str, float]:
    """Score the winner on the held-out test split, once.

    Args:
        candidate: The selected candidate.
        test: The test split.
        cfg: Merged configuration.

    Returns:
        Test metrics including class agreement.
    """
    boundary = float(cfg.get("health.classes.monitor_min", _DEFAULT_DEGRADED_BOUNDARY))
    predictions = np.clip(candidate.estimator.predict(test.x), 0.0, 100.0)
    return {
        **regression_metrics(test.y, predictions, boundary),
        **class_agreement(test.y, predictions, cfg),
    }


def comparison_table(candidates: list[HealthCandidate]) -> pd.DataFrame:
    """Flatten candidate metrics into a comparison table.

    Args:
        candidates: Trained candidates.

    Returns:
        A tidy frame with one row per (model, split), best validation
        ``mae_degraded`` first.
    """
    rows = [
        {"model": candidate.name, "algorithm": candidate.algorithm, "split": split, **metrics}
        for candidate in candidates
        for split, metrics in candidate.metrics.items()
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    ranking = (
        table.loc[table["split"] == "valid", ["model", "mae_degraded"]]
        .set_index("model")["mae_degraded"]
        .to_dict()
    )
    table["_rank"] = table["model"].map(ranking).fillna(float("inf"))
    return (
        table.sort_values(["_rank", "model", "split"])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


def predict_health_scores(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """Score a feature matrix and clamp the result to the published range.

    Args:
        estimator: The fitted health model.
        features: Feature matrix already aligned to the model's feature order.

    Returns:
        Health scores in ``[0, 100]``.
    """
    return np.clip(np.asarray(estimator.predict(features), dtype=float), 0.0, 100.0)


def training_timestamp() -> datetime:
    """Return the current UTC time, for the metadata document."""
    return datetime.now(UTC)


def library_versions() -> dict[str, str]:
    """Record the versions of the libraries the artifact depends on.

    Returns:
        A mapping of package name to version, or ``"not installed"``.
    """
    import importlib.metadata as importlib_metadata

    versions: dict[str, str] = {}
    for package in ("scikit-learn", "numpy", "pandas", "scipy", "joblib"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - broken env
            versions[package] = "not installed"
    return versions


def dataset_shape(frame: pd.DataFrame) -> dict[str, int]:
    """Summarise a prepared health frame for logging.

    Args:
        frame: The prepared frame.

    Returns:
        Row and turbine counts.
    """
    return {
        "rows": int(len(frame)),
        "turbines": int(frame[TURBINE_ID].nunique()) if TURBINE_ID in frame.columns else 0,
    }
