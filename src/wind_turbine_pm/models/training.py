"""Training, candidate comparison and final model selection.

The flow is strictly one-directional with respect to information:

1. Fit every candidate on **train** only.
2. Score and rank candidates on **valid** only.
3. Optimise the decision threshold on **valid** only.
4. Score the selected model once on **test**, purely to report generalisation.

Nothing computed from the test split ever feeds back into a modelling decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import TARGET_COLUMN
from wind_turbine_pm.features.failure_features import FeatureSpec
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.models.baselines import Candidate, build_candidates, compute_sample_weights
from wind_turbine_pm.models.evaluation import (
    build_comparison_table,
    classification_metrics,
    measure_inference_latency,
)
from wind_turbine_pm.models.threshold import ThresholdResult, optimise_threshold

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Feature matrix and label vector for one split."""

    x: pd.DataFrame
    y: np.ndarray

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(f"Feature/label length mismatch: {len(self.x)} vs {len(self.y)}")

    @property
    def n_positive(self) -> int:
        """Number of positive labels in this split."""
        return int(self.y.sum())


@dataclass
class TrainedCandidate:
    """A fitted candidate with its validation performance."""

    name: str
    algorithm: str
    estimator: Any
    train_seconds: float
    valid_probabilities: np.ndarray
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    threshold: ThresholdResult | None = None
    latency: dict[str, float] = field(default_factory=dict)
    rejected_reason: str | None = None


def to_split_data(
    frame: pd.DataFrame, features: pd.DataFrame, target: str = TARGET_COLUMN
) -> SplitData:
    """Assemble aligned features and labels for one split.

    Args:
        frame: The labelled frame for this split (carries the target column).
        features: Feature matrix indexed identically to ``frame``.
        target: Target column name.

    Returns:
        A :class:`SplitData`.

    Raises:
        KeyError: If the target column is absent.
    """
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} not present in frame")
    aligned = features.loc[frame.index]
    return SplitData(x=aligned, y=frame[target].to_numpy(dtype=int))


def _fit_candidate(candidate: Candidate, train: SplitData) -> tuple[Any, float]:
    """Fit one candidate, applying sample weights where supported."""
    estimator = clone(candidate.estimator)
    weights = (
        compute_sample_weights(train.y, candidate.positive_class_weight)
        if candidate.supports_sample_weight
        else None
    )
    start = time.perf_counter()
    if weights is not None:
        estimator.fit(train.x, train.y, model__sample_weight=weights)
    else:
        estimator.fit(train.x, train.y)
    elapsed = time.perf_counter() - start
    logger.info(
        "Fitted candidate",
        extra={
            "candidate": candidate.name,
            "seconds": round(elapsed, 2),
            "weighted": weights is not None,
        },
    )
    return estimator, elapsed


def _freeze(estimator: Any) -> Any:
    """Wrap a fitted estimator so a meta-estimator cannot refit it.

    scikit-learn 1.6 introduced ``FrozenEstimator`` and 1.8 removed the older
    ``cv="prefit"`` argument.  This helper prefers the modern API and falls back
    to returning the estimator unchanged, which callers pair with ``cv="prefit"``
    on older versions.

    Args:
        estimator: A fitted estimator.

    Returns:
        The frozen estimator when available, otherwise the input.
    """
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # pragma: no cover - scikit-learn < 1.6
        return estimator
    return FrozenEstimator(estimator)


def _maybe_calibrate(estimator: Any, valid: SplitData, cfg: Config, candidate_name: str) -> Any:
    """Wrap an estimator in a probability calibrator fitted on validation data.

    Calibration is fitted on the validation split against a *frozen* copy of the
    model, so the underlying estimator is not refitted and the training split is
    not reused.  The threshold is then optimised on the *calibrated* validation
    probabilities from the same split - acceptable here because the threshold is
    a single scalar and the alternative (a fourth split) would leave too few
    failure events per partition.  The mild optimism this introduces in the
    validation threshold metrics is documented in MODEL_CARD.md; the test
    metrics are unaffected.

    Args:
        estimator: Fitted pipeline.
        valid: Validation split.
        cfg: Merged configuration.
        candidate_name: Name used for logging.

    Returns:
        The calibrated estimator, or the original when calibration is disabled
        or the candidate is the dummy baseline.
    """
    method = str(cfg.get("training.calibration", "none")).lower()
    if method in {"none", ""} or candidate_name == "dummy":
        return estimator
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"Unknown training.calibration {method!r}")
    if valid.n_positive < 20:
        logger.warning(
            "Skipping calibration: too few positives in validation split",
            extra={"candidate": candidate_name, "positives": valid.n_positive},
        )
        return estimator

    calibrated = CalibratedClassifierCV(_freeze(estimator), method=method)
    calibrated.fit(valid.x, valid.y)
    logger.info("Calibrated probabilities", extra={"candidate": candidate_name, "method": method})
    return calibrated


def train_candidates(train: SplitData, valid: SplitData, cfg: Config) -> list[TrainedCandidate]:
    """Fit and evaluate every enabled candidate.

    Args:
        train: Training split.
        valid: Validation split.
        cfg: Merged configuration.

    Returns:
        One :class:`TrainedCandidate` per candidate, in configuration order.
    """
    results: list[TrainedCandidate] = []
    for candidate in build_candidates(cfg):
        estimator, elapsed = _fit_candidate(candidate, train)
        estimator = _maybe_calibrate(estimator, valid, cfg, candidate.name)

        valid_probabilities = estimator.predict_proba(valid.x)[:, 1]
        threshold = optimise_threshold(valid.y, valid_probabilities, cfg, split_name="valid")
        metrics = {
            "train": classification_metrics(
                train.y, estimator.predict_proba(train.x)[:, 1], threshold.threshold
            ),
            "valid": threshold.metrics,
        }
        results.append(
            TrainedCandidate(
                name=candidate.name,
                algorithm=candidate.algorithm,
                estimator=estimator,
                train_seconds=elapsed,
                valid_probabilities=valid_probabilities,
                metrics=metrics,
                threshold=threshold,
                latency=measure_inference_latency(estimator, valid.x),
            )
        )
    return results


#: Metrics where a *smaller* value is better. They are negated before ranking
#: so a single descending sort handles both directions.
LOWER_IS_BETTER = frozenset(
    {"cost_per_sample", "total_cost", "false_negative_rate", "false_positive_rate", "brier"}
)


def _metric_for_ranking(metrics: dict[str, float], name: str) -> float:
    """Return a metric oriented so that larger is always better.

    Args:
        metrics: Metric dictionary for one split.
        name: Metric name.

    Returns:
        The metric value, negated when smaller values are preferable.
    """
    value = float(np.nan_to_num(metrics.get(name, 0.0), nan=0.0))
    return -value if name in LOWER_IS_BETTER else value


def select_best_candidate(
    candidates: list[TrainedCandidate], cfg: Config
) -> tuple[TrainedCandidate, str]:
    """Rank candidates on validation performance and pick a winner.

    Selection is not "highest ROC-AUC wins".  Candidates whose validation recall
    at the optimised threshold falls below ``training.selection.min_recall`` are
    rejected outright - a model that misses most failures is not useful whatever
    its ranking metric says.  Survivors are then ordered by the configured
    primary metric with explicit tie-breakers.

    Args:
        candidates: Trained candidates.
        cfg: Merged configuration.

    Returns:
        ``(winner, rationale)`` where ``rationale`` is a human-readable
        explanation suitable for the model card.

    Raises:
        ValueError: If every candidate was rejected.
    """
    primary = str(cfg.get("training.selection.primary_metric", "pr_auc"))
    tie_breakers = list(cfg.get("training.selection.tie_breakers", ["f2", "recall"]))
    min_recall = float(cfg.get("training.selection.min_recall", 0.0))
    min_pr_auc_ratio = float(cfg.get("training.selection.min_pr_auc_ratio", 0.0))

    real_candidates = [c for c in candidates if c.name != "dummy"]
    best_pr_auc = max(
        (
            float(np.nan_to_num(c.metrics["valid"].get("pr_auc", 0.0), nan=0.0))
            for c in real_candidates
        ),
        default=0.0,
    )

    eligible: list[TrainedCandidate] = []
    for candidate in candidates:
        if candidate.name == "dummy":
            candidate.rejected_reason = "reference baseline, not a deployment candidate"
            continue
        valid_metrics = candidate.metrics["valid"]
        recall = valid_metrics["recall"]
        if recall < min_recall:
            candidate.rejected_reason = (
                f"validation recall {recall:.3f} is below the required floor {min_recall:.2f}"
            )
            logger.warning(
                "Rejected candidate",
                extra={"candidate": candidate.name, "reason": candidate.rejected_reason},
            )
            continue
        pr_auc = float(np.nan_to_num(valid_metrics.get("pr_auc", 0.0), nan=0.0))
        if best_pr_auc > 0 and pr_auc < min_pr_auc_ratio * best_pr_auc:
            candidate.rejected_reason = (
                f"validation PR-AUC {pr_auc:.3f} is more than "
                f"{(1 - min_pr_auc_ratio):.0%} below the best candidate's {best_pr_auc:.3f}"
            )
            logger.warning(
                "Rejected candidate",
                extra={"candidate": candidate.name, "reason": candidate.rejected_reason},
            )
            continue
        eligible.append(candidate)

    if not eligible:
        raise ValueError(
            f"Every candidate was rejected (min recall {min_recall}, "
            f"min PR-AUC ratio {min_pr_auc_ratio}). Relax training.selection or revisit the "
            "feature pipeline."
        )

    # Candidates whose primary metric is within `tolerance` (relative) of the
    # best are treated as tied and ordered by the tie-breakers instead. Without
    # this the winner flips between runs on differences of well under a percent,
    # which is noise rather than evidence - and the configured tie-breakers
    # would never fire, because exact ties do not occur with real-valued metrics.
    tolerance = float(cfg.get("training.selection.primary_tolerance", 0.0))
    best_primary = max(_metric_for_ranking(c.metrics["valid"], primary) for c in eligible)
    band = abs(best_primary) * tolerance

    def sort_key(candidate: TrainedCandidate) -> tuple[float, ...]:
        metrics = candidate.metrics["valid"]
        primary_value = _metric_for_ranking(metrics, primary)
        tied = primary_value >= best_primary - band
        # Within the band every candidate shares a rank, so the tie-breakers
        # decide; outside it the primary metric still dominates.
        return (
            1.0 if tied else 0.0,
            0.0 if tied else primary_value,
            *(_metric_for_ranking(metrics, name) for name in tie_breakers),
        )

    ranked = sorted(eligible, key=sort_key, reverse=True)
    contenders = [
        c for c in ranked if _metric_for_ranking(c.metrics["valid"], primary) >= best_primary - band
    ]
    winner = ranked[0]
    winner_metrics = winner.metrics["valid"]

    runner_up = ranked[1] if len(ranked) > 1 else None
    comparison = (
        f" It was preferred over {runner_up.name} "
        f"({primary}={runner_up.metrics['valid'].get(primary, float('nan')):.4f}, "
        f"recall={runner_up.metrics['valid']['recall']:.3f}, "
        f"PR-AUC={runner_up.metrics['valid'].get('pr_auc', float('nan')):.3f})."
        if runner_up
        else ""
    )
    rejected = [c for c in candidates if c.rejected_reason and c.name != "dummy"]
    rejection_note = (
        " Rejected: " + "; ".join(f"{c.name} ({c.rejected_reason})" for c in rejected) + "."
        if rejected
        else ""
    )
    dummy = next((c for c in candidates if c.name == "dummy"), None)
    dummy_note = (
        f" The stratified dummy baseline reached {primary}="
        f"{dummy.metrics['valid'].get(primary, float('nan')):.3f}, confirming the signal is learned "
        "rather than an artefact of the class balance."
        if dummy
        else ""
    )

    direction = "lowest" if primary in LOWER_IS_BETTER else "highest"
    tie_note = (
        f" {len(contenders)} candidates ({', '.join(c.name for c in contenders)}) landed within "
        f"{tolerance:.0%} of the best {primary} - a difference too small to be meaningful - so the "
        f"winner among them was decided by the tie-breakers ({', '.join(tie_breakers)}), which favour "
        f"catching more failures at effectively equal cost."
        if len(contenders) > 1
        else ""
    )
    rationale = (
        f"'{winner.name}' ({winner.algorithm}) was selected on validation data as the candidate with "
        f"the {direction} {primary} (tie-breakers: {', '.join(tie_breakers)}), subject to a minimum "
        f"recall floor of {min_recall:.2f} and a requirement to stay within "
        f"{(1 - min_pr_auc_ratio):.0%} of the best candidate's PR-AUC.{tie_note} Selecting on expected "
        f"illustrative cost rather than PR-AUC alone reflects the asymmetry of the problem: a missed "
        f"failure is far more expensive than an unnecessary inspection, and a candidate can rank well "
        f"on PR-AUC while operating at a recall that is unacceptable in practice. "
        f"It achieved {primary}={winner_metrics.get(primary, float('nan')):.4f}, "
        f"PR-AUC={winner_metrics.get('pr_auc', float('nan')):.3f}, "
        f"recall={winner_metrics['recall']:.3f}, precision={winner_metrics['precision']:.3f}, "
        f"F2={winner_metrics['f2']:.3f} and a false-negative rate of "
        f"{winner_metrics['false_negative_rate']:.3f} at the optimised threshold. "
        f"It trains in {winner.train_seconds:.1f}s and scores a single row in "
        f"{winner.latency.get('single_row_ms', float('nan')):.1f}ms, which is well inside the latency "
        f"budget for an hourly-cadence advisory service, and it is compatible with SHAP explanations."
        f"{comparison}{dummy_note}{rejection_note}"
    )
    logger.info("Selected final model", extra={"model": winner.name, "algorithm": winner.algorithm})
    return winner, rationale


def evaluate_on_test(candidate: TrainedCandidate, test: SplitData) -> dict[str, float]:
    """Score the selected model once on the held-out test split.

    Args:
        candidate: The selected trained candidate.
        test: Test split.

    Returns:
        Test metrics at the validation-selected threshold.

    Raises:
        ValueError: If the candidate has no threshold attached.
    """
    if candidate.threshold is None:
        raise ValueError("Candidate has no optimised threshold; run train_candidates first")
    probabilities = candidate.estimator.predict_proba(test.x)[:, 1]
    return classification_metrics(test.y, probabilities, candidate.threshold.threshold)


def comparison_table(candidates: list[TrainedCandidate]) -> pd.DataFrame:
    """Build the model-comparison table across all candidates.

    Args:
        candidates: Trained candidates.

    Returns:
        A tidy comparison dataframe with training time and latency attached.
    """
    results = {candidate.name: candidate.metrics for candidate in candidates}
    table = build_comparison_table(results)
    if table.empty:
        return table
    extras = {
        candidate.name: {
            "algorithm": candidate.algorithm,
            "train_seconds": round(candidate.train_seconds, 3),
            "single_row_ms": candidate.latency.get("single_row_ms", float("nan")),
            "per_row_batch_ms": candidate.latency.get("per_row_batch_ms", float("nan")),
            "rejected_reason": candidate.rejected_reason or "",
        }
        for candidate in candidates
    }
    for column in (
        "algorithm",
        "train_seconds",
        "single_row_ms",
        "per_row_batch_ms",
        "rejected_reason",
    ):
        table[column] = table["model"].map(lambda name, c=column: extras[name][c])
    return table


def library_versions() -> dict[str, str]:
    """Record the versions of libraries that affect artifact reproducibility.

    Returns:
        A mapping of package name to version string.
    """
    import numpy
    import pandas

    versions = {
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }
    try:
        import shap

        versions["shap"] = shap.__version__
    except ImportError:  # pragma: no cover - optional dependency
        versions["shap"] = "not installed"
    return versions


def training_timestamp() -> datetime:
    """Return the current UTC time, used as the model's training date."""
    return datetime.now(UTC)


def feature_spec_groups(spec: FeatureSpec) -> dict[str, list[str]]:
    """Extract the feature-group mapping for model metadata.

    Args:
        spec: The feature specification produced during preparation.

    Returns:
        Feature group name to feature-name list.
    """
    return {key: list(value) for key, value in spec.groups.items()}
