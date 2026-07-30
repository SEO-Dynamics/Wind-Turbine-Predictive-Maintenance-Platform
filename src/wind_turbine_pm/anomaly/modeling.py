"""One-class candidates, empirical calibration and honest model selection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from wind_turbine_pm.config import Config
from wind_turbine_pm.contracts.anomaly import AnomalyCalibration


@dataclass(frozen=True)
class CandidateResult:
    """A fitted one-class candidate and its validation evidence."""

    name: str
    algorithm: str
    estimator: Pipeline
    calibration: AnomalyCalibration
    metrics: dict[str, float]
    latency_ms: float
    valid_scores: np.ndarray


def _params(cfg: Config, name: str) -> dict[str, Any]:
    value = cfg.get(f"anomaly.training.candidates.{name}.params", {})
    return dict(value) if value is not None else {}


def build_candidates(cfg: Config) -> dict[str, Pipeline]:
    """Build every enabled novelty estimator with train-only preprocessing."""
    seed = int(cfg.get("random_seed", 42))
    candidates: dict[str, Pipeline] = {}
    configured = cfg.require("anomaly.training.candidates")
    for name in configured:
        if not bool(cfg.get(f"anomaly.training.candidates.{name}.enabled", False)):
            continue
        params = _params(cfg, str(name))
        if name == "isolation_forest":
            model = IsolationForest(random_state=seed, **params)
        elif name == "local_outlier_factor":
            model = LocalOutlierFactor(**params)
        elif name == "one_class_svm":
            model = OneClassSVM(**params)
        else:
            raise ValueError(f"Unknown anomaly candidate {name!r}")
        candidates[str(name)] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler(quantile_range=(5.0, 95.0))),
                ("model", model),
            ]
        )
    if not candidates:
        raise ValueError("No anomaly candidates are enabled")
    return candidates


def raw_novelty_score(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return scores oriented so larger always means more anomalous."""
    return -np.asarray(estimator.decision_function(features), dtype=float).reshape(-1)


def fit_calibration(
    healthy_scores: np.ndarray, warning_rate: float, alarm_rate: float
) -> AnomalyCalibration:
    """Fit empirical warning/alarm limits on healthy validation scores."""
    scores = np.asarray(healthy_scores, dtype=float)
    scores = np.sort(scores[np.isfinite(scores)])
    if scores.size < 20:
        raise ValueError("At least 20 healthy validation scores are required for calibration")
    warning_raw = float(np.quantile(scores, 1.0 - warning_rate))
    alarm_raw = float(np.quantile(scores, 1.0 - alarm_rate))
    return AnomalyCalibration(
        reference_scores=scores.tolist(),
        warning_raw=warning_raw,
        alarm_raw=alarm_raw,
        warning_percentile=1.0 - warning_rate,
        alarm_percentile=1.0 - alarm_rate,
        achieved_warning_rate=float((scores >= warning_raw).mean()),
        achieved_alarm_rate=float((scores >= alarm_raw).mean()),
    )


def percentile_scores(raw_scores: np.ndarray, calibration: AnomalyCalibration) -> np.ndarray:
    """Map raw scores through the healthy validation empirical CDF."""
    reference = np.asarray(calibration.reference_scores, dtype=float)
    ranks = np.searchsorted(reference, np.asarray(raw_scores, dtype=float), side="right")
    return np.clip(ranks / max(reference.size, 1), 0.0, 1.0)


def anomaly_metrics(
    truth: np.ndarray, scores: np.ndarray, calibration: AnomalyCalibration
) -> dict[str, float]:
    """Calculate ranking and warning-threshold metrics."""
    y = np.asarray(truth, dtype=int)
    predicted = scores >= calibration.warning_percentile
    base_rate = float(y.mean()) if y.size else 0.0
    return {
        "n_samples": float(y.size),
        "base_rate": base_rate,
        "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else 0.0,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "alert_rate": float(predicted.mean()),
        "healthy_alert_rate": float(predicted[y == 0].mean()) if (y == 0).any() else 0.0,
        "alarm_rate": float((scores >= calibration.alarm_percentile).mean()),
    }


def _latency(estimator: Pipeline, features: pd.DataFrame) -> float:
    sample = features.iloc[: min(len(features), 256)]
    start = time.perf_counter()
    raw_novelty_score(estimator, sample)
    return (time.perf_counter() - start) * 1000.0 / max(len(sample), 1)


def train_candidates(
    train_reference: pd.DataFrame,
    valid_features: pd.DataFrame,
    valid_truth: np.ndarray,
    valid_healthy: np.ndarray,
    cfg: Config,
) -> list[CandidateResult]:
    """Fit candidates on healthy train data and compare on validation only."""
    warning_rate = float(cfg.require("anomaly.calibration.warning_healthy_rate"))
    alarm_rate = float(cfg.require("anomaly.calibration.alarm_healthy_rate"))
    results: list[CandidateResult] = []
    for name, estimator in build_candidates(cfg).items():
        estimator.fit(train_reference)
        raw = raw_novelty_score(estimator, valid_features)
        calibration = fit_calibration(raw[valid_healthy], warning_rate, alarm_rate)
        scores = percentile_scores(raw, calibration)
        metrics = anomaly_metrics(valid_truth, scores, calibration)
        results.append(
            CandidateResult(
                name=name,
                algorithm=type(estimator.named_steps["model"]).__name__,
                estimator=estimator,
                calibration=calibration,
                metrics=metrics,
                latency_ms=_latency(estimator, valid_features),
                valid_scores=scores,
            )
        )
    return results


def select_best(results: list[CandidateResult]) -> tuple[CandidateResult, str]:
    """Select highest recall, then PR-AUC/F2, then lower latency."""
    if not results:
        raise ValueError("No trained anomaly candidates")
    ranked = sorted(
        results,
        key=lambda item: (
            item.metrics["recall"],
            item.metrics["pr_auc"],
            item.metrics["f2"],
            -item.latency_ms,
        ),
        reverse=True,
    )
    winner = ranked[0]
    rationale = (
        f"Selected {winner.name} ({winner.algorithm}) on validation recall at the empirically "
        f"calibrated healthy warning rate: recall={winner.metrics['recall']:.3f}, "
        f"PR-AUC={winner.metrics['pr_auc']:.3f}, F2={winner.metrics['f2']:.3f}, "
        f"healthy alert rate={winner.metrics['healthy_alert_rate']:.3f}. "
        "Test data did not participate in selection."
    )
    return winner, rationale


def deterministic_reference_sample(
    features: pd.DataFrame, keys: pd.DataFrame, maximum: int, seed: int
) -> pd.DataFrame:
    """Bound expensive one-class fitting while retaining fleet/regime coverage."""
    if len(features) <= maximum:
        return features
    frame = keys[["turbine_id", "operating_regime"]].copy()
    frame["_index"] = features.index
    group_count = max(int(frame.groupby(["turbine_id", "operating_regime"]).ngroups), 1)
    per_group = max(maximum // group_count, 1)
    pieces = [
        group.sample(n=min(per_group, len(group)), replace=False, random_state=seed)
        for _, group in frame.groupby(["turbine_id", "operating_regime"], sort=True)
    ]
    sampled = pd.concat(pieces).head(maximum)
    if len(sampled) < maximum:
        remaining = frame.loc[~frame["_index"].isin(sampled["_index"])]
        extra = remaining.sample(n=min(maximum - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra])
    return features.loc[sampled["_index"].tolist()]
