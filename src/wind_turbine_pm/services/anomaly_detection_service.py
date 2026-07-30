"""Single anomaly-detection path shared by API, dashboard and maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_pm.anomaly.config import get_anomaly_config
from wind_turbine_pm.anomaly.features import build_anomaly_features
from wind_turbine_pm.anomaly.modeling import percentile_scores, raw_novelty_score
from wind_turbine_pm.anomaly.persistence import (
    AnomalyBundle,
    bundle_available,
    load_bundle,
)
from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    OPERATING_REGIME,
    SENSOR_COLUMNS,
    TIMESTAMP,
    TURBINE_ID,
    OperatingRegime,
)
from wind_turbine_pm.contracts.anomaly import (
    AnomalyPrediction,
    BatchAnomalyPrediction,
)
from wind_turbine_pm.contracts.observations import TurbineWindow
from wind_turbine_pm.contracts.predictions import RiskFactor
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.health.regimes import attach_regimes
from wind_turbine_pm.services.failure_prediction_service import (
    InsufficientHistoryError,
    ServiceNotReadyError,
)
from wind_turbine_pm.utils.io import ArtifactNotFoundError

PIPELINE_COMMAND = "python scripts/run_anomaly_pipeline.py"


@dataclass(frozen=True)
class AnomalyServiceStatus:
    """Artifact status exposed by the platform health endpoint."""

    model_loaded: bool
    model_version: str | None
    algorithm: str | None
    n_features: int
    warning_threshold: float | None
    alarm_threshold: float | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "n_features": self.n_features,
            "warning_threshold": self.warning_threshold,
            "alarm_threshold": self.alarm_threshold,
            "detail": self.detail,
        }


class AnomalyDetectionService:
    """Load the selected novelty detector and assemble honest assessments."""

    def __init__(self, cfg: Config | None = None, eager: bool = False) -> None:
        self._cfg = cfg if cfg is not None else get_anomaly_config()
        self._bundle: AnomalyBundle | None = None
        self._load_error: str | None = None
        if eager:
            self._ensure_loaded()

    @property
    def config(self) -> Config:
        return self._cfg

    def _ensure_loaded(self) -> AnomalyBundle:
        if self._bundle is not None:
            return self._bundle
        try:
            self._bundle = load_bundle(self._cfg)
        except ArtifactNotFoundError as exc:
            self._load_error = str(exc)
            raise ServiceNotReadyError(
                "Anomaly model artifacts are not available.", exc.hint
            ) from exc
        self._load_error = None
        return self._bundle

    @property
    def is_ready(self) -> bool:
        if self._bundle is not None:
            return True
        if not bundle_available(self._cfg):
            return False
        try:
            self._ensure_loaded()
        except ServiceNotReadyError:
            return False
        return True

    @property
    def metadata(self):
        return self._ensure_loaded().metadata

    @property
    def calibration(self):
        """Return the validated empirical score calibration."""
        return self._ensure_loaded().calibration

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata.features)

    def minimum_history_hours(self) -> float:
        return float(self._cfg.get("anomaly.serving.min_history_hours", 72))

    def status(self) -> AnomalyServiceStatus:
        if not self.is_ready:
            return AnomalyServiceStatus(
                model_loaded=False,
                model_version=None,
                algorithm=None,
                n_features=0,
                warning_threshold=None,
                alarm_threshold=None,
                detail=self._load_error or f"Anomaly artifacts not found. Run: {PIPELINE_COMMAND}",
            )
        bundle = self._ensure_loaded()
        return AnomalyServiceStatus(
            model_loaded=True,
            model_version=bundle.metadata.model_version,
            algorithm=bundle.metadata.algorithm,
            n_features=bundle.metadata.n_features,
            warning_threshold=bundle.calibration.warning_percentile,
            alarm_threshold=bundle.calibration.alarm_percentile,
            detail="Anomaly artifacts loaded.",
        )

    def _data_quality(self, raw: pd.DataFrame) -> float:
        sensors = [column for column in SENSOR_COLUMNS if column in raw]
        if not sensors:
            return 0.0
        recent = raw.tail(24)
        return float(recent[sensors].notna().mean().mean())

    def _contributing_signals(self, row: pd.Series, bundle: AnomalyBundle) -> list[RiskFactor]:
        reference = bundle.reference.set_index("_statistic")
        if not {"median", "q1", "q3"}.issubset(reference.index):
            return []
        median = reference.loc["median"]
        scale = (reference.loc["q3"] - reference.loc["q1"]).abs().clip(lower=1e-6)
        impacts: list[tuple[float, RiskFactor]] = []
        for feature in bundle.metadata.features:
            value = float(row.get(feature, np.nan))
            centre = float(median.get(feature, np.nan))
            spread = float(scale.get(feature, np.nan))
            if not np.isfinite(value) or not np.isfinite(centre) or not np.isfinite(spread):
                continue
            deviation = abs(value - centre) / spread
            if deviation < 1.0:
                continue
            impacts.append(
                (
                    deviation,
                    RiskFactor(
                        feature=feature,
                        impact=round(float(deviation), 6),
                        direction="increases_risk",
                        value=round(value, 6),
                        description=(
                            f"{feature.replace('_', ' ')} is {deviation:.1f} healthy-reference "
                            "IQRs from the training median; this is association evidence, not a "
                            "causal diagnosis."
                        ),
                    ),
                )
            )
        impacts.sort(key=lambda item: -item[0])
        limit = int(self._cfg.get("anomaly.serving.top_k_signals", 5))
        return [factor for _, factor in impacts[:limit]]

    def _predict_row(
        self,
        row: pd.DataFrame,
        observation: pd.Series,
        data_quality: float,
    ) -> AnomalyPrediction:
        bundle = self._ensure_loaded()
        regime = OperatingRegime(
            str(observation.get(OPERATING_REGIME, OperatingRegime.MEDIUM_LOAD))
        )
        timestamp = pd.Timestamp(observation[TIMESTAMP]).to_pydatetime()
        turbine_id = str(observation[TURBINE_ID])
        state = str(observation.get("operational_status", "normal"))
        if state in {"fault", "maintenance"}:
            return AnomalyPrediction(
                turbine_id=turbine_id,
                timestamp=timestamp,
                model_version=bundle.metadata.model_version,
                assessment_state="not_applicable",
                severity="not_applicable",
                operating_regime=OperatingRegime.OFFLINE,
                data_quality=round(data_quality, 4),
                explanation=(
                    f"Anomaly scoring is not applicable while the controller reports {state}; "
                    "the known operating state takes precedence."
                ),
            )

        aligned = row.reindex(columns=bundle.metadata.features)
        raw = float(raw_novelty_score(bundle.estimator, aligned)[0])
        score = float(percentile_scores(np.array([raw]), bundle.calibration)[0])
        if score >= bundle.calibration.alarm_percentile:
            severity = "alarm"
        elif score >= bundle.calibration.warning_percentile:
            severity = "warning"
        else:
            severity = "normal"
        signals = self._contributing_signals(aligned.iloc[0], bundle)
        explanation = (
            f"Current behaviour is at the {score:.1%} percentile of novelty relative to "
            "healthy validation history."
        )
        if signals:
            explanation += (
                " Largest associated deviations: "
                + ", ".join(factor.feature.replace("_", " ") for factor in signals[:3])
                + "."
            )
        return AnomalyPrediction(
            turbine_id=turbine_id,
            timestamp=timestamp,
            model_version=bundle.metadata.model_version,
            raw_anomaly_score=round(raw, 6),
            anomaly_score=round(score, 6),
            is_anomaly=severity != "normal",
            severity=severity,
            warning_threshold=bundle.calibration.warning_percentile,
            alarm_threshold=bundle.calibration.alarm_percentile,
            operating_regime=regime,
            data_quality=round(data_quality, 4),
            contributing_signals=signals,
            explanation=explanation,
        )

    def detect_from_window(self, window: TurbineWindow) -> AnomalyPrediction:
        self._ensure_loaded()
        raw = window.to_frame()
        span = (
            (raw[TIMESTAMP].iloc[-1] - raw[TIMESTAMP].iloc[0]) / pd.Timedelta("1h")
            if len(raw) > 1
            else 0.0
        )
        required = self.minimum_history_hours()
        if span < required:
            raise InsufficientHistoryError(
                f"Turbine {window.turbine_id}: window spans {span:.1f}h but anomaly detection "
                f"requires at least {required:.0f}h."
            )
        data_quality = self._data_quality(raw)
        prepared, _ = preprocess(raw, self._cfg)
        prepared = attach_regimes(prepared, self._cfg)
        features, _ = build_anomaly_features(prepared, self._cfg)
        last = features.index[-1]
        return self._predict_row(
            features.loc[[last]], prepared.loc[last], data_quality=data_quality
        )

    def detect_batch_from_windows(self, windows: list[TurbineWindow]) -> BatchAnomalyPrediction:
        limit = int(self._cfg.get("anomaly.serving.max_batch_turbines", 200))
        if len(windows) > limit:
            raise ValueError(f"Batch of {len(windows)} exceeds the configured limit {limit}")
        predictions = [self.detect_from_window(window) for window in windows]
        return BatchAnomalyPrediction(
            predictions=predictions,
            model_version=self.model_version,
            count=len(predictions),
        )

    def score_frame(self, frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        bundle = self._ensure_loaded()
        aligned = features.reindex(columns=bundle.metadata.features)
        raw = raw_novelty_score(bundle.estimator, aligned)
        scores = percentile_scores(raw, bundle.calibration)
        severity = np.where(
            scores >= bundle.calibration.alarm_percentile,
            "alarm",
            np.where(scores >= bundle.calibration.warning_percentile, "warning", "normal"),
        )
        return pd.DataFrame(
            {
                TURBINE_ID: frame[TURBINE_ID].to_numpy(),
                TIMESTAMP: pd.to_datetime(frame[TIMESTAMP]).to_numpy(),
                "raw_anomaly_score": raw,
                "anomaly_score": scores,
                "is_anomaly": severity != "normal",
                "anomaly_severity": severity,
            },
            index=frame.index,
        )


@lru_cache(maxsize=1)
def get_anomaly_service() -> AnomalyDetectionService:
    return AnomalyDetectionService()
