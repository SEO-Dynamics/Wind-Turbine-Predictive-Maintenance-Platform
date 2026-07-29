"""The single assessment entry point for the Turbine Health Monitoring Module.

Both the FastAPI layer and the Streamlit dashboard use this class, for the same
reason the Failure Prediction Module has one service: an assessment computed at
serving time must be computed exactly as it was at training time, and the only
way to guarantee that is to have one implementation.

Two entry points, for two different callers:

:meth:`HealthMonitoringService.assess_from_window`
    Takes the platform's existing input contract - a
    :class:`~wind_turbine_pm.contracts.observations.TurbineWindow` of raw
    observations - and computes features itself with
    :func:`~wind_turbine_pm.health.health_features.build_health_features`.  This
    is the API path.
:meth:`HealthMonitoringService.assess_from_prepared`
    Takes an already-prepared frame and feature matrix.  This is the dashboard
    path, and it exists so the detail panel and the fleet table cannot disagree:
    both are then derived from the same feature matrix.

Note that the two can legitimately return different scores for the same turbine
at the same timestamp, because expanding baselines and drift statistics depend on
how much history the caller supplied.  A short window is not wrong, it is simply
less informed - which is what ``health.serving.min_history_hours`` bounds.

Neither entry point accepts a bare feature vector.  Unlike a failure
probability, an assessment reports component scores and rule violations computed
from *raw* readings, so a caller posting only features could not be given a
complete assessment, and silently returning a partial one would be worse than
not offering the mode.

Assembling one assessment
-------------------------
1. Preprocess the window and build the leakage-safe feature matrix, which also
   assigns the operating regime.
2. Score the **last** row with the published regression model - the raw health
   score.
3. Summarise the drift statistics for the window and deduct the (capped) drift
   penalty, giving the published score.
4. Band the published score into a :class:`~wind_turbine_pm.constants.HealthClass`.
5. Attribute the condition to named components from rule margins and
   regime-relative deviations, and report the sensor rules that fired.
6. Compose a narrative and an advisory recommendation from that evidence.

Artifacts are loaded lazily on first use and cached, so importing this module
never touches the filesystem and never triggers training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.constants import (
    HEALTH_CLASS,
    HEALTH_SCORE,
    OPERATING_REGIME,
    SENSOR_DISPLAY_NAMES,
    TIMESTAMP,
    TURBINE_ID,
    DriftSeverity,
    OperatingRegime,
)
from wind_turbine_pm.contracts.health import (
    BatchHealthAssessment,
    ComponentHealth,
    FleetHealthSummary,
    HealthAssessment,
    SensorDriftSignal,
    SensorRuleViolation,
)
from wind_turbine_pm.contracts.observations import TurbineWindow
from wind_turbine_pm.contracts.predictions import RiskFactor
from wind_turbine_pm.data.preprocessing import preprocess
from wind_turbine_pm.health.components import build_component_health
from wind_turbine_pm.health.config import get_health_config
from wind_turbine_pm.health.drift import (
    DriftCalibration,
    DriftSettings,
    compute_drift_statistics,
    drift_penalty,
    summarise_drift,
)
from wind_turbine_pm.health.health_class import (
    ClassBands,
    apply_drift_penalty,
    class_distribution,
    classify_health,
    classify_health_series,
)
from wind_turbine_pm.health.health_features import (
    align_to_feature_order,
    build_health_features,
    minimum_history_hours,
)
from wind_turbine_pm.health.health_score import predict_health_scores
from wind_turbine_pm.health.narratives import build_health_advisory, humanise_health_feature
from wind_turbine_pm.health.persistence import (
    HealthBundle,
    health_bundle_available,
    load_health_bundle,
)
from wind_turbine_pm.health.sensor_rules import (
    SensorRule,
    evaluate_rules,
    load_sensor_rules,
    rules_to_records,
)
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.models.persistence import check_runtime_compatibility
from wind_turbine_pm.services.failure_prediction_service import (
    InsufficientHistoryError,
    ServiceNotReadyError,
)
from wind_turbine_pm.utils.io import ArtifactNotFoundError

logger = get_logger(__name__)

#: Command that produces the health artifacts.
HEALTH_PIPELINE_COMMAND = "python scripts/run_health_pipeline.py"


@dataclass(frozen=True)
class HealthServiceStatus:
    """Readiness information about the service and its artifacts."""

    model_loaded: bool
    model_version: str | None
    algorithm: str | None
    n_features: int
    drift_detector_loaded: bool
    class_bands: dict[str, float]
    detail: str
    runtime_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "n_features": self.n_features,
            "drift_detector_loaded": self.drift_detector_loaded,
            "class_bands": dict(self.class_bands),
            "detail": self.detail,
            "runtime_warnings": list(self.runtime_warnings),
        }


class HealthMonitoringService:
    """Loads the published health bundle and turns turbine data into assessments."""

    def __init__(self, cfg: Config | None = None, eager: bool = False) -> None:
        """Create the service.

        Args:
            cfg: Merged configuration carrying the ``health`` namespace; loaded
                from disk when omitted.
            eager: Load artifacts immediately instead of on first use.  Useful
                in tests; the API deliberately stays lazy so a missing model
                does not prevent the process from starting.
        """
        self._cfg = cfg if cfg is not None else get_health_config()
        self._bundle: HealthBundle | None = None
        self._rules: dict[str, SensorRule] | None = None
        self._load_error: str | None = None
        if eager:
            self._ensure_loaded()

    # -- Configuration accessors -------------------------------------------
    @property
    def config(self) -> Config:
        """The configuration this service was built with."""
        return self._cfg

    @property
    def class_bands(self) -> ClassBands:
        """The resolved health-class score boundaries."""
        return ClassBands.from_config(self._cfg)

    @property
    def drift_calibration(self) -> DriftCalibration | None:
        """Persistence thresholds recorded in the published artifact.

        Returns:
            The calibration, or ``None`` when the artifact carries none (in
            which case the configured fallback thresholds apply).
        """
        if not self.is_ready:
            return None
        return DriftCalibration.from_dict(self.metadata.drift.get("calibration"))

    @property
    def rules(self) -> dict[str, SensorRule]:
        """The loaded sensor rule set, cached for the service's lifetime."""
        if self._rules is None:
            self._rules = load_sensor_rules(self._cfg)
        return self._rules

    def sensor_rule_records(self) -> list[dict[str, Any]]:
        """Return the rule set as JSON-serialisable records.

        Returns:
            One record per configured sensor rule, including its provenance.
        """
        return rules_to_records(self.rules)

    def minimum_history_hours(self) -> float:
        """Minimum window length in hours accepted by the assessment endpoints.

        The configured serving minimum is raised to the longest feature window
        when necessary, so a window can never be accepted that leaves a
        configured feature undefined.

        Returns:
            The required history in hours.
        """
        configured = float(self._cfg.get("health.serving.min_history_hours", 72))
        return max(configured, minimum_history_hours(self._cfg))

    # -- Artifact loading --------------------------------------------------
    def _ensure_loaded(self) -> HealthBundle:
        """Load and cache the health bundle, raising a helpful error on failure."""
        if self._bundle is not None:
            return self._bundle
        try:
            self._bundle = load_health_bundle(self._cfg)
        except ArtifactNotFoundError as exc:
            self._load_error = str(exc)
            raise ServiceNotReadyError(
                "Health model artifacts are not available.", exc.hint
            ) from exc
        self._load_error = None
        return self._bundle

    @property
    def is_ready(self) -> bool:
        """Whether artifacts are present and loadable without raising."""
        if self._bundle is not None:
            return True
        if not health_bundle_available(self._cfg):
            return False
        try:
            self._ensure_loaded()
        except ServiceNotReadyError:
            return False
        return True

    @property
    def metadata(self):
        """The loaded model's metadata document."""
        return self._ensure_loaded().metadata

    @property
    def feature_names(self) -> list[str]:
        """Feature names in the order the model expects them."""
        return self._ensure_loaded().features

    @property
    def model_version(self) -> str:
        """The published model version."""
        return self._ensure_loaded().version

    def status(self) -> HealthServiceStatus:
        """Report readiness without raising, for the ``/health`` endpoint.

        Returns:
            The current :class:`HealthServiceStatus`.
        """
        if not self.is_ready:
            return HealthServiceStatus(
                model_loaded=False,
                model_version=None,
                algorithm=None,
                n_features=0,
                drift_detector_loaded=False,
                class_bands=self.class_bands.to_dict(),
                detail=self._load_error
                or f"Health model artifacts not found. Run: {HEALTH_PIPELINE_COMMAND}",
            )
        bundle = self._ensure_loaded()
        metadata = bundle.metadata
        return HealthServiceStatus(
            model_loaded=True,
            model_version=metadata.model_version,
            algorithm=metadata.algorithm,
            n_features=metadata.n_features,
            drift_detector_loaded=bundle.drift_detector is not None,
            class_bands=self.class_bands.to_dict(),
            detail="Health model artifacts loaded.",
            runtime_warnings=check_runtime_compatibility(metadata),  # type: ignore[arg-type]
        )

    # -- Internal assembly -------------------------------------------------
    def _drift_statistics(self, prepared: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Return the drift statistics for a window.

        The feature builder already computes the drift columns when
        ``health.features.drift_features`` is enabled, so they are lifted from
        the feature matrix instead of being recomputed.  When the feature group
        is disabled but drift reporting is not, they are computed here.

        Args:
            prepared: The preprocessed frame.
            features: The feature matrix built from it.

        Returns:
            A frame of drift statistics aligned to ``prepared``, empty when
            drift detection is switched off.
        """
        if not bool(self._cfg.get("health.drift.enabled", True)):
            return pd.DataFrame(index=prepared.index)

        suffixes = ("_drift_z", "_cusum_pos", "_cusum_neg", "_cusum_signals", "_ewma")
        columns = [column for column in features.columns if column.endswith(suffixes)]
        if columns:
            # float32 in the feature matrix; the detectors compare against
            # sigma-scale limits, so widen back to float64 before thresholding.
            return features[columns].astype(float)
        return compute_drift_statistics(prepared, self._cfg)

    def _anomaly_score(self, statistics: pd.DataFrame) -> float | None:
        """Score the final row for multivariate novelty, when a detector exists."""
        bundle = self._ensure_loaded()
        detector = bundle.drift_detector
        if detector is None or statistics.empty:
            return None
        missing = [column for column in detector.columns if column not in statistics.columns]
        if missing:
            logger.warning(
                "Drift detector columns absent from the computed statistics; skipping the "
                "multivariate check",
                extra={"missing": len(missing)},
            )
            return None
        scores = detector.score(statistics)
        value = float(scores.iloc[-1])
        return None if not np.isfinite(value) else value

    def _rule_violations(
        self, prepared: pd.DataFrame, rules: dict[str, SensorRule]
    ) -> tuple[list[SensorRuleViolation], float]:
        """Evaluate the sensor rules over a window.

        Args:
            prepared: The preprocessed window for one turbine.
            rules: The loaded rule set.

        Returns:
            ``(violations, data_quality)``.  Violations describe the **last**
            observation's envelope state plus how much of the window was in
            violation; ``data_quality`` is the share of the window that passed
            every validity check.
        """
        evaluation = evaluate_rules(prepared, self._cfg, rules)
        if evaluation.flags.empty:
            return [], 1.0

        n_rows = max(len(prepared), 1)
        latest = prepared.iloc[-1]
        violations: list[SensorRuleViolation] = []

        for sensor, rule in rules.items():
            if sensor not in prepared.columns:
                continue
            label = SENSOR_DISPLAY_NAMES.get(sensor, sensor.replace("_", " "))
            value = float(pd.to_numeric(pd.Series([latest.get(sensor)]), errors="coerce").iloc[0])

            severity = rule.severity_of(value)
            if severity is not DriftSeverity.NONE:
                limit = rule.alarm_limit if severity is DriftSeverity.ALARM else rule.warning_limit
                comparison = "above" if rule.direction == "high_is_bad" else "below"
                column = f"{sensor}__{'alarm' if severity is DriftSeverity.ALARM else 'warning'}"
                fraction = (
                    float(evaluation.flags[column].mean())
                    if column in evaluation.flags.columns
                    else 0.0
                )
                violations.append(
                    SensorRuleViolation(
                        sensor=sensor,
                        rule="alarm" if severity is DriftSeverity.ALARM else "warning",
                        severity=severity,
                        observed=round(value, 3),
                        limit=limit,
                        fraction_of_window=round(fraction, 4),
                        description=(
                            f"{label} at {value:.1f} {rule.unit} is {comparison} the "
                            f"{severity.value} limit of {limit:.1f} {rule.unit}, for "
                            f"{fraction:.0%} of the assessed window"
                        ),
                    )
                )

            # Validity checks are reported whenever they fired anywhere in the
            # window: a sensor that was frozen for six hours is a finding even
            # if the final sample happens to have moved.
            for check, wording in (
                ("out_of_range", "produced readings outside its physical range"),
                ("stuck", "was frozen (no movement beyond its tolerance)"),
                ("rate", "changed faster than physically plausible"),
            ):
                column = f"{sensor}__{check}"
                if column not in evaluation.flags.columns:
                    continue
                count = int(evaluation.flags[column].sum())
                if not count:
                    continue
                violations.append(
                    SensorRuleViolation(
                        sensor=sensor,
                        rule=check,
                        severity=DriftSeverity.WARNING,
                        observed=round(value, 3) if np.isfinite(value) else None,
                        limit=None,
                        fraction_of_window=round(count / n_rows, 4),
                        description=(
                            f"{label} {wording} in {count} of {n_rows} samples; the reading "
                            "may not reflect the machine"
                        ),
                    )
                )

        return violations, evaluation.data_quality()

    def _latest_z_scores(self, features: pd.DataFrame) -> dict[str, float]:
        """Extract the final row's regime-relative robust z-score per sensor."""
        suffix = "_regime_robust_z"
        if features.empty:
            return {}
        latest = features.iloc[-1]
        return {
            column.removesuffix(suffix): float(latest[column])
            for column in features.columns
            if column.endswith(suffix)
        }

    def _top_factors(
        self,
        components: list[ComponentHealth],
        z_scores: dict[str, float],
        observation: pd.Series,
        top_k: int,
    ) -> list[RiskFactor]:
        """Rank the condition deviations that drove the assessment.

        Attribution is derived from the *same* evidence as the component
        roll-up - normalised rule exceedance and bad-direction deviation from
        the turbine's own baseline - rather than from a model explainer.  This
        keeps every reported factor checkable against the raw trend, which is
        what an engineer needs in order to act on it, and avoids inventing an
        attribution the regression objective never constrained.

        Args:
            components: The component roll-up.
            z_scores: Regime-relative robust z-scores by sensor.
            observation: The final raw observation.
            top_k: Maximum number of factors to report.

        Returns:
            Ranked :class:`~wind_turbine_pm.contracts.predictions.RiskFactor`
            objects, largest deviation first.
        """
        component_of = {
            sensor: component.component for component in components for sensor in component.sensors
        }
        scored: list[tuple[float, RiskFactor]] = []

        for sensor, rule in self.rules.items():
            value = float(
                pd.to_numeric(pd.Series([observation.get(sensor)]), errors="coerce").iloc[0]
            )
            exceedance = 0.0
            if np.isfinite(value):
                exceedance = float(rule.exceedance(pd.Series([value])).iloc[0])

            z_score = z_scores.get(sensor, float("nan"))
            signed = z_score if rule.direction == "high_is_bad" else -z_score
            deviation = max(float(signed), 0.0) if np.isfinite(signed) else 0.0

            impact = max(exceedance, deviation / 4.0)
            if impact <= 0.05:
                continue

            label = SENSOR_DISPLAY_NAMES.get(sensor, sensor.replace("_", " "))
            if exceedance >= deviation / 4.0:
                feature = f"{sensor}_rule_exceedance"
                detail = (
                    f"{label} is {exceedance:.2f} of the way from its warning limit to its "
                    f"alarm limit"
                )
            else:
                feature = f"{sensor}_regime_robust_z"
                detail = (
                    f"{label} is {deviation:.1f} sigma from this turbine's own baseline for "
                    f"the current operating regime"
                )
            component = component_of.get(sensor)
            scored.append(
                (
                    impact,
                    RiskFactor(
                        feature=feature,
                        impact=round(impact, 6),
                        direction="increases_risk",
                        value=round(value, 6) if np.isfinite(value) else None,
                        description=(
                            f"{detail}"
                            + (f" ({component})" if component else "")
                            + f". Reported as {humanise_health_feature(feature)}."
                        ),
                    ),
                )
            )

        scored.sort(key=lambda item: -item[0])
        return [factor for _, factor in scored[:top_k]]

    def _assess_last_row(
        self, prepared: pd.DataFrame, features: pd.DataFrame, bands: ClassBands
    ) -> HealthAssessment:
        """Assemble the assessment for the final row of a single-turbine window.

        Args:
            prepared: Preprocessed frame for one turbine, ordered by time, with
                the operating regime attached.
            features: Feature matrix aligned to ``prepared``.
            bands: The class bands to apply.

        Returns:
            The complete :class:`~wind_turbine_pm.contracts.health.HealthAssessment`.
        """
        bundle = self._ensure_loaded()
        aligned = align_to_feature_order(features, bundle.features)
        last = features.index[-1]

        raw_score = float(predict_health_scores(bundle.estimator, aligned.loc[[last]])[0])

        statistics = self._drift_statistics(prepared, features)
        signals: list[SensorDriftSignal] = []
        if not statistics.empty:
            signals = summarise_drift(
                statistics,
                self._cfg,
                timestamps=pd.to_datetime(prepared[TIMESTAMP]),
                anomaly_score=self._anomaly_score(statistics),
                calibration=self.drift_calibration,
            )
        penalty = drift_penalty(signals, self._cfg)
        published = apply_drift_penalty(raw_score, penalty)
        health_class = classify_health(published, self._cfg, bands)

        violations, data_quality = self._rule_violations(prepared, self.rules)
        z_scores = self._latest_z_scores(features)
        observation = prepared.loc[last]
        components = build_component_health(
            observation.to_dict(), z_scores, self.rules, self._cfg, bands
        )

        regime_value = str(observation.get(OPERATING_REGIME, OperatingRegime.MEDIUM_LOAD))
        regime = OperatingRegime(regime_value)
        advisory = build_health_advisory(
            score=published,
            health_class=health_class,
            regime=regime,
            components=components,
            drift_signals=signals,
            drift_penalty=penalty,
            data_quality=data_quality,
        )

        return HealthAssessment(
            turbine_id=str(observation[TURBINE_ID]),
            timestamp=pd.Timestamp(observation[TIMESTAMP]).to_pydatetime(),
            model_version=bundle.version,
            health_score=round(published, 2),
            health_class=health_class,
            raw_health_score=round(raw_score, 2),
            # Rounded consistently with the published score so the contract's
            # "published == raw - penalty" invariant survives serialisation.
            drift_penalty=round(round(raw_score, 2) - round(published, 2), 2),
            operating_regime=regime,
            data_quality=round(float(np.clip(data_quality, 0.0, 1.0)), 4),
            component_health=components,
            drift_signals=signals,
            rule_violations=violations,
            top_factors=self._top_factors(
                components,
                z_scores,
                observation,
                int(self._cfg.get("health.serving.top_k_factors", 5)),
            ),
            explanation=advisory.explanation,
            recommendation=advisory.recommendation,
        )

    # -- Public assessment APIs -------------------------------------------
    def assess_from_window(self, window: TurbineWindow) -> HealthAssessment:
        """Assess turbine health from a window of raw observations.

        The assessment corresponds to the window's **last** timestamp.

        Args:
            window: A validated :class:`TurbineWindow`.

        Returns:
            The :class:`HealthAssessment` for the final observation.

        Raises:
            ServiceNotReadyError: If health artifacts are unavailable.
            InsufficientHistoryError: If the window is shorter than the longest
                configured feature window.
        """
        self._ensure_loaded()
        frame = window.to_frame()
        required = self.minimum_history_hours()
        span_hours = (
            (frame[TIMESTAMP].iloc[-1] - frame[TIMESTAMP].iloc[0]) / pd.Timedelta("1h")
            if len(frame) > 1
            else 0.0
        )
        if span_hours < required:
            raise InsufficientHistoryError(
                f"Turbine {window.turbine_id}: window spans {span_hours:.1f}h but at least "
                f"{required:.0f}h of history is required so the trailing health features and "
                f"the drift baselines are defined. Supply more observations."
            )

        prepared, _ = preprocess(frame, self._cfg)
        if len(prepared) < 2:
            raise InsufficientHistoryError(
                f"Turbine {window.turbine_id}: fewer than two usable observations after cleaning."
            )

        features, _ = build_health_features(prepared, self._cfg)
        # `build_health_features` attaches the regime to its own working copy;
        # re-attach it here so the assessment can report the operating point.
        if OPERATING_REGIME not in prepared.columns:
            from wind_turbine_pm.health.regimes import attach_regimes

            prepared = attach_regimes(prepared, self._cfg)

        assessment = self._assess_last_row(prepared, features, self.class_bands)
        logger.info(
            "Served health assessment",
            extra={
                "turbine_id": assessment.turbine_id,
                "health_score": assessment.health_score,
                "health_class": str(assessment.health_class),
                "drift_signals": len(assessment.drift_signals),
            },
        )
        return assessment

    def assess_from_prepared(
        self,
        frame: pd.DataFrame,
        features: pd.DataFrame,
        turbine_id: str | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> HealthAssessment:
        """Assess a turbine from an already-prepared frame and feature matrix.

        This is the path the dashboard uses.  It exists so that the detail panel
        and the fleet table cannot disagree: :meth:`score_frame` scores the
        prepared features, and re-deriving features from a short raw window would
        give a different answer for the same turbine at the same timestamp.  The
        difference is real rather than a bug - expanding baselines and drift
        statistics depend on how much history is available - which is exactly why
        both views must be fed from the same matrix.

        Args:
            frame: Prepared frame carrying keys, the operating regime and the raw
                sensor channels (as written by ``scripts/prepare_health_data.py``).
            features: Feature matrix sharing ``frame``'s index.
            turbine_id: Restrict to one turbine; required when ``frame`` holds
                more than one.
            as_of: Assess the last observation at or before this timestamp.

        Returns:
            The :class:`HealthAssessment` for the selected observation.

        Raises:
            ServiceNotReadyError: If health artifacts are unavailable.
            ValueError: If the selection is empty or spans several turbines.
        """
        self._ensure_loaded()
        working = frame
        if turbine_id is not None:
            working = working.loc[working[TURBINE_ID].astype(str) == str(turbine_id)]
        if as_of is not None:
            working = working.loc[pd.to_datetime(working[TIMESTAMP]) <= pd.Timestamp(as_of)]
        if working.empty:
            raise ValueError(
                f"No prepared observations for turbine {turbine_id!r} at or before {as_of}"
            )
        if working[TURBINE_ID].nunique() > 1:
            raise ValueError(
                "assess_from_prepared expects a single turbine; pass turbine_id to select one"
            )

        working = working.sort_values(TIMESTAMP)
        return self._assess_last_row(working, features.loc[working.index], self.class_bands)

    def assess_batch_from_windows(self, windows: list[TurbineWindow]) -> BatchHealthAssessment:
        """Assess several turbines at once.

        Args:
            windows: One window per turbine.

        Returns:
            A :class:`BatchHealthAssessment`.

        Raises:
            ServiceNotReadyError: If health artifacts are unavailable.
            ValueError: If the batch exceeds ``health.serving.max_batch_turbines``.
        """
        limit = int(self._cfg.get("health.serving.max_batch_turbines", 200))
        if len(windows) > limit:
            raise ValueError(
                f"Batch of {len(windows)} exceeds the configured limit of {limit} turbines"
            )
        assessments = [self.assess_from_window(window) for window in windows]
        logger.info("Served batch health assessment", extra={"count": len(assessments)})
        return BatchHealthAssessment(
            assessments=assessments,
            model_version=self.model_version,
            count=len(assessments),
        )

    def score_frame(
        self, frame: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Score a prepared multi-turbine frame, one row per observation.

        This is the bulk path used by the dashboard's fleet view and by the
        evaluation script.  It reports scores, classes and per-row drift
        evidence, but not the full per-turbine assessment: assembling component
        roll-ups and narratives for every row of a fleet-sized frame would be
        wasted work when only the latest row per turbine is displayed.

        Args:
            frame: Frame carrying ``turbine_id`` and ``timestamp``.
            features: Precomputed feature matrix aligned to ``frame``; computed
                from ``frame`` when omitted.

        Returns:
            A frame with ``turbine_id``, ``timestamp``, ``operating_regime``,
            ``raw_health_score``, ``health_score`` and ``health_class``.

        Raises:
            ServiceNotReadyError: If health artifacts are unavailable.
        """
        bundle = self._ensure_loaded()
        working = frame
        if features is None:
            working, _ = preprocess(frame, self._cfg)
            features, _ = build_health_features(working, self._cfg)

        aligned = align_to_feature_order(features.loc[working.index], bundle.features)
        raw = predict_health_scores(bundle.estimator, aligned)

        penalties = self._row_drift_penalties(working, features)
        published = np.clip(raw - penalties, 0.0, 100.0)
        bands = ClassBands.from_config(self._cfg)
        scores = pd.Series(published, index=working.index)

        regime = (
            working[OPERATING_REGIME].astype(str)
            if OPERATING_REGIME in working.columns
            else pd.Series(str(OperatingRegime.MEDIUM_LOAD), index=working.index)
        )
        return pd.DataFrame(
            {
                TURBINE_ID: working[TURBINE_ID].to_numpy(),
                TIMESTAMP: pd.to_datetime(working[TIMESTAMP]).to_numpy(),
                OPERATING_REGIME: regime.to_numpy(),
                "raw_health_score": raw,
                HEALTH_SCORE: published,
                HEALTH_CLASS: classify_health_series(scores, self._cfg, bands).to_numpy(),
            },
            index=working.index,
        )

    def _row_drift_penalties(self, frame: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        """Per-row drift penalty for the bulk scoring path.

        Evaluating :func:`~wind_turbine_pm.health.drift.summarise_drift` per row
        would mean one pass over the window per observation.  The same banding is
        applied vectorised instead - CUSUM graded on its trailing signal count,
        EWMA on its control-limit ratio, the worse of the two taken per sensor,
        the configured points summed and capped - which is what
        :func:`~wind_turbine_pm.health.drift.drift_penalty` does for one row.

        Args:
            frame: The preprocessed frame.
            features: The feature matrix built from it.

        Returns:
            An array of penalties aligned to ``frame``.
        """
        statistics = self._drift_statistics(frame, features)
        if statistics.empty:
            return np.zeros(len(frame), dtype=float)

        settings = DriftSettings.from_config(self._cfg)
        calibration = self.drift_calibration
        warning_points = float(self._cfg.get("health.drift.penalty.per_warning_points", 2.5))
        alarm_points = float(self._cfg.get("health.drift.penalty.per_alarm_points", 5.0))
        maximum = float(self._cfg.get("health.drift.penalty.max_points", 15.0))
        ewma_limit = settings.ewma_control_limit

        total = np.zeros(len(frame), dtype=float)
        for sensor in settings.sensors:
            signal_column = f"{sensor}_cusum_signals"
            ewma_column = f"{sensor}_ewma"
            if signal_column not in statistics.columns and ewma_column not in statistics.columns:
                continue

            # 0 = healthy, 1 = warning, 2 = alarm. Grading each detector onto the
            # same ladder is what lets the worse of the two be taken per sensor,
            # so CUSUM and EWMA firing together still count as one drifting
            # channel - matching `drift_penalty`.
            severity = np.zeros(len(frame), dtype=float)
            if signal_column in statistics.columns:
                counts = np.nan_to_num(statistics[signal_column].to_numpy(dtype=float), nan=0.0)
                if calibration is not None:
                    warning_at, alarm_at = calibration.thresholds_for(sensor, settings)
                else:
                    warning_at, alarm_at = 1.0, float(settings.alarm_signals)
                severity = np.where(
                    counts >= alarm_at, 2.0, np.where(counts >= warning_at, 1.0, 0.0)
                )
            if ewma_column in statistics.columns:
                if calibration is not None:
                    ewma_warning_at, ewma_alarm_at = calibration.ewma_thresholds_for(
                        sensor, settings
                    )
                else:
                    ewma_warning_at = ewma_limit * settings.warning_ratio
                    ewma_alarm_at = ewma_limit * settings.alarm_ratio
                magnitude = np.nan_to_num(
                    np.abs(statistics[ewma_column].to_numpy(dtype=float)), nan=0.0
                )
                severity = np.fmax(
                    severity,
                    np.where(
                        magnitude >= ewma_alarm_at,
                        2.0,
                        np.where(magnitude >= ewma_warning_at, 1.0, 0.0),
                    ),
                )

            total += np.where(
                severity >= 2.0, alarm_points, np.where(severity >= 1.0, warning_points, 0.0)
            )

        # The multivariate detector must be included here too. `summarise_drift`
        # adds it on the single-assessment path, so omitting it made the bulk path
        # report a score up to `multivariate_points` higher for the same turbine at
        # the same timestamp - the exact fleet-table-versus-detail-panel
        # disagreement `assess_from_prepared` exists to prevent.
        detector = self._ensure_loaded().drift_detector
        if (
            detector is not None
            and settings.isolation_enabled
            and all(column in statistics.columns for column in detector.columns)
        ):
            multivariate_points = float(
                self._cfg.get("health.drift.penalty.multivariate_points", 5.0)
            )
            warning_at, _ = (
                calibration.multivariate_thresholds_for(settings)
                if calibration is not None
                else (settings.isolation_threshold, 0.0)
            )
            scores = np.nan_to_num(detector.score(statistics).to_numpy(dtype=float), nan=0.0)
            total += np.where(scores >= warning_at, multivariate_points, 0.0)

        return np.minimum(total, maximum)

    def fleet_summary(
        self, scored: pd.DataFrame, as_of: pd.Timestamp | None = None
    ) -> FleetHealthSummary:
        """Roll a scored frame up to a fleet snapshot.

        Args:
            scored: Output of :meth:`score_frame`.
            as_of: Cut-off timestamp; the newest observation per turbine at or
                before it is used.  The frame's own maximum when omitted.

        Returns:
            The :class:`FleetHealthSummary`.
        """
        latest = latest_per_turbine(scored, as_of)
        if latest.empty:
            return FleetHealthSummary(
                as_of=None if as_of is None else pd.Timestamp(as_of).to_pydatetime(),
                n_turbines=0,
                mean_health_score=0.0,
                min_health_score=0.0,
                class_counts=class_distribution(pd.Series([], dtype=object)),
                regime_counts={},
                n_drift_alerts=0,
                worst_turbines=[],
                model_version=self.model_version if self.is_ready else "",
            )

        scores = pd.to_numeric(latest[HEALTH_SCORE], errors="coerce")
        # A non-zero drift deduction is the row-level evidence that at least one
        # channel drifted, which is what the bulk path records instead of the
        # full signal list.
        raw_scores = pd.to_numeric(latest["raw_health_score"], errors="coerce")
        drift_alerts = int(((raw_scores - scores) > 1e-6).sum())

        worst = latest.nsmallest(min(5, len(latest)), HEALTH_SCORE)
        return FleetHealthSummary(
            as_of=pd.Timestamp(latest[TIMESTAMP].max()).to_pydatetime(),
            n_turbines=int(latest[TURBINE_ID].nunique()),
            mean_health_score=round(float(scores.mean()), 2),
            min_health_score=round(float(scores.min()), 2),
            class_counts=class_distribution(latest[HEALTH_CLASS]),
            regime_counts={
                str(key): int(value)
                for key, value in latest[OPERATING_REGIME].value_counts().items()
            }
            if OPERATING_REGIME in latest.columns
            else {},
            n_drift_alerts=drift_alerts,
            worst_turbines=[
                {
                    "turbine_id": str(row[TURBINE_ID]),
                    "timestamp": str(row[TIMESTAMP]),
                    "health_score": round(float(row[HEALTH_SCORE]), 2),
                    "health_class": str(row[HEALTH_CLASS]),
                    "operating_regime": str(row.get(OPERATING_REGIME, "")),
                }
                for _, row in worst.iterrows()
            ],
            model_version=self.model_version,
        )


def latest_per_turbine(scored: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Reduce a scored frame to each turbine's latest row at a point in time.

    The ``as_of`` cut-off makes a fleet view answer "what did the fleet look
    like at time T?", which is how an operator reads it.  Without it the view is
    pinned to the final timestamp of the record.

    Args:
        scored: Output of :meth:`HealthMonitoringService.score_frame`.
        as_of: Cut-off timestamp; the newest observation is used when omitted.

    Returns:
        One row per turbine, worst health score first.  The original index is
        preserved so feature rows can still be looked up.
    """
    frame = scored.copy()
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])
    if as_of is not None:
        frame = frame.loc[frame[TIMESTAMP] <= pd.Timestamp(as_of)]
    if frame.empty:
        return frame
    latest = frame.sort_values(TIMESTAMP).groupby(TURBINE_ID, as_index=False).tail(1)
    return latest.sort_values(HEALTH_SCORE)


@lru_cache(maxsize=1)
def get_health_service() -> HealthMonitoringService:
    """Return the process-wide health service instance.

    Cached so the FastAPI dependency and the dashboard share one loaded model
    rather than deserialising it per request.

    Returns:
        The shared :class:`HealthMonitoringService`.
    """
    return HealthMonitoringService()
