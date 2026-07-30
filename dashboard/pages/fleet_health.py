"""Fleet Health dashboard page.

Rendered by ``dashboard/app.py``.  Every section guards against missing
artifacts and shows the exact command that produces them, so the page never
raises for a partially built project.

The page is organised the way an operator reads a fleet: the fleet snapshot
first (who needs attention), then one turbine in detail (why), then the sensor
evidence behind that verdict (is it real), then how well the model performs and
what it cannot do.  Assessments come from
:class:`~wind_turbine_pm.services.health_monitoring_service.HealthMonitoringService`
- the same class the API uses - so the two front ends can never disagree.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    component_health_bar,
    drift_statistic_timeline,
    health_class_distribution,
    health_score_distribution,
    health_score_timeline,
    regime_distribution,
    sensor_envelope_timeline,
)
from dashboard.components.metrics import (
    advisory_banner,
    format_metrics_table,
    health_badge,
    metric_row,
    missing_artifact_notice,
    show_figure,
)
from dashboard.data_access import (
    HEALTH_PIPELINE_COMMAND,
    figure_path,
    get_health_service_cached,
    load_health_data_quality,
    load_health_dataset,
    load_health_features,
    load_health_metrics,
    score_health,
)
from wind_turbine_pm.constants import (
    HEALTH_CLASS,
    HEALTH_SCORE,
    OPERATING_REGIME,
    TIMESTAMP,
    TURBINE_ID,
)
from wind_turbine_pm.health.regimes import regime_summary
from wind_turbine_pm.services.health_monitoring_service import latest_per_turbine


def render() -> None:
    """Render the whole Fleet Health page."""
    st.title("Turbine Health Monitoring")
    st.caption(
        "Current condition of every turbine on a 0-100 scale, banded into Healthy / Monitor / "
        "Degraded / Critical, attributed to named components and cross-checked against sensor "
        "drift."
    )
    advisory_banner()

    service = get_health_service_cached()
    if not service.is_ready:
        missing_artifact_notice(
            "The trained health-score model",
            HEALTH_PIPELINE_COMMAND,
            service.status().detail,
        )
        # `return`, not `st.stop()`: st.stop() only halts inside a Streamlit
        # script run. Under a plain `import dashboard.app` - which CI does as a
        # smoke test - it does not halt, and execution would fall through to
        # code that assumes artifacts exist.
        return

    scored = score_health("test")
    if scored is None or scored.empty:
        missing_artifact_notice("Scored fleet health data", HEALTH_PIPELINE_COMMAND)
        return

    bands = service.class_bands.to_dict()
    as_of = _as_of_selector(scored)

    _fleet_snapshot(service, scored, as_of, bands)
    st.divider()
    _fleet_table(scored, as_of)
    st.divider()
    turbine = _turbine_detail(scored, as_of, bands)
    st.divider()
    _sensor_evidence(service, turbine)
    st.divider()
    _drift_section(service, scored, turbine)
    st.divider()
    _operating_regimes(scored)
    st.divider()
    _model_performance(bands)
    st.divider()
    _limitations(service)


# ---------------------------------------------------------------------------
def _as_of_selector(scored: pd.DataFrame) -> pd.Timestamp:
    """Let the operator choose the point in time the fleet view describes."""
    times = pd.to_datetime(scored[TIMESTAMP])
    minimum, maximum = times.min().to_pydatetime(), times.max().to_pydatetime()
    chosen = st.slider(
        "Fleet snapshot as of",
        min_value=minimum,
        max_value=maximum,
        value=maximum,
        format="YYYY-MM-DD HH:mm",
        help=(
            "The snapshot and fleet table describe each turbine's most recent assessment at or "
            "before this time. Drag it back to inspect earlier fleet states."
        ),
    )
    return pd.Timestamp(chosen)


def _fleet_snapshot(
    service, scored: pd.DataFrame, as_of: pd.Timestamp, bands: dict[str, float]
) -> None:
    """Render the headline fleet numbers and the class composition."""
    st.subheader("Fleet snapshot")
    summary = service.fleet_summary(scored, as_of)
    if summary.n_turbines == 0:
        st.info("No assessments at or before the selected time.")
        return

    needs_attention = sum(
        count for name, count in summary.class_counts.items() if name != "healthy"
    )
    metric_row(
        [
            ("Turbines assessed", summary.n_turbines, None),
            (
                "Mean health score",
                f"{summary.mean_health_score:.1f}",
                "Fleet average of the published score.",
            ),
            (
                "Lowest health score",
                f"{summary.min_health_score:.1f}",
                "The worst turbine in the fleet at this time.",
            ),
            (
                "Needing attention",
                needs_attention,
                "Turbines outside the Healthy band.",
            ),
            (
                "Drift flags",
                summary.n_drift_alerts,
                "Turbines whose score carried a sensor-drift deduction.",
            ),
        ]
    )

    latest = latest_per_turbine(scored, as_of)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(health_class_distribution(latest), width="stretch")
    with right:
        st.plotly_chart(health_score_distribution(latest, bands), width="stretch")

    if summary.worst_turbines:
        st.markdown("**Lowest-scoring turbines**")
        worst = pd.DataFrame(summary.worst_turbines)
        st.dataframe(worst, width="stretch", hide_index=True)


def _fleet_table(scored: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Render the per-turbine fleet table."""
    st.subheader("Fleet detail")
    latest = latest_per_turbine(scored, as_of)
    if latest.empty:
        st.info("No assessments at or before the selected time.")
        return

    columns = [
        column
        for column in (
            TURBINE_ID,
            TIMESTAMP,
            HEALTH_SCORE,
            "raw_health_score",
            HEALTH_CLASS,
            OPERATING_REGIME,
            "health_score_target",
            "true_health_class",
        )
        if column in latest.columns
    ]
    table = latest[columns].copy()
    table["drift_deduction"] = (latest["raw_health_score"] - latest[HEALTH_SCORE]).round(2)
    st.dataframe(
        table.round(2),
        width="stretch",
        hide_index=True,
        column_config={
            HEALTH_SCORE: st.column_config.ProgressColumn(
                "Health score", min_value=0, max_value=100, format="%.1f"
            )
        },
    )
    st.caption(
        "`health_score_target` and `true_health_class` are the ground truth from the synthetic "
        "generator's degradation state and are shown here only because this is a demonstration "
        "dataset. They are not available on a real fleet."
    )


def _turbine_detail(scored: pd.DataFrame, as_of: pd.Timestamp, bands: dict[str, float]) -> str:
    """Render one turbine's health history and component roll-up.

    Returns:
        The selected turbine identifier.
    """
    st.subheader("Turbine detail")
    turbines = sorted(scored[TURBINE_ID].unique())
    # Default to the worst turbine at the selected time: that is the one an
    # operator opens the page to look at.
    ranked = latest_per_turbine(scored, as_of)
    worst = str(
        ranked[TURBINE_ID].iloc[0]
        if not ranked.empty
        else scored.sort_values(HEALTH_SCORE)[TURBINE_ID].iloc[0]
    )
    turbine = st.selectbox(
        "Turbine",
        turbines,
        index=turbines.index(worst) if worst in turbines else 0,
        help="Defaults to the worst turbine at the selected snapshot time.",
    )

    history = scored.loc[scored[TURBINE_ID] == turbine].sort_values(TIMESTAMP)
    st.plotly_chart(health_score_timeline(history, bands), width="stretch")

    assessment = _assess_at(turbine, as_of)
    if assessment is None:
        st.info(
            "A full assessment needs enough raw history for the trailing windows and the drift "
            "baselines. Generate the SCADA dataset to enable this section."
        )
        return turbine

    st.caption(
        f"Assessed from the raw observation window ending {assessment.timestamp:%Y-%m-%d %H:%M}, "
        "the same cut-off as the fleet snapshot above."
    )
    latest = history.iloc[-1]
    st.markdown(
        f"**{turbine}** - published score **{assessment.health_score:.1f}** "
        f"{health_badge(str(assessment.health_class))} &nbsp; "
        f"regime `{assessment.operating_regime.value}` &nbsp; "
        f"data quality `{assessment.data_quality:.1%}`",
        unsafe_allow_html=True,
    )
    if assessment.drift_penalty > 0:
        st.caption(
            f"Model score {assessment.raw_health_score:.1f} less a "
            f"{assessment.drift_penalty:.1f}-point sensor-drift deduction."
        )

    left, right = st.columns([3, 2])
    with left:
        components = pd.DataFrame(
            [
                {
                    "component": item.component,
                    "score": item.score,
                    "health_class": str(item.health_class),
                }
                for item in assessment.component_health
            ]
        )
        if not components.empty:
            st.plotly_chart(component_health_bar(components), width="stretch")
    with right:
        st.markdown("**Assessment**")
        st.write(assessment.explanation)
        st.markdown("**Advisory**")
        st.info(assessment.recommendation)

    if assessment.top_factors:
        st.markdown("**Largest condition deviations**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "feature": factor.feature,
                        "magnitude": round(factor.impact, 3),
                        "value": factor.value,
                        "description": factor.description,
                    }
                    for factor in assessment.top_factors
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Attribution comes from the same rule-margin and own-baseline evidence as the "
            "component scores, not from a model explainer, so every factor can be checked "
            "against the raw trend below."
        )

    if assessment.rule_violations:
        st.markdown("**Sensor rule violations**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "sensor": violation.sensor,
                        "rule": violation.rule,
                        "severity": str(violation.severity),
                        "observed": violation.observed,
                        "limit": violation.limit,
                        "share of window": f"{violation.fraction_of_window:.0%}",
                        "description": violation.description,
                    }
                    for violation in assessment.rule_violations
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    if "health_score_target" in latest.index and pd.notna(latest.get("health_score_target")):
        st.caption(
            f"Ground truth for this observation: {float(latest['health_score_target']):.1f} "
            f"(synthetic degradation state)."
        )
    return turbine


@st.cache_data(show_spinner="Assessing turbine...")
def _assess_at(turbine: str, as_of: pd.Timestamp):
    """Assess a turbine as of a point in time, from the prepared artifacts.

    Deliberately built from the *prepared* dataset and feature matrix rather than
    by re-deriving features from a raw window: the fleet table is scored from
    those same features, and a window-derived assessment would report a different
    number for the same turbine at the same timestamp, because expanding
    baselines and drift statistics depend on how much history is available.

    The service is resolved inside the function rather than passed in, so
    Streamlit only has to hash the arguments.

    Args:
        turbine: Turbine identifier.
        as_of: Cut-off timestamp; observations after it are ignored.

    Returns:
        The :class:`~wind_turbine_pm.contracts.health.HealthAssessment`, or
        ``None`` when it cannot be produced.
    """
    dataset, features = load_health_dataset(), load_health_features()
    if dataset is None or features is None:
        return None
    try:
        return get_health_service_cached().assess_from_prepared(
            dataset, features, turbine_id=turbine, as_of=as_of
        )
    except (ValueError, KeyError):
        # An empty or unusable selection must not take the page down; the caller
        # renders an explanation instead.
        return None


def _sensor_evidence(service, turbine: str) -> None:
    """Plot the raw sensor trends against the limits the rules applied."""
    st.subheader("Sensor evidence")
    st.caption(
        "The rules are the auditable part of an assessment: this is the raw channel with the "
        "exact warning and alarm limits that produced the component deductions."
    )
    # The prepared dataset, not the raw file: this is the exact cleaned series the
    # rules were evaluated against, so what the operator sees is what produced
    # the deductions.
    frame = load_health_dataset()
    if frame is None:
        missing_artifact_notice("The prepared health dataset", HEALTH_PIPELINE_COMMAND)
        return

    rules = {record["sensor"]: record for record in service.sensor_rule_records()}
    available = [sensor for sensor in rules if sensor in frame.columns]
    if not available:
        st.info("No configured sensor rules match the columns in the dataset.")
        return

    chosen = st.multiselect(
        "Channels",
        available,
        default=[s for s in ("vibration", "gearbox_temperature", "oil_pressure") if s in available]
        or available[:2],
    )
    history = frame.loc[frame[TURBINE_ID] == turbine].sort_values(TIMESTAMP).tail(1200)
    for sensor in chosen:
        st.plotly_chart(sensor_envelope_timeline(history, sensor, rules[sensor]), width="stretch")
        rule = rules[sensor]
        st.caption(
            f"Limit provenance: **{rule['source'].replace('_', ' ')}** - {rule['rationale']}"
        )

    with st.expander("All sensor validation rules"):
        st.dataframe(pd.DataFrame(service.sensor_rule_records()), width="stretch", hide_index=True)


def _drift_section(service, scored: pd.DataFrame, turbine: str) -> None:
    """Render the sensor-drift statistics and their calibrated limits."""
    st.subheader("Sensor drift")
    st.caption(
        "Drift asks a different question from condition: has this *channel* moved away from what "
        "this turbine's own history says it should read? A drifting sensor can sit inside every "
        "limit while quietly making the health score wrong, which is why it carries its own "
        "capped deduction."
    )

    calibration = service.drift_calibration
    if calibration is not None:
        st.markdown(
            f"Thresholds were calibrated on **{calibration.fitted_rows:,}** healthy training "
            f"observations so that only **{calibration.target_warning_rate:.0%}** of healthy "
            f"observations raise a warning and **{calibration.target_alarm_rate:.0%}** an alarm."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "sensor": sensor,
                        "CUSUM crossings -> warning": calibration.warning_at.get(sensor),
                        "CUSUM crossings -> alarm": calibration.alarm_at.get(sensor),
                        "EWMA magnitude -> warning": round(
                            calibration.ewma_warning_at.get(sensor, float("nan")), 3
                        ),
                    }
                    for sensor in sorted(
                        set(calibration.warning_at) | set(calibration.ewma_warning_at)
                    )
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning(
            "The published artifact carries no drift calibration, so the uncalibrated fallback "
            "thresholds are in force. Those assume statistically independent residuals, which "
            "hourly SCADA channels are not, and they raise far more alerts than they should. "
            "Retrain to fit the calibration."
        )

    features = load_health_features()
    dataset = load_health_dataset()
    if features is None or dataset is None:
        return

    sensors = sorted(
        {
            column.removesuffix("_cusum_pos")
            for column in features.columns
            if column.endswith("_cusum_pos")
        }
    )
    if not sensors:
        st.info("Drift statistics are disabled in configuration.")
        return

    sensor = st.selectbox("Drift channel", sensors)
    mask = dataset[TURBINE_ID] == turbine
    history = (
        dataset.loc[mask, [TIMESTAMP]].join(features.loc[mask]).sort_values(TIMESTAMP).tail(1200)
    )
    st.plotly_chart(drift_statistic_timeline(history, sensor), width="stretch")
    st.caption(
        "The CUSUM arms reset to zero each time they reach the decision limit (Page's restart), "
        "so the sawtooth pattern is expected: each tooth is one detection. Frequency, not height, "
        "is what severity is graded on."
    )

    flagged = scored.loc[(scored["raw_health_score"] - scored[HEALTH_SCORE]) > 1e-6]
    st.markdown(
        f"**{len(flagged):,}** of {len(scored):,} scored observations "
        f"({len(flagged) / max(len(scored), 1):.1%}) carried a drift deduction."
    )


def _operating_regimes(scored: pd.DataFrame) -> None:
    """Render the operating-regime composition."""
    st.subheader("Operating regimes")
    st.caption(
        "A reading is only interpretable relative to what the machine was doing: 60 degC in the "
        "gearbox is unremarkable at rated power and a warning sign while idling. Every "
        "regime-relative feature and drift baseline is conditioned on this label."
    )
    if OPERATING_REGIME not in scored.columns:
        st.info("The scored frame carries no operating-regime column.")
        return
    st.plotly_chart(regime_distribution(regime_summary(scored)), width="stretch")

    quality = load_health_data_quality()
    if quality and "sensor_rules" in quality:
        rules_summary = quality["sensor_rules"]
        st.caption(
            f"Sensor validity across the prepared dataset: "
            f"{float(rules_summary.get('data_quality', 1.0)):.2%} of "
            f"{int(rules_summary.get('n_rows', 0)):,} observations passed every validity check."
        )


def _model_performance(bands: dict[str, float]) -> None:
    """Render the published health-model metrics and figures."""
    st.subheader("Model performance")
    metrics = load_health_metrics()
    if metrics is None:
        missing_artifact_notice("Health model metrics", HEALTH_PIPELINE_COMMAND)
        return

    test = dict(metrics.get("metrics", {}).get("test", {}))
    formatted = format_metrics_table(
        test,
        ["mae", "mae_degraded", "rmse", "spearman", "class_agreement", "class_optimistic_rate"],
    )
    metric_row(
        [
            (
                "MAE (test)",
                formatted["mae"],
                "Mean absolute error in health-score points across all observations.",
            ),
            (
                "MAE on degraded",
                formatted["mae_degraded"],
                f"Error restricted to observations whose true score is below "
                f"{bands.get('monitor_min', 60.0):.0f}. This is the primary selection metric: "
                "overall MAE is dominated by the healthy majority.",
            ),
            ("RMSE", formatted["rmse"], None),
            (
                "Spearman",
                formatted["spearman"],
                "Rank correlation with the true condition - whether the score orders turbines "
                "correctly, which is what drives maintenance prioritisation.",
            ),
            (
                "Class agreement",
                formatted["class_agreement"],
                "Share of observations placed in the correct health band.",
            ),
            (
                "Optimistic rate",
                formatted["class_optimistic_rate"],
                "Share placed in a healthier band than the truth - the dangerous direction.",
            ),
        ]
    )

    st.caption(
        f"Selected model: {metrics.get('algorithm', 'unknown')}. "
        f"{metrics.get('selection_rationale', '')}"
    )

    if metrics.get("error_by_band"):
        st.markdown("**Error by true health band**")
        st.dataframe(pd.DataFrame(metrics["error_by_band"]), width="stretch", hide_index=True)
        st.caption(
            "Reported per band because a good overall error can be carried entirely by the "
            "healthy majority while the score is unreliable exactly where it has to be trusted."
        )

    left, right = st.columns(2)
    with left:
        show_figure(
            figure_path("health_predicted_vs_actual.png"),
            "Predicted vs actual health score (test split)",
            HEALTH_PIPELINE_COMMAND,
        )
        show_figure(
            figure_path("health_model_comparison.png"),
            "Candidate comparison (validation split)",
            HEALTH_PIPELINE_COMMAND,
        )
    with right:
        show_figure(
            figure_path("health_class_confusion.png"),
            "Health class agreement",
            HEALTH_PIPELINE_COMMAND,
        )
        show_figure(
            figure_path("health_error_by_band.png"),
            "Error distribution by true band",
            HEALTH_PIPELINE_COMMAND,
        )


def _limitations(service) -> None:
    """State plainly what this module does not do."""
    st.subheader("Limitations")
    metadata = service.metadata
    st.markdown(
        f"""
- **Synthetic ground truth.** The health target is derived from the simulator's
  `{metadata.target_source}` column. On a real fleet the label must come from a source
  *independent of the SCADA channels the features are built from* - inspection reports, oil
  analysis, borescope findings or a certified condition-monitoring index. Training against a
  label derived from the same signals would make the model learn its own input.
- **The drift detector is not validated against real calibration drift.** This dataset contains
  no injected sensor drift, so the detector has nothing genuine to find here and its thresholds
  are calibrated only to control the false-alarm rate on the healthy population. Its ability to
  catch a real drifting instrument is untested.
- **Drift is deliberately not a condition signal.** It reports that a *measurement* has moved
  away from its own baseline, which is a different question from whether the machine is
  degrading. The deduction is capped so drift alone can never manufacture a Critical.
- **Class boundaries are an operational choice, not a model property.** They are configuration
  (`health.classes`) and must be re-derived against real inspection outcomes before use.
- **Envelope limits are advisory defaults** for a 2 MW class machine and must be re-derived from
  the operator's own turbine type, control strategy and historical alarm log.
- **Component scores are rule- and baseline-driven, not model-derived.** That makes them
  auditable but means they will not always agree with the overall score, which is fleet-trained.
  A disagreement is information, not a bug.
- **Advisory only.** Nothing here is a certified safety system or a substitute for inspection.
"""
    )
