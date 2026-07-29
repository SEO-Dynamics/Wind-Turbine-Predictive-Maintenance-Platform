"""Failure Prediction dashboard page.

Rendered by ``dashboard/app.py``.  Every section guards against missing
artifacts and shows the exact command that produces them, so the page never
raises for a partially built project.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    global_importance_bar,
    power_curve_scatter,
    probability_timeline,
    risk_distribution,
    risk_factor_bar,
    sensor_timeline,
)
from dashboard.components.metrics import (
    advisory_banner,
    format_metrics_table,
    metric_row,
    missing_artifact_notice,
    risk_badge,
    show_figure,
)
from dashboard.data_access import (
    EVALUATE_COMMAND,
    PIPELINE_COMMAND,
    figure_path,
    get_config_cached,
    get_service_cached,
    latest_per_turbine,
    load_features,
    load_global_importance,
    load_metrics,
    load_raw_sample,
    score_split,
)
from wind_turbine_pm.constants import SENSOR_COLUMNS, TIMESTAMP, TURBINE_ID

DEFAULT_SENSORS = ["vibration", "gearbox_temperature", "oil_pressure"]


def render() -> None:
    """Render the whole Failure Prediction page."""
    st.title("Failure Prediction")
    st.caption(
        "Estimated probability that a turbine experiences a failure within the next 48 hours, "
        "with the drivers behind each estimate."
    )
    advisory_banner()

    service = get_service_cached()
    if not service.is_ready:
        missing_artifact_notice(
            "The trained failure-prediction model",
            PIPELINE_COMMAND,
            service.status().detail,
        )
        # `return`, not `st.stop()`: st.stop() only halts inside a Streamlit
        # script run. Under a plain `import dashboard.app` - which CI does as a
        # smoke test - it does not halt, and execution would fall through to
        # code that assumes artifacts exist.
        return

    scored = score_split("test")
    if scored is None or scored.empty:
        missing_artifact_notice("Scored fleet data", PIPELINE_COMMAND)
        return

    as_of = _as_of_selector(scored)
    _executive_summary(service, scored, as_of)
    st.divider()
    _fleet_table(scored, as_of)
    st.divider()
    _turbine_detail(service, scored)
    st.divider()
    _model_performance()
    st.divider()
    _explainability()
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
            "The executive summary and fleet table describe each turbine's most recent "
            "observation at or before this time. Drag it back to inspect earlier fleet states."
        ),
    )
    return pd.Timestamp(chosen)


def _executive_summary(service, scored: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Render the headline fleet numbers."""
    st.subheader("Executive summary")
    latest = latest_per_turbine(scored, as_of)
    if latest.empty:
        st.info("No observations at or before the selected time.")
        return
    n_high = int((latest["risk_level"] == "high").sum())
    top = latest.iloc[0]

    metric_row(
        [
            (
                "Turbines evaluated",
                f"{latest[TURBINE_ID].nunique():,}",
                "Distinct turbines in the scored split.",
            ),
            (
                "High-risk turbines",
                f"{n_high}",
                "Turbines whose latest probability is in the high band.",
            ),
            (
                "Mean failure probability",
                f"{latest['failure_probability'].mean():.1%}",
                "Average across turbines' most recent observation.",
            ),
            (
                "Highest-risk turbine",
                f"{top[TURBINE_ID]} ({top['failure_probability']:.1%})",
                "Turbine with the highest latest probability.",
            ),
            ("Model version", service.model_version, f"Algorithm: {service.metadata.algorithm}"),
        ]
    )
    st.plotly_chart(risk_distribution(latest), use_container_width=True)


def _fleet_table(scored: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Render the sortable, filterable fleet risk table."""
    st.subheader("Fleet failure risk")
    latest = latest_per_turbine(scored, as_of)
    if latest.empty:
        st.info("No observations at or before the selected time.")
        return

    left, right = st.columns([1, 2])
    levels = left.multiselect(
        "Filter by risk level", ["high", "medium", "low"], default=["high", "medium", "low"]
    )
    minimum = right.slider("Minimum failure probability", 0.0, 1.0, 0.0, 0.01)

    filtered = latest.loc[
        latest["risk_level"].isin(levels) & (latest["failure_probability"] >= minimum)
    ]
    if filtered.empty:
        st.info("No turbines match the current filters.")
        return

    top_factor = _top_factor_for_rows(filtered)
    table = pd.DataFrame(
        {
            "Turbine": filtered[TURBINE_ID].to_numpy(),
            "Last observation": pd.to_datetime(filtered[TIMESTAMP]).dt.strftime("%Y-%m-%d %H:%M"),
            "Failure probability": filtered["failure_probability"].to_numpy(),
            "Prediction": filtered["prediction"].to_numpy(),
            "Risk level": filtered["risk_level"].to_numpy(),
            "Top risk factor": top_factor,
        }
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Failure probability": st.column_config.ProgressColumn(
                "Failure probability", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    st.caption(
        "Click a column header to sort. 'Prediction' applies the optimised decision threshold."
    )


def _top_factor_for_rows(rows: pd.DataFrame) -> list[str]:
    """Compute the single most important risk factor for each row."""
    features = load_features()
    service = get_service_cached()
    if features is None:
        return ["n/a"] * len(rows)

    labels: list[str] = []
    for index in rows.index:
        try:
            attributions = service.explain_observation(features.loc[[index]], top_k=1)
        except (ValueError, KeyError):
            attributions = []
        labels.append(attributions[0].feature if attributions else "n/a")
    return labels


def _turbine_detail(service, scored: pd.DataFrame) -> None:
    """Render the per-turbine drill-down."""
    st.subheader("Turbine detail")
    turbines = sorted(scored[TURBINE_ID].unique())
    left, middle, right = st.columns([1, 1, 2])
    turbine = left.selectbox("Turbine", turbines)

    turbine_rows = scored.loc[scored[TURBINE_ID] == turbine].sort_values(TIMESTAMP)
    times = pd.to_datetime(turbine_rows[TIMESTAMP])
    date_range = middle.date_input(
        "Time range",
        value=(times.min().date(), times.max().date()),
        min_value=times.min().date(),
        max_value=times.max().date(),
    )
    sensors = right.multiselect("Sensor variables", list(SENSOR_COLUMNS), default=DEFAULT_SENSORS)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        turbine_rows = turbine_rows.loc[(times >= start) & (times < end)]

    if turbine_rows.empty:
        st.info("No observations in the selected range.")
        return

    low_max, medium_max = service.risk_bands
    st.plotly_chart(
        probability_timeline(turbine_rows, service.threshold, low_max, medium_max),
        use_container_width=True,
    )

    raw = load_raw_sample()
    if raw is not None and sensors:
        raw_rows = raw.loc[raw[TURBINE_ID] == turbine].copy()
        raw_rows[TIMESTAMP] = pd.to_datetime(raw_rows[TIMESTAMP])
        window = raw_rows.loc[
            (raw_rows[TIMESTAMP] >= turbine_rows[TIMESTAMP].min())
            & (raw_rows[TIMESTAMP] <= turbine_rows[TIMESTAMP].max())
        ]
        st.plotly_chart(
            sensor_timeline(window, sensors, f"{turbine} sensor trends"), use_container_width=True
        )

    latest_row = turbine_rows.iloc[-1]
    st.markdown(
        f"**Latest assessment for {turbine}** &nbsp; {risk_badge(str(latest_row['risk_level']))}",
        unsafe_allow_html=True,
    )
    metric_row(
        [
            ("Failure probability", f"{latest_row['failure_probability']:.1%}", None),
            ("Prediction", "Failure risk flagged" if latest_row["prediction"] else "No flag", None),
            (
                "Decision threshold",
                f"{service.threshold:.3f}",
                "Optimised on the validation split.",
            ),
            ("Horizon", f"{service.horizon_hours} h", None),
        ]
    )

    _local_explanation(service, latest_row, turbine)


def _local_explanation(service, latest_row: pd.Series, turbine: str) -> None:
    """Render the local explanation and advisory message for one observation."""
    features = load_features()
    if features is None:
        missing_artifact_notice("The feature matrix", PIPELINE_COMMAND)
        return

    dataset_index = latest_row.name
    try:
        row_features = features.loc[[dataset_index]]
    except KeyError:
        st.caption("_Feature row unavailable for this observation._")
        return

    prediction = service.predict_from_features(
        row_features.iloc[0].to_dict(),
        turbine_id=turbine,
        timestamp=pd.Timestamp(latest_row[TIMESTAMP]).to_pydatetime(),
    )

    if prediction.top_risk_factors:
        factors = pd.DataFrame(
            [
                {"feature": factor.feature, "impact": factor.impact, "direction": factor.direction}
                for factor in prediction.top_risk_factors
            ]
        )
        st.plotly_chart(risk_factor_bar(factors), use_container_width=True)
    else:
        st.caption("_No feature-level attribution was available for this observation._")

    st.markdown("**Explanation**")
    st.write(prediction.explanation)
    st.markdown("**Advisory recommendation**")
    st.success(prediction.recommendation)


def _model_performance() -> None:
    """Render evaluation metrics and figures."""
    st.subheader("Model performance")
    metrics = load_metrics()
    if metrics is None:
        missing_artifact_notice("Evaluation metrics", EVALUATE_COMMAND)
        return

    evaluation = metrics.get("evaluation") or metrics.get("metrics") or {}
    if not evaluation:
        missing_artifact_notice("Evaluation metrics", EVALUATE_COMMAND)
        return

    keys = [
        "precision",
        "recall",
        "f1",
        "f2",
        "pr_auc",
        "roc_auc",
        "false_negative_rate",
        "false_positive_rate",
    ]
    rows = {
        split: format_metrics_table(values, keys)
        for split, values in evaluation.items()
        if isinstance(values, dict)
    }
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
    st.caption(
        "Accuracy is deliberately omitted as a headline metric: with a ~2% positive rate, "
        "predicting 'no failure' everywhere already scores ~98%."
    )

    left, middle, right = st.columns(3)
    with left:
        show_figure(
            figure_path("confusion_matrix_test.png"), "Test confusion matrix", EVALUATE_COMMAND
        )
    with middle:
        show_figure(
            figure_path("pr_curve_test.png"), "Test precision-recall curve", EVALUATE_COMMAND
        )
    with right:
        show_figure(figure_path("roc_curve_test.png"), "Test ROC curve", EVALUATE_COMMAND)

    with st.expander("Threshold optimisation"):
        threshold_info = metrics.get("threshold", {})
        if threshold_info:
            st.write(
                f"Selected threshold **{threshold_info.get('threshold', float('nan')):.4f}** "
                f"using the **{threshold_info.get('method', 'n/a')}** method on the "
                f"**{threshold_info.get('selected_on', 'valid')}** split."
            )
            st.json(
                {
                    "costs": threshold_info.get("costs", {}),
                    "alternatives": threshold_info.get("alternatives", {}),
                }
            )
            st.caption(
                "Cost units are hypothetical and not calibrated against real operator economics."
            )
        show_figure(figure_path("threshold_curve.png"), "Threshold performance", EVALUATE_COMMAND)

    with st.expander("Candidate model comparison"):
        show_figure(figure_path("model_comparison.png"), "Model comparison", PIPELINE_COMMAND)
        candidates = metrics.get("candidates", {})
        if candidates:
            table = pd.DataFrame(
                {
                    name: {
                        "algorithm": info.get("algorithm", ""),
                        **{
                            k: round(float(v), 4)
                            for k, v in info.get("valid", {}).items()
                            if k in keys
                        },
                        "rejected_reason": info.get("rejected_reason", ""),
                    }
                    for name, info in candidates.items()
                }
            ).T
            st.dataframe(table, use_container_width=True)
        st.info(metrics.get("selection_rationale", ""))


def _explainability() -> None:
    """Render global explainability artifacts."""
    st.subheader("Explainability")
    importance = load_global_importance()
    if importance is not None:
        st.plotly_chart(global_importance_bar(importance), use_container_width=True)
    else:
        missing_artifact_notice("Global feature importance", EVALUATE_COMMAND)

    left, right = st.columns(2)
    with left:
        show_figure(figure_path("shap_summary.png"), "SHAP summary (test split)", EVALUATE_COMMAND)
    with right:
        show_figure(figure_path("shap_bar.png"), "Mean |SHAP value|", EVALUATE_COMMAND)

    metrics = load_metrics() or {}
    example = (metrics.get("explainability") or {}).get("example_high_risk")
    if example:
        with st.expander("Worked local explanation (highest-risk test observation)", expanded=True):
            st.write(
                f"**{example['turbine_id']}** at {example['timestamp']} - "
                f"probability {example['failure_probability']:.1%}, "
                f"actual label {example['actual_label']}"
            )
            show_figure(
                figure_path("local_explanation_high_risk.png"),
                "Local explanation",
                EVALUATE_COMMAND,
            )
            st.write(example["explanation"])
            st.success(example["recommendation"])

    raw = load_raw_sample()
    if raw is not None:
        with st.expander("Power curve"):
            sample = raw.dropna(subset=["wind_speed", "power_output"]).sample(
                min(8000, len(raw)), random_state=0
            )
            st.plotly_chart(power_curve_scatter(sample), use_container_width=True)


def _limitations(service) -> None:
    """Render the standing limitations section."""
    st.subheader("Limitations")
    cfg = get_config_cached()
    is_synthetic = service.metadata.dataset.is_synthetic
    st.markdown(
        f"""
- **Data source:** {
            "**synthetic** - generated by this project, not measured from real turbines."
            if is_synthetic
            else f"`{service.metadata.dataset.source}`"
        }
  All metrics on this page describe performance on that data only.
- **Advisory only.** Output supports human decisions. It must not stop, start, derate or
  schedule maintenance on plant automatically.
- **Not validated in the real world.** The model has never been evaluated against measured
  failures on operating turbines. Real-world validation is required before any operational use.
- **Not a certified safety system.** It does not replace protection systems, condition-monitoring
  hardware, or the judgement of a qualified maintenance engineer.
- **Cost figures are hypothetical.** The threshold was optimised against an illustrative cost
  matrix (FN={cfg.get("threshold.costs.false_negative")}, FP={
            cfg.get("threshold.costs.false_positive")
        }),
  not against any operator's real economics.
- **Distribution shift.** Performance will degrade as turbines age, sensors drift or the fleet
  changes. Sensor-drift monitoring is the responsibility of a future module.
"""
    )
