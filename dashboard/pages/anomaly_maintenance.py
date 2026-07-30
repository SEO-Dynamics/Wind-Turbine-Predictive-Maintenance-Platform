"""Anomaly and maintenance dashboard backed by the production services."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.metrics import advisory_banner, metric_row, missing_artifact_notice
from dashboard.data_access import (
    ANOMALY_PIPELINE_COMMAND,
    assess_maintenance_fleet,
    get_anomaly_service_cached,
    get_maintenance_service_cached,
    load_anomaly_metrics,
    score_anomaly,
)


def render() -> None:
    """Render fleet priority, turbine evidence and model calibration."""
    st.title("Anomaly & Maintenance")
    st.caption(
        "Healthy-reference novelty, three independent model assessments, unified risk "
        "coverage and deterministic maintenance actions."
    )
    advisory_banner()

    anomaly_service = get_anomaly_service_cached()
    maintenance_service = get_maintenance_service_cached()
    if not anomaly_service.is_ready:
        missing_artifact_notice(
            "The calibrated anomaly model",
            ANOMALY_PIPELINE_COMMAND,
            anomaly_service.status().detail,
        )
        return

    scored = score_anomaly("test")
    assessments = assess_maintenance_fleet()
    if scored is None or scored.empty:
        missing_artifact_notice("Scored anomaly data", ANOMALY_PIPELINE_COMMAND)
        return

    latest = (
        scored.sort_values("timestamp")
        .groupby("turbine_id", as_index=False)
        .tail(1)
        .sort_values("anomaly_score", ascending=False)
    )
    status = maintenance_service.status()
    metric_row(
        [
            ("Turbines", latest["turbine_id"].nunique(), None),
            ("Warning / alarm", int((latest["anomaly_severity"] != "normal").sum()), None),
            ("Decision coverage", f"{float(status['coverage']):.0%}", None),
            ("Missing modules", ", ".join(status["missing_modules"]) or "none", None),
        ]
    )

    st.subheader("Fleet priority queue")
    if assessments:
        queue = pd.DataFrame(
            [
                {
                    "turbine": item["turbine_id"],
                    "action": item["recommendation"]["action"],
                    "within_hours": item["recommendation"]["recommended_within_hours"],
                    "risk": item["risk_level"],
                    "unified_score": item["unified_risk_score"],
                    "coverage": item["coverage"],
                    "confidence": item["decision_confidence"],
                    "missing_modules": ", ".join(item["missing_modules"]),
                }
                for item in assessments
            ]
        )
        st.dataframe(queue, hide_index=True, width="stretch")
        turbine_ids = queue["turbine"].tolist()
    else:
        st.warning(
            "No unified queue could be built. The anomaly-only ranking remains available; "
            "run `python scripts/run_all_pipelines.py` to restore all evidence."
        )
        st.dataframe(
            latest[["turbine_id", "anomaly_score", "anomaly_severity", "operational_status"]],
            hide_index=True,
            width="stretch",
        )
        turbine_ids = latest["turbine_id"].tolist()

    st.divider()
    st.subheader("Turbine detail")
    turbine_id = st.selectbox("Turbine", turbine_ids)
    turbine_scores = scored.loc[scored["turbine_id"] == turbine_id].copy()
    turbine_scores["timestamp"] = pd.to_datetime(turbine_scores["timestamp"])
    st.line_chart(
        turbine_scores.set_index("timestamp")[["anomaly_score", "degradation_level"]],
        height=280,
    )

    selected = next(
        (item for item in assessments or [] if item["turbine_id"] == turbine_id),
        None,
    )
    if selected:
        recommendation = selected["recommendation"]
        metric_row(
            [
                ("Unified risk", f"{selected['unified_risk_score']:.1%}", None),
                ("Risk level", selected["risk_level"], None),
                ("Action", recommendation["action"].replace("_", " "), None),
                (
                    "Recommended window",
                    (
                        f"{recommendation['recommended_within_hours']} h"
                        if recommendation["recommended_within_hours"]
                        else "routine cadence"
                    ),
                    None,
                ),
            ]
        )
        evidence = pd.DataFrame(
            [
                {
                    "module": module,
                    "risk": risk,
                    "available": module not in selected["missing_modules"],
                }
                for module, risk in {
                    "failure": selected["component_scores"].get("failure"),
                    "anomaly": selected["component_scores"].get("anomaly"),
                    "health": selected["component_scores"].get("health"),
                }.items()
            ]
        )
        st.markdown("**Independent model evidence**")
        st.dataframe(evidence, hide_index=True, width="stretch")
        st.markdown("**Reasons and target components**")
        for reason in recommendation["reasons"] or ["No guardrail was triggered."]:
            st.write(f"- {reason}")
        st.caption(
            "Targets: "
            + (", ".join(recommendation["target_components"]) or "no specific component")
        )

    st.divider()
    _model_calibration()


def _model_calibration() -> None:
    """Show comparison and the measured healthy-population alert contract."""
    st.subheader("Model comparison & threshold calibration")
    document = load_anomaly_metrics()
    if document is None:
        st.info("Calibration metrics are unavailable.")
        return
    comparison = pd.DataFrame(document.get("candidates", []))
    if not comparison.empty:
        st.dataframe(comparison, hide_index=True, width="stretch")
    calibration = document["calibration"]
    rates = pd.DataFrame(
        {
            "threshold": ["warning", "alarm"],
            "target healthy alert rate": [0.05, 0.01],
            "measured healthy alert rate": [
                calibration["achieved_warning_rate"],
                calibration["achieved_alarm_rate"],
            ],
        }
    ).set_index("threshold")
    st.bar_chart(rates, height=260)
    st.caption(
        "Scores are empirical healthy-validation percentiles. Signal deviations are "
        "associations against median/IQR references, not causal diagnoses."
    )
