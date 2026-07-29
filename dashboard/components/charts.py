"""Plotly chart builders shared across dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wind_turbine_pm.constants import (
    COLUMN_UNITS,
    HEALTH_CLASS,
    HEALTH_SCORE,
    OPERATING_REGIME,
    TIMESTAMP,
)

RISK_COLOURS: dict[str, str] = {"low": "#2f855a", "medium": "#b7791f", "high": "#c53030"}


def sensor_timeline(
    frame: pd.DataFrame, sensors: list[str], title: str = "Sensor trends"
) -> go.Figure:
    """Plot one or more sensor channels over time.

    Args:
        frame: Frame with a ``timestamp`` column and the requested sensors.
        sensors: Sensor column names to plot.
        title: Figure title.

    Returns:
        A Plotly figure with one subplot row per sensor.
    """
    available = [sensor for sensor in sensors if sensor in frame.columns]
    if not available:
        return go.Figure().update_layout(title="No sensors selected")

    figure = go.Figure()
    for sensor in available:
        unit = COLUMN_UNITS.get(sensor, "")
        figure.add_trace(
            go.Scatter(
                x=frame[TIMESTAMP],
                y=frame[sensor],
                name=f"{sensor} ({unit})" if unit else sensor,
                mode="lines",
                line={"width": 1.4},
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        height=420,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def probability_timeline(
    frame: pd.DataFrame, threshold: float, low_max: float, medium_max: float
) -> go.Figure:
    """Plot failure probability over time with the threshold and risk bands.

    Args:
        frame: Frame with ``timestamp`` and ``failure_probability``.
        threshold: The decision threshold to mark.
        low_max: Upper bound of the low-risk band.
        medium_max: Upper bound of the medium-risk band.

    Returns:
        The Plotly figure.
    """
    figure = go.Figure()
    figure.add_hrect(y0=0, y1=low_max, fillcolor=RISK_COLOURS["low"], opacity=0.08, line_width=0)
    figure.add_hrect(
        y0=low_max, y1=medium_max, fillcolor=RISK_COLOURS["medium"], opacity=0.10, line_width=0
    )
    figure.add_hrect(
        y0=medium_max, y1=1.0, fillcolor=RISK_COLOURS["high"], opacity=0.10, line_width=0
    )

    figure.add_trace(
        go.Scatter(
            x=frame[TIMESTAMP],
            y=frame["failure_probability"],
            name="Failure probability",
            mode="lines",
            line={"width": 1.8, "color": "#2b6cb0"},
        )
    )
    if "failure_within_48h" in frame.columns:
        positives = frame.loc[frame["failure_within_48h"] == 1]
        if not positives.empty:
            figure.add_trace(
                go.Scatter(
                    x=positives[TIMESTAMP],
                    y=positives["failure_probability"],
                    name="Actual: failure within 48h",
                    mode="markers",
                    marker={"size": 5, "color": "#c53030", "symbol": "x"},
                )
            )
    figure.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"threshold {threshold:.3f}",
        annotation_position="top left",
    )
    figure.update_layout(
        title="Predicted failure probability",
        xaxis_title="Time",
        yaxis_title="P(failure within 48h)",
        yaxis_range=[0, 1],
        height=380,
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def risk_distribution(frame: pd.DataFrame) -> go.Figure:
    """Plot the fleet's risk-level composition.

    Args:
        frame: Frame with a ``risk_level`` column.

    Returns:
        The Plotly figure.
    """
    counts = frame["risk_level"].value_counts().reindex(["low", "medium", "high"]).fillna(0)
    figure = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.to_numpy(),
            marker_color=[RISK_COLOURS[level] for level in counts.index],
            text=counts.to_numpy().astype(int),
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Fleet risk distribution",
        xaxis_title="Risk level",
        yaxis_title="Turbines",
        height=320,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def risk_factor_bar(factors: pd.DataFrame) -> go.Figure:
    """Plot signed local risk-factor contributions.

    Args:
        factors: Frame with ``feature`` and ``impact`` columns.

    Returns:
        The Plotly figure.
    """
    ordered = factors.sort_values("impact")
    colours = [
        RISK_COLOURS["high"] if value >= 0 else RISK_COLOURS["low"] for value in ordered["impact"]
    ]
    figure = go.Figure(
        go.Bar(x=ordered["impact"], y=ordered["feature"], orientation="h", marker_color=colours)
    )
    figure.update_layout(
        title="Top contributing risk factors",
        xaxis_title="Contribution to predicted risk (log-odds)",
        height=max(280, 42 * len(ordered)),
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def global_importance_bar(importance: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Plot global feature importance.

    Args:
        importance: Frame with ``feature`` and ``importance`` columns.
        top_n: Number of features to display.

    Returns:
        The Plotly figure.
    """
    subset = importance.head(top_n).iloc[::-1]
    figure = px.bar(subset, x="importance", y="feature", orientation="h")
    figure.update_traces(marker_color="#2b6cb0")
    figure.update_layout(
        title=f"Top {top_n} features by global importance",
        xaxis_title="Mean |contribution|",
        yaxis_title="",
        height=max(360, 24 * len(subset)),
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


# ---------------------------------------------------------------------------
# Turbine Health Monitoring
# ---------------------------------------------------------------------------
#: Colour per health class, matching the training figures.
HEALTH_COLOURS: dict[str, str] = {
    "healthy": "#2f855a",
    "monitor": "#b7791f",
    "degraded": "#c05621",
    "critical": "#c53030",
}


def health_class_distribution(frame: pd.DataFrame) -> go.Figure:
    """Plot the fleet's health-class composition.

    Args:
        frame: Frame with a ``health_class`` column.

    Returns:
        The Plotly figure.
    """
    order = list(HEALTH_COLOURS)
    counts = frame[HEALTH_CLASS].value_counts().reindex(order).fillna(0)
    figure = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.to_numpy(),
            marker_color=[HEALTH_COLOURS[name] for name in counts.index],
            text=counts.to_numpy().astype(int),
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Fleet health classification",
        xaxis_title="Health class",
        yaxis_title="Turbines",
        height=320,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def health_score_timeline(frame: pd.DataFrame, bands: dict[str, float]) -> go.Figure:
    """Plot a turbine's health score over time with the class bands shaded.

    Args:
        frame: Frame with ``timestamp`` and ``health_score``; ``raw_health_score``
            and ``health_score_target`` are drawn when present.
        bands: Mapping with ``healthy_min``, ``monitor_min`` and ``degraded_min``.

    Returns:
        The Plotly figure.
    """
    healthy_min = float(bands.get("healthy_min", 80.0))
    monitor_min = float(bands.get("monitor_min", 60.0))
    degraded_min = float(bands.get("degraded_min", 40.0))

    figure = go.Figure()
    for lower, upper, name in (
        (healthy_min, 100.0, "healthy"),
        (monitor_min, healthy_min, "monitor"),
        (degraded_min, monitor_min, "degraded"),
        (0.0, degraded_min, "critical"),
    ):
        figure.add_hrect(
            y0=lower, y1=upper, fillcolor=HEALTH_COLOURS[name], opacity=0.10, line_width=0
        )

    if "health_score_target" in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame[TIMESTAMP],
                y=frame["health_score_target"],
                name="Actual condition",
                mode="lines",
                line={"width": 1.4, "color": "#718096", "dash": "dot"},
            )
        )
    if "raw_health_score" in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame[TIMESTAMP],
                y=frame["raw_health_score"],
                name="Model score (before drift penalty)",
                mode="lines",
                line={"width": 1.0, "color": "#a0aec0"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=frame[TIMESTAMP],
            y=frame[HEALTH_SCORE],
            name="Published health score",
            mode="lines",
            line={"width": 2.0, "color": "#2b6cb0"},
        )
    )
    figure.update_layout(
        title="Health score over time",
        xaxis_title="Time",
        yaxis_title="Health score (0-100)",
        yaxis_range=[0, 100],
        height=400,
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def component_health_bar(components: pd.DataFrame) -> go.Figure:
    """Plot per-component health scores, worst first.

    Args:
        components: Frame with ``component`` and ``score`` columns.

    Returns:
        The Plotly figure.
    """
    ordered = components.sort_values("score", ascending=False)
    colours = [
        HEALTH_COLOURS.get(str(health_class), "#4a5568")
        for health_class in ordered.get("health_class", ["healthy"] * len(ordered))
    ]
    figure = go.Figure(
        go.Bar(
            x=ordered["score"],
            y=ordered["component"],
            orientation="h",
            marker_color=colours,
            text=ordered["score"].round(1),
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Component condition (rule and baseline evidence)",
        xaxis_title="Component score (0-100, higher is healthier)",
        xaxis_range=[0, 105],
        height=max(280, 46 * len(ordered)),
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def regime_distribution(summary: pd.DataFrame) -> go.Figure:
    """Plot how much time the fleet spends in each operating regime.

    Args:
        summary: Output of
            :func:`~wind_turbine_pm.health.regimes.regime_summary`.

    Returns:
        The Plotly figure.
    """
    populated = summary.loc[summary["observations"] > 0]
    figure = go.Figure(
        go.Bar(
            x=populated[OPERATING_REGIME],
            y=populated["observations"],
            marker_color="#4c51bf",
            text=(populated["share"] * 100).round(1).astype(str) + "%",
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Time spent per operating regime",
        xaxis_title="Operating regime",
        yaxis_title="Observations",
        height=320,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def health_score_distribution(frame: pd.DataFrame, bands: dict[str, float]) -> go.Figure:
    """Plot the distribution of health scores across the fleet.

    Args:
        frame: Frame with a ``health_score`` column.
        bands: Mapping with the class boundaries.

    Returns:
        The Plotly figure.
    """
    figure = go.Figure(
        go.Histogram(x=frame[HEALTH_SCORE], nbinsx=40, marker_color="#2b6cb0", opacity=0.85)
    )
    for key, name in (
        ("healthy_min", "healthy"),
        ("monitor_min", "monitor"),
        ("degraded_min", "degraded"),
    ):
        figure.add_vline(
            x=float(bands.get(key, 0.0)),
            line_dash="dash",
            line_color=HEALTH_COLOURS[name],
            annotation_text=name,
            annotation_position="top",
        )
    figure.update_layout(
        title="Health score distribution",
        xaxis_title="Health score",
        yaxis_title="Observations",
        height=330,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def drift_statistic_timeline(frame: pd.DataFrame, sensor: str) -> go.Figure:
    """Plot a sensor's CUSUM arms and EWMA against their limits.

    Args:
        frame: Frame with ``timestamp`` and the sensor's drift statistic columns.
        sensor: Sensor channel name.

    Returns:
        The Plotly figure.
    """
    figure = go.Figure()
    for suffix, label, colour in (
        ("_cusum_pos", "CUSUM upward", "#c53030"),
        ("_cusum_neg", "CUSUM downward", "#2b6cb0"),
        ("_ewma", "EWMA residual", "#b7791f"),
    ):
        column = f"{sensor}{suffix}"
        if column in frame.columns:
            figure.add_trace(
                go.Scatter(
                    x=frame[TIMESTAMP],
                    y=frame[column],
                    name=label,
                    mode="lines",
                    line={"width": 1.3, "color": colour},
                )
            )
    figure.update_layout(
        title=f"Drift statistics - {sensor.replace('_', ' ')}",
        xaxis_title="Time",
        yaxis_title="Statistic (sigma units)",
        height=340,
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.25},
    )
    return figure


def sensor_envelope_timeline(
    frame: pd.DataFrame, sensor: str, rule: dict[str, float | str | None]
) -> go.Figure:
    """Plot a sensor against its warning and alarm limits.

    This is the chart that makes a component score checkable: the operator sees
    the raw trend and the exact limit the deduction was based on.

    Args:
        frame: Frame with ``timestamp`` and the sensor column.
        sensor: Sensor channel name.
        rule: The sensor's rule record, as returned by the API.

    Returns:
        The Plotly figure.
    """
    unit = str(rule.get("unit") or COLUMN_UNITS.get(sensor, ""))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame[TIMESTAMP],
            y=frame[sensor],
            name=sensor.replace("_", " "),
            mode="lines",
            line={"width": 1.4, "color": "#2b6cb0"},
        )
    )
    for key, colour, label in (
        ("warn_above", "#b7791f", "warning"),
        ("alarm_above", "#c53030", "alarm"),
        ("warn_below", "#b7791f", "warning"),
        ("alarm_below", "#c53030", "alarm"),
    ):
        limit = rule.get(key)
        if limit is None:
            continue
        figure.add_hline(
            y=float(limit),
            line_dash="dash",
            line_color=colour,
            annotation_text=f"{label} {float(limit):g}",
            annotation_position="top left",
        )
    figure.update_layout(
        title=f"{sensor.replace('_', ' ')} against its operating envelope",
        xaxis_title="Time",
        yaxis_title=f"{sensor.replace('_', ' ')} ({unit})" if unit else sensor,
        height=360,
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def power_curve_scatter(frame: pd.DataFrame) -> go.Figure:
    """Plot the observed power curve, coloured by risk when available.

    Args:
        frame: Frame with ``wind_speed`` and ``power_output``.

    Returns:
        The Plotly figure.
    """
    colour = "risk_level" if "risk_level" in frame.columns else None
    figure = px.scatter(
        frame,
        x="wind_speed",
        y="power_output",
        color=colour,
        color_discrete_map=RISK_COLOURS,
        opacity=0.45,
        render_mode="webgl",
    )
    figure.update_traces(marker={"size": 4})
    figure.update_layout(
        title="Power curve",
        xaxis_title="Wind speed (m/s)",
        yaxis_title="Power output (kW)",
        height=380,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure
