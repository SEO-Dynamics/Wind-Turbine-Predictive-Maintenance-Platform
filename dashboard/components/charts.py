"""Plotly chart builders shared across dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wind_turbine_pm.constants import COLUMN_UNITS, TIMESTAMP

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
