"""Reusable Streamlit metric and status components.

Shared by every dashboard page.  Future modules should reuse
:func:`missing_artifact_notice` so that a half-built project degrades into
actionable instructions rather than a traceback.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from wind_turbine_pm.constants import ADVISORY_DISCLAIMER

#: Colour per risk level, used consistently across pages.
RISK_COLOURS: dict[str, str] = {"low": "#2f855a", "medium": "#b7791f", "high": "#c53030"}

#: Colour per health class, used consistently across pages and figures.
HEALTH_COLOURS: dict[str, str] = {
    "healthy": "#2f855a",
    "monitor": "#b7791f",
    "degraded": "#c05621",
    "critical": "#c53030",
}


def missing_artifact_notice(what: str, command: str, detail: str = "") -> None:
    """Render an actionable notice for a missing artifact.

    Args:
        what: What is missing, in plain language.
        command: The exact shell command that produces it.
        detail: Optional extra context (e.g. the underlying error).
    """
    st.warning(f"**{what} is not available yet.**")
    if detail:
        st.caption(detail)
    st.markdown("Generate it by running:")
    st.code(command, language="bash")


def advisory_banner() -> None:
    """Render the standing advisory-only disclaimer."""
    st.info(f":warning: {ADVISORY_DISCLAIMER}")


def metric_row(items: list[tuple[str, Any, str | None]]) -> None:
    """Render a row of Streamlit metrics.

    Args:
        items: ``(label, value, help_text)`` triples.
    """
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)


def risk_badge(level: str) -> str:
    """Return a coloured HTML badge for a risk level.

    Args:
        level: ``"low"``, ``"medium"`` or ``"high"``.

    Returns:
        An inline HTML span.
    """
    colour = RISK_COLOURS.get(level.lower(), "#4a5568")
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.85em;font-weight:600'>{level.upper()}</span>"
    )


def health_badge(health_class: str) -> str:
    """Return a coloured HTML badge for a health class.

    Args:
        health_class: ``"healthy"``, ``"monitor"``, ``"degraded"`` or
            ``"critical"``.

    Returns:
        An inline HTML span.
    """
    colour = HEALTH_COLOURS.get(health_class.lower(), "#4a5568")
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.85em;font-weight:600'>{health_class.upper()}</span>"
    )


def format_metrics_table(metrics: dict[str, Any], keys: list[str]) -> dict[str, str]:
    """Format selected metrics for display.

    Args:
        metrics: A metric dictionary.
        keys: Metric names to extract.

    Returns:
        A mapping of metric name to formatted string, using ``"n/a"`` for
        anything absent.
    """
    out: dict[str, str] = {}
    for key in keys:
        value = metrics.get(key)
        if value is None:
            out[key] = "n/a"
        elif isinstance(value, float):
            out[key] = f"{value:.4f}"
        else:
            out[key] = str(value)
    return out


def show_figure(path, caption: str, fallback_command: str) -> None:
    """Display a figure, or explain how to generate it when it is missing.

    Args:
        path: Path to a PNG file.
        caption: Image caption.
        fallback_command: Command that produces the figure.
    """
    if path is not None and path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"_{caption} is not available._")
        st.code(fallback_command, language="bash")
