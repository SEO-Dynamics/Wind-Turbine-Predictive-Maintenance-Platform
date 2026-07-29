"""Streamlit entry point for the Wind Turbine Predictive Maintenance Platform.

Run with:
    streamlit run dashboard/app.py

Navigation is driven by :data:`PAGES`, which lists both the modules that exist
today and the ones planned for later stages.  A future engineer adds a page by
appending one entry and dropping a module with a ``render()`` function into
``dashboard/pages/`` - nothing else in this file changes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# Allow `streamlit run dashboard/app.py` from the repository root without an
# editable install of the dashboard package itself.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.pages import failure_prediction  # noqa: E402
from wind_turbine_pm import __version__  # noqa: E402
from wind_turbine_pm.constants import ADVISORY_DISCLAIMER  # noqa: E402


@dataclass(frozen=True)
class Page:
    """A dashboard page and its availability status."""

    label: str
    render: Callable[[], None] | None
    available: bool
    note: str = ""


def _coming_soon(module: str, description: str) -> Callable[[], None]:
    """Build a placeholder renderer for a module that is not built yet."""

    def render() -> None:
        st.title(module)
        st.info(
            f"**{module} is planned for a future development stage and is not implemented yet.**"
        )
        st.markdown(description)
        st.caption("See docs/OZAN_HANDOFF.md for the extension points this module should build on.")

    return render


#: Dashboard navigation. Future modules append entries here.
PAGES: list[Page] = [
    Page("Failure Prediction", failure_prediction.render, available=True),
    Page(
        "Turbine Health Monitoring",
        _coming_soon(
            "Turbine Health Monitoring",
            "- Fleet-level health scoring\n- Component condition indices\n- Sensor drift monitoring",
        ),
        available=False,
        note="Planned - Stage 2",
    ),
    Page(
        "Anomaly Detection",
        _coming_soon(
            "Anomaly Detection & Maintenance Decision Support",
            "- Unsupervised anomaly detection\n- Unified risk score\n"
            "- Maintenance prioritisation across models",
        ),
        available=False,
        note="Planned - Stage 3",
    ),
]


def main() -> None:
    """Configure the app shell and render the selected page."""
    st.set_page_config(
        page_title="Wind Turbine Predictive Maintenance",
        page_icon=":wind_face:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Wind Turbine PM")
        st.caption(f"Platform v{__version__}")
        labels = [
            page.label if page.available else f"{page.label}  ·  {page.note}" for page in PAGES
        ]
        choice = st.radio("Module", labels, index=0)
        selected = PAGES[labels.index(choice)]

        st.divider()
        st.markdown("**Roadmap**")
        for page in PAGES:
            marker = ":white_check_mark:" if page.available else ":hourglass_flowing_sand:"
            st.markdown(f"{marker} {page.label}")
        st.divider()
        st.caption(ADVISORY_DISCLAIMER)

    if selected.render is not None:
        selected.render()


main()
