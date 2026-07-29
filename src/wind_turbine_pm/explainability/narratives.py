"""Human-readable explanations and advisory recommendations.

Every sentence produced here is derived from an actual attribution value.  The
generator never invents a driver, never asserts a physical cause it cannot see
in the attributions, and never states or implies certainty.  When no
attributions are available it says so explicitly instead of producing plausible
filler.

Feature names are decoded structurally (``vibration_roll_mean_12h`` becomes
"12-hour average vibration") rather than through a hand-maintained lookup
table, so features added later are described correctly without code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wind_turbine_pm.constants import ADVISORY_DISCLAIMER, SENSOR_DISPLAY_NAMES, RiskLevel
from wind_turbine_pm.explainability.shap_explainer import LocalAttribution

#: Suffix patterns decoded into readable phrases. Order matters: the first
#: match wins, so more specific patterns come first.
_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(.*)_roll_(mean|median)_(\d+)h$"), r"\3-hour average \1"),
    (re.compile(r"^(.*)_roll_std_(\d+)h$"), r"\2-hour variability of \1"),
    (re.compile(r"^(.*)_roll_range_(\d+)h$"), r"\2-hour range of \1"),
    (re.compile(r"^(.*)_roll_max_(\d+)h$"), r"\2-hour peak \1"),
    (re.compile(r"^(.*)_roll_min_(\d+)h$"), r"\2-hour minimum \1"),
    (re.compile(r"^(.*)_slope_(\d+)h$"), r"\2-hour trend in \1"),
    (re.compile(r"^(.*)_rate_(\d+)h$"), r"rate of change of \1 over \2 hours"),
    (re.compile(r"^(.*)_diff_(\d+)h$"), r"\2-hour change in \1"),
    (re.compile(r"^(.*)_pct_change_(\d+)h$"), r"\2-hour relative change in \1"),
    (re.compile(r"^(.*)_lag_(\d+)h$"), r"\1 \2 hours ago"),
    (re.compile(r"^(.*)_dev_from_(\d+)h_baseline$"), r"deviation of \1 from its \2-hour baseline"),
    (
        re.compile(r"^(.*)_dev_from_turbine_(median|mean)$"),
        r"deviation of \1 from this turbine's usual level",
    ),
    (
        re.compile(r"^(.*)_dev_from_normal_regime$"),
        r"deviation of \1 from its normal-operation baseline",
    ),
    (re.compile(r"^(.*)_turbine_robust_z$"), r"how unusual \1 is for this turbine"),
    (re.compile(r"^(.*)_temp_above_ambient$"), r"\1 temperature rise above ambient"),
    (re.compile(r"^(.*)_per_load$"), r"\1 relative to load"),
)

#: Fully custom names for engineered features whose structure is not regular.
_EXPLICIT_NAMES: dict[str, str] = {
    "rotor_generator_speed_ratio": "the rotor-to-generator speed ratio",
    "power_ratio": "power output relative to the expected power curve",
    "power_curve_residual": "the shortfall against the expected power curve",
    "power_per_wind_cubed": "power produced per unit of available wind energy",
    "expected_power": "the power the wind conditions should support",
    "load_factor": "the current load factor",
    "vibration_per_load": "vibration relative to the current load",
    "vibration_per_rotor_speed": "vibration relative to rotor speed",
    "oil_pressure_temp_ratio": "the oil pressure-to-temperature ratio",
    "oil_pressure_temp_product": "the combined oil pressure and temperature state",
    "thermal_stress_index": "the overall drivetrain thermal stress index",
    "thermal_spread": "the temperature spread across drivetrain components",
    "hydraulic_per_load": "hydraulic pressure relative to load",
}


@dataclass(frozen=True)
class Advisory:
    """A narrative explanation paired with an advisory recommendation."""

    explanation: str
    recommendation: str


def humanise_feature(name: str) -> str:
    """Turn a model feature name into a readable phrase.

    Args:
        name: The raw feature name.

    Returns:
        A human-readable description, falling back to the underscore-stripped
        name when no pattern matches.
    """
    if name in _EXPLICIT_NAMES:
        return _EXPLICIT_NAMES[name]
    if name.startswith("status_"):
        return f"the turbine reporting '{name.removeprefix('status_')}' status"
    if name in SENSOR_DISPLAY_NAMES:
        return SENSOR_DISPLAY_NAMES[name]

    for pattern, template in _SUFFIX_PATTERNS:
        match = pattern.match(name)
        if match:
            phrase = pattern.sub(template, name)
            # Replace the embedded sensor token with its display name.
            base = match.group(1)
            readable_base = SENSOR_DISPLAY_NAMES.get(base, base.replace("_", " "))
            return phrase.replace(base, readable_base, 1)

    return name.replace("_", " ")


def describe_factor(attribution: LocalAttribution) -> str:
    """Describe a single attribution in one clause.

    Args:
        attribution: The attribution to describe.

    Returns:
        A clause such as ``"12-hour average vibration was elevated"``.
    """
    phrase = humanise_feature(attribution.feature)
    verb = "was elevated" if attribution.impact >= 0 else "was within its expected range"
    return f"{phrase} {verb}"


def build_explanation(
    attributions: list[LocalAttribution],
    probability: float,
    risk_level: RiskLevel | str,
    explainer_method: str = "shap",
) -> str:
    """Compose a narrative from actual attribution values.

    Args:
        attributions: Ranked local attributions for one observation.
        probability: The predicted failure probability.
        risk_level: The mapped risk band.
        explainer_method: Method used, mentioned when it is not exact SHAP.

    Returns:
        A narrative sentence, or a clear statement that no attributions were
        available.
    """
    level = str(risk_level)
    if not attributions:
        return (
            f"The model estimates a {probability:.1%} probability of failure within the prediction "
            f"horizon ({level} risk). No feature-level attribution was available for this "
            "observation, so no driver-level explanation can be given."
        )

    increasing = [a for a in attributions if a.impact > 0]
    decreasing = [a for a in attributions if a.impact < 0]

    parts: list[str] = [
        f"The model estimates a {probability:.1%} probability of failure within the prediction "
        f"horizon ({level} risk)."
    ]

    if increasing:
        drivers = [humanise_feature(a.feature) for a in increasing[:3]]
        joined = _join(drivers)
        verb = "was" if len(drivers) == 1 else "were"
        parts.append(
            f"The estimate was pushed up mainly because {joined} {verb} outside the pattern the model associates with healthy operation."
        )

    if decreasing:
        protective = [humanise_feature(a.feature) for a in decreasing[:2]]
        joined = _join(protective)
        verb = "was" if len(protective) == 1 else "were"
        parts.append(
            f"Offsetting this, {joined} {verb} consistent with normal behaviour and reduced the estimate."
        )

    if not explainer_method.startswith("shap"):
        parts.append(
            "These drivers come from a permutation-importance approximation rather than exact "
            "SHAP attribution, so treat the ranking as indicative."
        )

    return " ".join(parts)


def _join(items: list[str]) -> str:
    """Join phrases into a natural English list."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def build_recommendation(
    risk_level: RiskLevel | str,
    attributions: list[LocalAttribution],
) -> str:
    """Compose an advisory maintenance message.

    The message escalates with the risk band, names the observed drivers where
    they exist, and always carries the human-review disclaimer.  It never
    instructs anyone to stop, start or reconfigure a machine.

    Args:
        risk_level: The mapped risk band.
        attributions: Ranked local attributions.

    Returns:
        The advisory message.
    """
    level = str(risk_level)
    drivers = [humanise_feature(a.feature) for a in attributions if a.impact > 0][:3]
    driver_clause = f" The signals contributing most were {_join(drivers)}." if drivers else ""

    if level == str(RiskLevel.LOW):
        body = (
            "Continue routine monitoring. No elevated failure indication was identified for this "
            "turbine in the current window."
        )
    elif level == str(RiskLevel.MEDIUM):
        body = (
            "Review recent sensor trends and consider scheduling a non-urgent diagnostic inspection "
            f"at the next convenient opportunity.{driver_clause}"
        )
    else:
        body = (
            "Prioritise review of the identified risk factors by a qualified maintenance engineer "
            f"before continued high-load operation.{driver_clause}"
        )

    return f"{body} {ADVISORY_DISCLAIMER}"


def build_advisory(
    attributions: list[LocalAttribution],
    probability: float,
    risk_level: RiskLevel | str,
    explainer_method: str = "shap",
) -> Advisory:
    """Produce both the narrative and the recommendation.

    Args:
        attributions: Ranked local attributions.
        probability: The predicted failure probability.
        risk_level: The mapped risk band.
        explainer_method: Method used to produce the attributions.

    Returns:
        The paired :class:`Advisory`.
    """
    return Advisory(
        explanation=build_explanation(attributions, probability, risk_level, explainer_method),
        recommendation=build_recommendation(risk_level, attributions),
    )
