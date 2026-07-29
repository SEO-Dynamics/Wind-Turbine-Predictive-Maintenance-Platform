"""Grounded explanations and advisory text for health assessments.

Every sentence produced here traces back to a value that was actually measured:
a rule exceedance, a deviation from the turbine's own baseline, a drift
statistic, or the score itself.  Nothing is inferred about a physical cause the
evidence does not show, and nothing is stated with more confidence than the
evidence supports.

This is a sibling of :mod:`wind_turbine_pm.explainability.narratives` rather
than a reuse of it, because the two describe different things.  The failure
narrative explains a *probability* in terms of SHAP attributions over model
features.  A health assessment explains a *condition score* in terms of named
components and the channels that drove them - evidence an engineer can check
directly against the raw trend, which is the whole point of the component
roll-up being rule-driven rather than model-driven.

Feature-name humanisation is shared with the failure module
(:func:`~wind_turbine_pm.explainability.narratives.humanise_feature`) and
extended here with the suffixes only the health feature set produces
(``_rms_24h``, ``_crest_24h``, ``_cusum_pos``, ``_regime_robust_z`` and so on),
so a feature added later is still described correctly without code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wind_turbine_pm.constants import (
    ADVISORY_DISCLAIMER,
    SENSOR_DISPLAY_NAMES,
    DriftSeverity,
    HealthClass,
    OperatingRegime,
)
from wind_turbine_pm.contracts.health import ComponentHealth, SensorDriftSignal
from wind_turbine_pm.explainability.narratives import humanise_feature
from wind_turbine_pm.health.components import component_label
from wind_turbine_pm.health.health_class import recommended_action

#: Health-specific suffix patterns, tried before the shared failure patterns.
#: Order matters: the first match wins, so more specific patterns come first.
_HEALTH_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(.*)_rms_(\d+)h$"), r"\2-hour RMS level of \1"),
    (re.compile(r"^(.*)_p2p_(\d+)h$"), r"\2-hour peak-to-peak swing in \1"),
    (re.compile(r"^(.*)_crest_(\d+)h$"), r"\2-hour crest factor of \1"),
    (re.compile(r"^(.*)_kurtosis_(\d+)h$"), r"\2-hour impulsiveness of \1"),
    (re.compile(r"^(.*)_skew_(\d+)h$"), r"\2-hour skewness of \1"),
    (re.compile(r"^(.*)_hf_ratio_(\d+)h$"), r"\2-hour high-frequency content of \1"),
    (re.compile(r"^(.*)_zcr_(\d+)h$"), r"\2-hour zero-crossing rate of \1"),
    (
        re.compile(r"^(.*)_rule_exceedance_mean_(\d+)h$"),
        r"average \2-hour encroachment of \1 into its alarm band",
    ),
    (
        re.compile(r"^(.*)_rule_exceedance_max_(\d+)h$"),
        r"worst \2-hour encroachment of \1 into its alarm band",
    ),
    (re.compile(r"^(.*)_rule_exceedance$"), r"how far \1 is into its alarm band"),
    (
        re.compile(r"^(.*)_dev_from_regime_baseline$"),
        r"deviation of \1 from this turbine's own baseline for the current operating regime",
    ),
    (
        re.compile(r"^(.*)_regime_robust_z$"),
        r"how unusual \1 is for this turbine in the current operating regime",
    ),
    (re.compile(r"^(.*)_cusum_pos$"), r"accumulated upward drift in \1"),
    (re.compile(r"^(.*)_cusum_neg$"), r"accumulated downward drift in \1"),
    (re.compile(r"^(.*)_drift_z$"), r"the standardised drift residual of \1"),
    (re.compile(r"^(.*)_ewma$"), r"the smoothed drift residual of \1"),
)

#: Fully custom names for health features whose structure is not regular.
_HEALTH_EXPLICIT_NAMES: dict[str, str] = {
    "rule_warning_count": "the number of channels above their warning limit",
    "rule_alarm_count": "the number of channels above their alarm limit",
    "rule_validity_flag_count": "the number of channels failing a validity check",
    "oil_pressure_per_load": "oil pressure relative to load",
}

#: Component score below which the roll-up is worth naming to an operator.
#: Above it the deduction is ordinary operating variation rather than a finding.
COMPONENT_ATTENTION_SCORE: float = 80.0

#: Human-readable phrasing for each operating regime.
_REGIME_PHRASES: dict[OperatingRegime, str] = {
    OperatingRegime.OFFLINE: "offline (maintenance or fault reported by the controller)",
    OperatingRegime.IDLE: "idle and not meaningfully producing",
    OperatingRegime.LOW_LOAD: "running at low load",
    OperatingRegime.MEDIUM_LOAD: "running at medium load",
    OperatingRegime.HIGH_LOAD: "running at high load",
    OperatingRegime.CURTAILED: "producing well below its power curve, consistent with curtailment",
}


@dataclass(frozen=True)
class HealthAdvisory:
    """A narrative explanation paired with an advisory recommendation."""

    explanation: str
    recommendation: str


def humanise_health_feature(name: str) -> str:
    """Turn a health feature name into a readable phrase.

    Health-specific patterns are tried first; anything else falls through to the
    shared platform humaniser, so rolling, trend and physical features are
    described identically in both modules.

    Args:
        name: The raw feature name.

    Returns:
        A human-readable description.
    """
    if name in _HEALTH_EXPLICIT_NAMES:
        return _HEALTH_EXPLICIT_NAMES[name]
    if name.startswith("regime_"):
        regime = name.removeprefix("regime_")
        return f"the turbine operating in the '{regime}' regime"

    for pattern, template in _HEALTH_SUFFIX_PATTERNS:
        match = pattern.match(name)
        if match:
            phrase = pattern.sub(template, name)
            base = match.group(1)
            readable_base = SENSOR_DISPLAY_NAMES.get(base, base.replace("_", " "))
            return phrase.replace(base, readable_base, 1)

    return humanise_feature(name)


def _join(items: list[str]) -> str:
    """Join phrases into a natural English list."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def describe_drift_signal(signal: SensorDriftSignal) -> str:
    """Describe one drift signal in a single clause.

    Any trailing full stop is removed so the caller can punctuate the sentence
    the clause is embedded in.

    Args:
        signal: The fired drift signal.

    Returns:
        A clause naming the channel, the detector and the severity.
    """
    if signal.description:
        return signal.description.rstrip().rstrip(".")
    label = SENSOR_DISPLAY_NAMES.get(signal.sensor, signal.sensor.replace("_", " "))
    return (
        f"{label} shows a {signal.severity.value}-level {signal.method.replace('_', ' ')} "
        f"drift signal at {signal.statistic:.2f} against a limit of {signal.control_limit:.2f}"
    )


def build_health_explanation(
    score: float,
    health_class: HealthClass,
    regime: OperatingRegime,
    components: list[ComponentHealth],
    drift_signals: list[SensorDriftSignal],
    drift_penalty: float = 0.0,
    data_quality: float = 1.0,
) -> str:
    """Compose the assessment narrative from measured evidence.

    The narrative is built in a fixed order - score, operating context, worst
    component and its drivers, drift, data quality - so two assessments are
    directly comparable and an operator learns where to look.

    Args:
        score: The published 0-100 health score.
        health_class: The band the score falls into.
        regime: Operating regime the assessment was made in.
        components: The component roll-up, worst first.
        drift_signals: Drift signals that fired; empty means none.
        drift_penalty: Points deducted from the raw score for drift.
        data_quality: Share of the assessed window that passed validity checks.

    Returns:
        The narrative paragraph.
    """
    parts: list[str] = [
        f"Health score {score:.1f}/100, classified as {health_class.value} "
        f"while {_REGIME_PHRASES.get(regime, regime.value.replace('_', ' '))}."
    ]

    notable = [component for component in components if component.score < COMPONENT_ATTENTION_SCORE]
    if notable:
        worst = notable[0]
        parts.append(
            f"The lowest-scoring component is the {component_label(worst.component)} at "
            f"{worst.score:.1f}/100."
        )
        if worst.drivers:
            parts.append(f"That is driven by: {_join(worst.drivers)}.")
        others = [
            f"{component_label(component.component)} ({component.score:.0f})"
            for component in notable[1:3]
        ]
        if others:
            parts.append(f"Also below nominal: {_join(others)}.")
    else:
        parts.append(
            "No component is showing a rule exceedance or sustained deviation from this "
            "turbine's own baseline."
        )

    if drift_signals:
        alarms = [s for s in drift_signals if s.severity is DriftSeverity.ALARM]
        leading = alarms[:2] if alarms else drift_signals[:2]
        parts.append(
            f"{len(drift_signals)} sensor-drift signal(s) were detected: "
            f"{_join([describe_drift_signal(signal) for signal in leading])}."
        )
        if drift_penalty > 0:
            parts.append(
                f"{drift_penalty:.1f} points were deducted for that drift, because a channel "
                "that has moved away from its own baseline makes the score itself less "
                "trustworthy."
            )
    else:
        parts.append("No sensor drift was detected against this turbine's own baselines.")

    if data_quality < 1.0:
        parts.append(
            f"{(1.0 - data_quality):.0%} of the assessed window failed at least one sensor "
            "validity check (out-of-range, frozen or impossibly fast change), so treat the "
            "assessment with corresponding caution."
        )

    return " ".join(parts)


def build_health_recommendation(
    health_class: HealthClass,
    components: list[ComponentHealth],
    drift_signals: list[SensorDriftSignal],
) -> str:
    """Compose the advisory maintenance message.

    The message escalates with the health class, names the component and the
    drifting channels where they exist, and always carries the platform
    disclaimer.  It never instructs anyone to stop, start or reconfigure a
    machine.

    Args:
        health_class: The classified health band.
        components: The component roll-up, worst first.
        drift_signals: Drift signals that fired.

    Returns:
        The advisory message.
    """
    body = recommended_action(health_class)

    # A component is named whenever one is meaningfully off-baseline, including
    # when the overall class is Healthy. The two can legitimately disagree - the
    # score is a fleet-trained estimate of overall condition, the component score
    # is this turbine's own rule and baseline evidence - and silently dropping the
    # component finding would leave an operator reading "no action required"
    # while one subsystem is in its alarm band.
    degraded = [
        component for component in components if component.score < COMPONENT_ATTENTION_SCORE
    ]
    if degraded:
        worst = degraded[0]
        if health_class is HealthClass.HEALTHY:
            body += (
                f" Note that the {component_label(worst.component)} scores "
                f"{worst.score:.0f}/100 on its own rule and baseline evidence despite the "
                "overall score being in the healthy band; review that channel's trend before "
                "dismissing it."
            )
        else:
            body += f" Start with the {component_label(worst.component)}, which scores lowest."

    drifting = sorted(
        {signal.sensor for signal in drift_signals if signal.sensor != "multivariate"}
    )
    if drifting:
        labels = [SENSOR_DISPLAY_NAMES.get(sensor, sensor.replace("_", " ")) for sensor in drifting]
        body += (
            f" Separately, verify the calibration of {_join(labels)}: the channel(s) have drifted "
            "from their own baseline, which may be an instrument fault rather than a change in "
            "the machine."
        )

    return f"{body} {ADVISORY_DISCLAIMER}"


def build_health_advisory(
    score: float,
    health_class: HealthClass,
    regime: OperatingRegime,
    components: list[ComponentHealth],
    drift_signals: list[SensorDriftSignal],
    drift_penalty: float = 0.0,
    data_quality: float = 1.0,
) -> HealthAdvisory:
    """Produce both the narrative and the recommendation.

    Args:
        score: The published 0-100 health score.
        health_class: The band the score falls into.
        regime: Operating regime the assessment was made in.
        components: The component roll-up, worst first.
        drift_signals: Drift signals that fired.
        drift_penalty: Points deducted for drift.
        data_quality: Share of the assessed window that passed validity checks.

    Returns:
        The paired :class:`HealthAdvisory`.
    """
    return HealthAdvisory(
        explanation=build_health_explanation(
            score, health_class, regime, components, drift_signals, drift_penalty, data_quality
        ),
        recommendation=build_health_recommendation(health_class, components, drift_signals),
    )
