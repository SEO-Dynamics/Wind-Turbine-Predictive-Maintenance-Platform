# Stage 2 Handoff — Turbine Health Monitoring

**Author:** Şahin (@SBRKBNL)
**Scope:** 0-100 health scoring, health classification, component roll-up, operating regimes, sensor validation and drift detection
**Module version:** 1.0.0
**Status:** ✅ Complete

> Part of the platform handoff set. Start at
> [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) for how the three modules fit together and
> which contracts are shared.

---

Everything in this section is stable and safe to consume.

### Consuming an assessment

Use the service, never the model file:

```python
from wind_turbine_pm.health.config import get_health_config
from wind_turbine_pm.services.health_monitoring_service import get_health_service

service = get_health_service()  # process-wide, lazily loaded, cached
if service.is_ready:
    assessment = service.assess_from_window(window)  # a shared TurbineWindow
```

`HealthAssessment` subclasses `BasePrediction`, so it already carries `turbine_id`,
`timestamp`, `model_version` and `advisory_only`. The health-specific fields:

| Field | Meaning |
|---|---|
| `health_score` | Published 0-100 score. **This is the number to use.** |
| `raw_health_score` | The model's own estimate, before the drift deduction |
| `drift_penalty` | Points deducted; `health_score == max(raw - penalty, 0)` is enforced by the contract |
| `health_class` | `healthy` / `monitor` / `degraded` / `critical` |
| `operating_regime` | Operating point the assessment was made in |
| `data_quality` | Share of the assessed window that passed the sensor validity checks |
| `component_health` | Per-component roll-up, worst first |
| `drift_signals` | Only signals that fired; an empty list means no drift |
| `rule_violations` | Envelope and validity findings |
| `top_factors` | Largest condition deviations, as shared `RiskFactor` objects |

Two assessment entry points, and they can legitimately disagree:

* `assess_from_window(window)` — computes features from the raw window. Use this for live
  data. Needs at least `service.minimum_history_hours()` of history.
* `assess_from_prepared(frame, features, turbine_id=..., as_of=...)` — uses an
  already-built feature matrix. Use this when you have the prepared artifacts, and use it
  for anything that must agree with `score_frame()`.

The difference is not a bug: expanding baselines and drift statistics depend on how much
history the caller supplied, so a short window is less informed rather than wrong.

### Bulk scoring and fleet roll-up

```python
scored = service.score_frame(dataset, features)  # one row per observation
summary = service.fleet_summary(scored)  # FleetHealthSummary
```

`score_frame` returns `turbine_id`, `timestamp`, `operating_regime`,
`raw_health_score`, `health_score` and `health_class`. It deliberately does **not**
assemble component roll-ups or narratives per row — that would be wasted work for a
fleet-sized frame where only the latest row per turbine is displayed.

### Combining health with failure risk

For the unified score in §6.4, the two modules answer different questions and should be
combined as different evidence, not averaged:

* failure probability — *"is something about to break in the next 48 hours?"* (sharp,
  event-driven, has a threshold)
* health score — *"what condition is this machine in, and is it getting worse?"* (slow,
  condition-driven, has bands)

A turbine can be Healthy with a high failure probability (a sudden fault signature on an
otherwise sound machine) or Degraded with a low one (worn but not near failure). Both
cases are operationally meaningful, so **do not collapse them into one number without
retaining both**. `HealthAssessment.component_health` is the part that says *where* to
send an engineer, which the failure probability cannot.

Also note that `drift_signals` are about the **instrument**, not the machine. Feed them
into a data-quality or confidence term, not into the severity of the machine's condition.

### Artifacts

All health artifacts are prefixed `health_`, so nothing collides with `failure_*`:

| Artifact | Path |
|---|---|
| Estimator | `artifacts/models/health_model.joblib` |
| Drift detector | `artifacts/models/health_drift_detector.joblib` |
| Feature background | `artifacts/models/health_background.parquet` |
| Metadata | `artifacts/metadata/health_model_metadata.json` |
| Metrics | `artifacts/metrics/health_metrics.json` |
| Candidate comparison | `artifacts/metrics/health_model_comparison.csv` |
| Error by band | `artifacts/metrics/health_error_by_band.csv` |
| Data quality | `artifacts/metrics/health_data_quality.json` |
| Prepared dataset | `data/processed/health_dataset.parquet` |
| Prepared features | `data/processed/health_features.parquet` |
| Feature spec | `data/processed/health_feature_spec.json` |
| Figures | `artifacts/figures/health_*.png` |

Read metadata through `HealthModelMetadata.model_validate(...)` rather than raw JSON, so a
corrupted artifact fails loudly. `health/persistence.py` has a path helper per artifact —
use those instead of composing paths yourself.

### Reusable pieces you may want

| Piece | Why you might want it |
|---|---|
| `health/regimes.py` → `attach_regimes` | Regime conditioning applies to anomaly detection too: an anomaly at idle is not the same as one at rated power |
| `health/sensor_rules.py` → `evaluate_rules` | Envelope and validity checks, and `data_quality()` as a confidence input |
| `health/drift.py` → `DriftCalibration` | The pattern for calibrating a detector against a healthy reference period rather than a textbook constant. **Read the class docstring before choosing control limits for your own detector** — the autocorrelation problem it documents applies to any control chart on this data |
| `health/drift.py` → `regime_conditioned_z` | Past-only robust z-score against a turbine's own regime-restricted baseline |
| `health/evaluation.py` | Regression figures: predicted-vs-actual, error by band, class confusion |
| `health/narratives.py` → `humanise_health_feature` | Feature-name humanisation that already covers the health suffixes |

### Configuration namespacing — please copy this

`configs/health_model.yaml` puts **every** key under a single top-level `health:`
namespace. That is deliberate: the file is deep-merged on top of `base` / `data` /
`features` / `failure_model`, so a shared top-level key such as `model:`, `training:` or
`serving:` would silently overwrite the Failure Prediction Module's settings. Namespace
your module the same way (`anomaly:`), and load it with a module-specific loader like
`health/config.py`.

### Health-specific limitations

Beyond §8:

1. **The health target comes from the simulator's `degradation_level`.** On a real fleet it
   must come from a source independent of the SCADA channels the features are built from.
   `build_health_target` raises with that instruction rather than inventing a target.
2. **The drift detector is not validated against real calibration drift.** This dataset
   contains none, so its thresholds are calibrated only to control the false-alarm rate on
   the healthy population. Its ability to catch a genuinely drifting instrument is
   untested. Do not present it as a validated instrument-fault detector.
3. **Component scores are rule- and baseline-driven, not model-derived.** They are
   auditable but will not always agree with the overall score, which is fleet-trained.
4. **Class boundaries and envelope limits are configuration**, chosen for a synthetic 2 MW
   fleet, and must be re-derived against real inspection outcomes and the operator's own
   alarm history.

---
