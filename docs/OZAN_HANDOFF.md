# Handoff — Failure Prediction Module

**From:** Ozan (Stage 1 — Failure Prediction)
**To:** Stage 2 (Turbine Health Monitoring) and Stage 3 (Anomaly Detection & Maintenance
Decision Support)
**Branch:** `feature/ozan-failure-prediction`
**Module version:** 1.0.0
**Date:** 2026-07-29

This document is the contract between my module and yours. Everything here is stable and
safe to build on. Read §9 before changing anything.

---

## 1. Completed components

| Area | What exists | Where |
|---|---|---|
| Configuration | Layered YAML (`base` + `data` + `features` + `failure_model`), deep-merged, attribute + dotted access, invariants validated at load | `src/wind_turbine_pm/config.py`, `configs/` |
| Paths | Repo-root discovery via `pyproject.toml` marker, `WTPM_PROJECT_ROOT` override, no hard-coded paths anywhere | `utils/paths.py` |
| Logging | Structured stdout logging, optional JSON, idempotent setup | `logging_config.py` |
| Reproducibility | Single seed; per-stream seeds derived by hash so turbine streams stay independent | `utils/reproducibility.py` |
| IO | Atomic writes, Parquet/CSV/joblib/JSON, `ArtifactNotFoundError` carrying a fix command | `utils/io.py` |
| **Contracts** | `TurbineObservation`, `TurbineWindow`, `BasePrediction`, `FailurePrediction`, `RiskFactor`, `ModelMetadata` | `contracts/` |
| Synthetic data | Physics-flavoured 20-turbine generator with degradation episodes, benign episodes, injected defects | `data/synthetic.py` |
| Ingestion | `synthetic` and `file` sources behind one interface | `data/ingestion.py` |
| **Validation** | 13 checks, severity-tagged findings, machine-readable report | `data/validation.py` |
| Preprocessing | Key normalisation, de-duplication, invalid-value treatment, past-only imputation | `data/preprocessing.py` |
| Target | `failure_within_48h` with documented edge cases | `data/preprocessing.py` |
| Splitting | Chronological split with embargo, integrity verification | `data/splitting.py` |
| **Feature primitives** | Leakage-safe grouped lag / rolling / slope / expanding / robust-z | `features/transformers.py` |
| Features | 395 features in 7 groups, one entry point | `features/failure_features.py` |
| Models | 4 candidates, cost-aware selection with tolerance band and guard rails | `models/` |
| Threshold | F2 and cost methods, validation-only, full curve persisted | `models/threshold.py` |
| Explainability | SHAP tree/permutation with fallbacks, grounded narratives, advisory layer | `explainability/` |
| **Service** | `FailurePredictionService` — the single prediction path | `services/` |
| API | FastAPI, 9 endpoints, structured errors, lazy loading | `api/` |
| Dashboard | Streamlit, 6 sections, degrades gracefully | `dashboard/` |
| Tests | 162 tests, ~14 s, deterministic | `tests/` |
| Ops | Dockerfile, compose (API + dashboard + opt-in pipeline), GitHub Actions CI, Makefile | root |

---

## 2. Data contract

The canonical raw SCADA schema. **Import the column names from
`wind_turbine_pm.constants` — do not hard-code strings.**

### Required columns

| Column | Type | Unit | Notes |
|---|---|---|---|
| `timestamp` | `datetime64[ns]` | — | ISO-8601, UTC recommended. Naive datetimes are treated as UTC. |
| `turbine_id` | `str` | — | `^[A-Za-z0-9_\-]{1,32}$`. Generator uses `T01`…`T20`. |
| `wind_speed` | `float` | m/s | 0–45 |
| `rotor_speed` | `float` | rpm | 0–25 |
| `generator_speed` | `float` | rpm | 0–2200 |
| `power_output` | `float` | kW | −100–3600 |
| `generator_temperature` | `float` | °C | −30–200 |
| `gearbox_temperature` | `float` | °C | −30–160 |
| `bearing_temperature` | `float` | °C | −30–160 |
| `oil_temperature` | `float` | °C | −30–130 |
| `oil_pressure` | `float` | bar | 0–12 |
| `vibration` | `float` | mm/s RMS | 0–25 |
| `ambient_temperature` | `float` | °C | −45–55 |
| `nacelle_temperature` | `float` | °C | −45–90 |
| `hydraulic_pressure` | `float` | bar | 0–300 |
| `brake_temperature` | `float` | °C | −30–400 |
| `operational_status` | `str` | — | `normal` \| `idle` \| `derated` \| `maintenance` \| `fault` |
| `failure_event` | `int8` | — | 0/1, marks the moment of failure |

Ranges live in `configs/data.yaml → validation.physical_ranges` and are enforced by both
the validation layer and the Pydantic observation contract.

### Optional ground-truth columns (synthetic only)

`maintenance_event`, `degradation_level`, `failure_mode`, `episode_id`,
`hours_to_failure`.

> **These are for EDA and analysis only.** They are listed in
> `features.exclude_columns` and can never become model features. If you build a
> supervised model, exclude them the same way.

### Constants to import

```python
from wind_turbine_pm.constants import (
    TIMESTAMP,
    TURBINE_ID,
    SENSOR_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    OPERATIONAL_STATUS,
    FAILURE_EVENT,
    COLUMN_UNITS,
    SENSOR_DISPLAY_NAMES,
    OperationalStatus,
    RiskLevel,
    SplitName,
    ValidationSeverity,
    ADVISORY_DISCLAIMER,
)
```

### Target definition

`failure_within_48h` = 1 iff a `failure_event` occurs for **the same turbine** in
`(t, t + 48h]`. Full edge-case documentation in
[README §10](../README.md#10-target-definition). If your module needs a different horizon,
call `create_failure_target()` with a modified config rather than reimplementing it.

---

## 3. Feature contract

### Entry point

```python
from wind_turbine_pm.features.failure_features import build_failure_features

features, spec = build_failure_features(preprocessed_frame, cfg)
```

**Input requirements:**

- Preprocessed via `wind_turbine_pm.data.preprocessing.preprocess()`
- Sorted by `turbine_id`, then `timestamp`
- Non-empty; contains `turbine_id` and `timestamp`
- At least 48 hours of history per turbine for the longest windows to be defined

**Output:**

- `features` — `DataFrame`, `float32`, **same index as the input** so you can join labels
  and metadata back. Model-facing columns only.
- `spec` — `FeatureSpec` with `names` (ordered tuple), `groups` (name → members) and
  `step_hours`.

### Feature ordering rules

**Order is part of the contract.** The order in `spec.names` is the order the model
expects and is persisted in `ModelMetadata.features`.

- Never reorder columns before scoring. Use
  `align_to_feature_order(features, metadata.features)`.
- The service validates order on every prediction and **raises** on mismatch — it does
  not silently coerce.
- Adding a feature changes the order → **retrain and republish**. A model artifact and a
  feature pipeline are versioned together.

### Reusable primitives for your module

`features/transformers.py` is deliberately generic — nothing in it is
failure-specific. Use these rather than writing your own rolling logic:

| Function | Guarantee |
|---|---|
| `grouped_lag(series, groups, periods)` | Never crosses a group boundary |
| `grouped_diff` / `grouped_pct_change` | Guarded against divide-by-zero |
| `grouped_rolling(series, groups, window, stat)` | Trailing window, inclusive of current row |
| `rolling_slope(series, groups, window)` | Closed-form trailing OLS slope |
| `expanding_baseline(series, groups, stat)` | **Shifted by one row** — excludes the current observation |
| `robust_zscore(series, groups)` | Expanding median / MAD, outlier-resistant |
| `cyclical_encode(values, period, name)` | Sine/cosine encoding |
| `assert_no_future_leakage(...)` | Test helper: perturbs the future, asserts the past is unchanged |

**If you write a new temporal feature, add a leakage test.** Copy the pattern from
`tests/test_failure_features.py::test_no_future_leakage`.

---

## 4. Artifact contract

All paths come from `configs/base.yaml → paths` and are resolved by
`utils/paths.resolve()`. **Use the accessors in `models/persistence.py`, not literals.**

| Artifact | Path | Accessor |
|---|---|---|
| Model (pipeline) | `artifacts/models/failure_model.joblib` | `model_path(cfg)` |
| SHAP background | `artifacts/models/shap_background.parquet` | `background_path(cfg)` |
| Model metadata | `artifacts/metadata/failure_model_metadata.json` | `metadata_path(cfg)` |
| Threshold | `artifacts/metadata/failure_threshold.json` | `threshold_path(cfg)` |
| Metrics | `artifacts/metrics/failure_metrics.json` | `metrics_path(cfg)` |
| Model comparison | `artifacts/metrics/model_comparison.csv` | `comparison_path(cfg)` |
| Threshold curve | `artifacts/metrics/threshold_curve.csv` | — |
| Global importance | `artifacts/metrics/global_feature_importance.csv` | — |
| Validation reports | `artifacts/metrics/{raw_,}validation_report.json` | — |
| Figures | `artifacts/figures/*.png` | `figures_dir(cfg)` |
| Processed data | `data/processed/failure_{dataset,features}.parquet` | `scripts/prepare_data.load_prepared(cfg)` |
| Feature spec | `data/processed/feature_spec.json` | — |
| Split boundaries | `data/processed/split_boundaries.json` | — |

**Naming convention for your module:** prefix artifacts with your module name, e.g.
`artifacts/models/health_model.joblib`,
`artifacts/metadata/health_model_metadata.json`. Do not overwrite anything named
`failure_*`.

**Loading:**

```python
from wind_turbine_pm.models.persistence import load_bundle, bundle_available

if bundle_available(cfg):
    bundle = load_bundle(cfg)  # .estimator, .metadata, .features, .threshold, .version
```

`load_bundle` validates the metadata against the Pydantic contract, so a corrupted or
hand-edited artifact fails loudly at load rather than producing silently wrong output.

**Publish your own model** with `ModelMetadata` and `save_bundle()` — the dashboard and
API can then render it with no new code.

---

## 5. API contract

Base: `http://localhost:8000` · Swagger: `/docs`

| Method | Endpoint | Success | Errors |
|---|---|---|---|
| `GET` | `/` | 200 | — |
| `GET` | `/health` | 200 (`ok` or `degraded`) | — |
| `GET` | `/failure/model-info` | 200 | 503 |
| `GET` | `/failure/metrics` | 200 | 404, 503 |
| `GET` | `/failure/features` | 200 | 503 |
| `GET` | `/failure/example` | 200 | 503 |
| `POST` | `/failure/predict` | 200 | 422, 503 |
| `POST` | `/failure/predict/batch` | 200 | 422, 503 |
| `POST` | `/failure/predict/prepared` | 200 | 422, 503 |

### Request — `POST /failure/predict`

```json
{
  "turbine_id": "T01",
  "observations": [
    { "turbine_id": "T01", "timestamp": "2024-11-28T19:00:00Z",
      "wind_speed": 8.4, "rotor_speed": 12.1, "generator_speed": 1174.0,
      "power_output": 720.0, "generator_temperature": 62.0,
      "gearbox_temperature": 55.0, "bearing_temperature": 48.0,
      "oil_temperature": 46.0, "oil_pressure": 5.1, "vibration": 3.2,
      "ambient_temperature": 11.0, "nacelle_temperature": 19.0,
      "hydraulic_pressure": 188.0, "brake_temperature": 17.0,
      "operational_status": "normal" }
  ]
}
```

Requires ≥72 hours of chronologically ordered observations for one turbine. Sensor fields
are optional (nullable); `turbine_id` and `timestamp` are not. Unknown fields are
rejected.

### Response

```json
{
  "turbine_id": "T19", "timestamp": "2024-11-28T19:00:00Z",
  "failure_probability": 0.923624, "prediction": 1,
  "risk_level": "high", "threshold": 0.06, "horizon_hours": 48,
  "top_risk_factors": [
    { "feature": "vibration_roll_max_48h", "impact": 0.41,
      "direction": "increases_risk", "value": 7.82,
      "description": "48-hour peak vibration was elevated" }
  ],
  "explanation": "The model estimates a 92.4% probability ...",
  "recommendation": "Prioritise review ...",
  "model_version": "1.0.0", "advisory_only": true
}
```

### Error envelope

All handled errors return a consistent body:

```json
{ "error": "<machine_readable_code>", "detail": "<human readable>", "hint": "<fix or null>" }
```

Codes: `validation_error`, `insufficient_history`, `invalid_window`, `invalid_batch`,
`feature_contract_violation`, `metrics_unavailable`, `model_unavailable`.

**`/health` returns 200 even with no model**, reporting `status: "degraded"`. Keep this
behaviour for your endpoints — orchestrators need liveness separate from readiness.

---

## 6. Extension points

### 6.1 Adding a Health Monitoring router

1. Create `src/wind_turbine_pm/api/routers/health_monitoring.py`:

```python
from fastapi import APIRouter
router = APIRouter(prefix="/health-monitoring", tags=["health-monitoring"])

@router.get("/score")
def fleet_health_score(...): ...

def router_description() -> dict:
    return {"module": "turbine_health_monitoring", "prefix": router.prefix,
            "status": "available",
            "endpoints": sorted({r.path for r in router.routes})}
```

2. Register it in `api/main.py` — **one line, nothing else changes**:

```python
MODULE_ROUTERS = [
    (failure.router, failure.router_description),
    (health_monitoring.router, health_monitoring.router_description),  # add
]
```

3. Move your module out of `planned_modules` in the `root()` handler.

### 6.2 Adding a Health Monitoring dashboard page

1. Create `dashboard/pages/health_monitoring.py` exposing `render() -> None`.
2. Replace the placeholder entry in `dashboard/app.py`:

```python
(Page("Turbine Health Monitoring", health_monitoring.render, available=True),)
```

3. Reuse `dashboard/components/` — `missing_artifact_notice`, `metric_row`, `risk_badge`,
   `show_figure`, and the chart builders. **Please reuse `missing_artifact_notice`** so
   the whole dashboard degrades consistently.
4. Add loaders to `dashboard/data_access.py` following the existing pattern: return
   `None` on a missing artifact, never raise.

### 6.3 Adding an Anomaly Detection router

Same as §6.1 with `prefix="/anomaly"`. Consume `TurbineWindow` so the same ingestion path
feeds you. Subclass `BasePrediction` for your output so the dashboard can render it:

```python
class AnomalyPrediction(BasePrediction):
    anomaly_score: float
    is_anomaly: bool
    contributing_signals: list[RiskFactor]
```

### 6.4 Building the unified risk service

Compose the existing services; **do not reach into their internals**.

```python
class UnifiedRiskService:
    def __init__(self, cfg):
        self._failure = FailurePredictionService(cfg)
        self._health = HealthMonitoringService(cfg)  # yours
        self._anomaly = AnomalyDetectionService(cfg)  # yours

    def assess(self, window: TurbineWindow) -> UnifiedRiskAssessment:
        components = []
        if self._failure.is_ready:
            components.append(self._failure.predict_from_window(window))
        ...
```

Guidance:

- Check `is_ready` before calling — a module whose artifacts are missing must degrade the
  unified score, not break it.
- Put the combination weights in `configs/`, never in code.
- Keep `advisory_only: true` on the combined output.
- Reuse `map_risk_level()` from `models/threshold.py` so bands stay consistent.

### 6.5 Adding configuration

Create `configs/<your_module>.yaml` and load it alongside the existing files:

```python
cfg = load_config(("data.yaml", "features.yaml", "failure_model.yaml", "health_model.yaml"))
```

Add cross-cutting invariants to `config.validate_config()` if your module needs them.

---

## 7. Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
make install                       # pip install -r requirements.txt && pip install -e . --no-deps
# If `import wind_turbine_pm` fails after an editable install:
export PYTHONPATH=src

# Full pipeline (~2.5 min)
make pipeline                      # python scripts/run_failure_pipeline.py

# Stage by stage
python scripts/generate_synthetic_data.py
python scripts/prepare_data.py
python scripts/train_failure_model.py
python scripts/evaluate_failure_model.py

# Services
make api                           # uvicorn wind_turbine_pm.api.main:app --reload
make dashboard                     # streamlit run dashboard/app.py

# Quality
make test                          # pytest        (162 tests, ~14 s)
make lint                          # ruff check .
make format                        # ruff format . && ruff check --fix .

# Docker
docker compose up --build
docker compose --profile pipeline run --rm pipeline
```

**Verification one-liners:**

```bash
python -c "from wind_turbine_pm.api.main import app; print(app.title)"
curl -s localhost:8000/health | python -m json.tool
```

---

## 8. Known limitations

Be aware of these before building on top:

1. **Synthetic data.** No result here is evidence of real-world performance. Do not quote
   these metrics as validated.
2. **Precision 0.42, FNR 0.157.** About 3 in 5 flags are false alarms and about 1 in 6
   failures is missed. If you combine this into a unified score, propagate that
   uncertainty — do not treat the probability as ground truth.
3. **Feature build is ~20 s for 175k rows.** Acceptable in batch, and single-window
   serving is fast, but do not call `build_failure_features()` per row in a loop.
4. **`predict_from_window` needs ≥72 h of history.** No prediction is possible for a
   newly commissioned turbine until it has that.
5. **No uncertainty quantification.** A probability of 0.62 has no confidence interval.
6. **Calibration and threshold share the validation split**, so validation threshold
   metrics are mildly optimistic. Test metrics are unaffected.
7. **Per-turbine median imputation** uses a full-history location statistic — a
   deliberate, documented deviation from strict past-only purity.
8. **Only 80 failure events**, ~19 in the test split. Metric confidence intervals are
   wide. Do not over-interpret small differences.
9. **The service caches the model per process** (`lru_cache`). Retraining requires a
   restart; there is no hot reload.
10. **The dashboard imports the service directly** rather than calling the API over HTTP.
    Fine for a single-host deployment; if you split the processes, add an HTTP client
    behind the same interface.
11. **No authentication on the API.** It is a local/demo service. Add auth before exposing
    it anywhere.
12. **`month`/`day_of_week` are excluded by default** and this is intentional — see
    `configs/features.yaml`. Do not re-enable them without re-measuring.

---

## 9. Files you should not change unnecessarily

These are **shared contracts and core utilities**. Other modules and the persisted
artifacts depend on them. Changing them is a breaking change for the whole platform.

### 🔴 Do not change without coordinating

| File | Why |
|---|---|
| `contracts/observations.py` | Shared input schema for every module |
| `contracts/predictions.py` | Shared output schema; the dashboard renders against it |
| `contracts/metadata.py` | Every published model serialises this; changing it invalidates saved artifacts |
| `constants.py` | Column names and enums used platform-wide; a rename breaks saved data |
| `utils/paths.py` | Path resolution for every module |
| `utils/io.py` | Artifact IO and the missing-artifact error contract |
| `config.py` | Configuration loading and merge semantics |

**If you must change one:** open an issue first, bump the version, and update this
document. Additive changes (a new optional field) are usually safe; renames and removals
are not.

### 🟡 Extend, don't modify

| File | How to extend |
|---|---|
| `api/main.py` | Append to `MODULE_ROUTERS` only |
| `dashboard/app.py` | Append to `PAGES` only |
| `features/transformers.py` | Add new primitives; don't change existing semantics |
| `data/validation.py` | Add new checks; don't change existing severities |
| `models/threshold.py` | `map_risk_level` is shared — keep band semantics |
| `configs/base.yaml` | Add keys; don't remove or rename |

### 🟢 Yours to own

Anything under a new module namespace — new routers, new dashboard pages, new services,
new configs, new artifacts prefixed with your module name.

### 🔵 Failure-prediction internals — please leave alone

`data/synthetic.py`, `data/preprocessing.py`, `features/failure_features.py`,
`models/*`, `explainability/*`, `services/failure_prediction_service.py`,
`api/routers/failure.py`, `dashboard/pages/failure_prediction.py`.

If you need behaviour from these, call them. If you need *different* behaviour, add a
config option rather than editing the logic — nearly everything is already
config-driven.

---

## 10. Suggested commit groups

```
chore: initialize project structure, configuration and tooling
feat: add deterministic physics-flavoured SCADA data generation
feat: add data validation and preprocessing layers
feat: implement 48-hour failure target and leakage-safe feature pipeline
feat: add chronological splitting with embargo
feat: train and evaluate failure prediction models
feat: add threshold optimisation and SHAP explanations
feat: add advisory recommendation and prediction service
feat: implement failure prediction API
feat: add Streamlit failure prediction dashboard
test: add failure module test suite
build: add Docker, CI workflow and Makefile
docs: add README, model card, EDA notebook and handoff contract
```

---

## 11. Questions

Start with:

1. `README.md` — architecture, results, design decisions
2. `MODEL_CARD.md` — intended use, limitations, risks
3. `notebooks/01_failure_prediction_eda.ipynb` — the data and the leakage discussion
4. `tests/` — the tests document expected behaviour precisely; a test name is usually a
   faster answer than a docstring

Then open an issue on the repository.

Good luck. 🌬️
