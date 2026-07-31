# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-07-30

First complete release. All three modules are implemented, integrated and verified
end to end on synthetic data.

> **Advisory decision support only.** Trained and evaluated entirely on synthetic SCADA
> data. Not a certified safety system and not validated against real turbines.

### Added

**Stage 1 — Failure Prediction** (Ozan)

- Physics-flavoured synthetic SCADA generator: 20 turbines, hourly, one year, with a
  power curve, geared drivetrain, first-order thermal lag, lubrication coupling and
  per-turbine baselines. 45% of degradation episodes heal without failing, so a rising
  trend is necessary but not sufficient evidence of failure.
- Validation layer with 13 severity-tagged checks and a machine-readable report.
- `failure_within_48h` target with six documented edge cases.
- 395 leakage-safe features in 7 groups; every temporal operation grouped by turbine and
  past-only, verified by perturbation tests.
- Chronological train/validation/test split with a 48-hour embargo matching the horizon.
- Four compared candidates with a recall floor, a PR-AUC guard rail, cost-based selection
  and a tolerance band so near-ties are broken by recall rather than noise.
- Threshold optimisation (F2 and expected-cost methods) on validation only.
- SHAP explanations with permutation fallbacks, and narratives generated from actual
  attribution values.
- `FailurePredictionService`, failure API router, dashboard page, model card.

**Stage 2 — Turbine Health Monitoring** (Şahin)

- Physical and operational sensor validation rules, plus frozen-sensor, impossible-slew
  and sudden-jump detection.
- Operating regime detection (idle / low / medium / high load, derated, fault).
- 0–100 health score with Healthy / Monitor / Degraded / Critical banding.
- Component-level roll-up: thermal, vibration, lubrication, power efficiency.
- Calibrated sensor-drift detection (EWMA/CUSUM-style) with a persisted detector.
- Health model comparison; Ridge selected for its accuracy on *degraded* turbines rather
  than overall MAE.
- `HealthMonitoringService`, health-monitoring API router (namespaced `/health-monitoring`
  so it never collides with the `/health` liveness probe), fleet health dashboard page,
  model card.

**Stage 3 — Anomaly Detection & Maintenance Decision** (Emir)

- Healthy-reference selection and anomaly feature engineering.
- Isolation Forest, Local Outlier Factor (`novelty=True`) and One-Class SVM compared;
  LOF selected on PR-AUC 0.725 / recall 0.693.
- Anomaly score calibration with warning and alarm thresholds fitted on validation only.
- `MaintenanceService`: configurable weighted unified risk score across all three modules,
  guardrails, coverage reporting and `missing_modules` — a missing module is renormalised,
  never treated as zero risk.
- Deterministic maintenance actions, inspection windows and suspected components.
- Anomaly and maintenance API routers, combined dashboard page, model card.
- Hashed dependency locks (`requirements.txt` / `requirements-dev.txt` via `make lock`),
  `SECURITY.md`, bandit and pip-audit in the dev tooling.

**Platform**

- One FastAPI application exposing 25 endpoints across four namespaces; artifacts load
  lazily so the API starts and reports `degraded` when a model is missing.
- One Streamlit dashboard with three module pages that degrade to actionable messages
  rather than crashing when artifacts are absent.
- 432 tests, ruff lint and format, multi-stage Dockerfile with a non-root user and
  read-only artifact mounts, GitHub Actions CI covering lint, tests, pipeline smoke test
  and Docker build.
- Runtime/artifact library-version compatibility check surfaced in `GET /health`.

### Fixed during the pre-release audit

- `/anomaly/example` and `/maintenance/example` shipped a noiseless sinusoidal window in
  which every sensor was an affine function of one sine wave. The novelty detector
  correctly scored it at the 99.9th percentile and returned `alarm`, which tripped the
  maintenance guardrail and reported a turbine with a 100/100 health score and a 0.2%
  failure probability as **high risk, urgent review**. Examples are now sampled from the
  same synthetic generator the models were trained on, and score `normal` / `low`.
- `CONTRIBUTING.md` told contributors to run `make install` followed by `make test`, but
  the runtime lock does not contain `pytest`, so `make test` failed on a clean machine.
  It now documents `make install-dev`.
- Dashboard used `use_container_width`, removed from Streamlit after 2025-12-31, which
  produced a deprecation warning on every chart and table render. Replaced with `width`.
- README: referred to three modules as "Both", claimed a stale test count, started
  section numbering at 3, had two sections numbered 28, and marked Stage 3
  "Implemented" while Stages 1–2 said "Complete". Headings are now unnumbered, counts are
  accurate, and a recruiter-facing first screen was added.
- Handoff documentation had grown into a single 741-line file mixing all three stages.
  Split into `PROJECT_HANDOFF.md` (integration contract) plus one document per stage, with
  a redirect stub kept at the old path.

### Known limitations

See [`docs/PROJECT_HANDOFF.md`](docs/PROJECT_HANDOFF.md) §7 and each model card. In short:
synthetic data only; advisory output only; unified-risk weights and the failure cost
matrix are policy assumptions rather than validated economics; the health model's target
is a synthetic proxy that would need inspection-derived labels in reality; one simulated
site and one year, so no seasonal generalisation.

[1.0.0]: https://github.com/SEO-Dynamics/Wind-Turbine-Predictive-Maintenance-Platform/releases/tag/v1.0.0
