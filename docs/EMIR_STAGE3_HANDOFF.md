# Stage 3 Handoff — Anomaly Detection & Maintenance Decision

**Author:** Emir (@emirdikmen)
**Scope:** healthy-reference novelty detection, unified risk scoring across all three modules, maintenance prioritisation, and final integration / release hardening
**Module version:** 1.0.0
**Status:** ✅ Complete

> Part of the platform handoff set. Start at
> [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) for how the three modules fit together and
> which contracts are shared.

---

The extension points described above are now implemented:

- `anomaly/` owns leakage-safe 72-hour features, IF/LOF/One-Class SVM comparison,
  healthy-validation empirical calibration and prefixed persistence.
- `AnomalyPrediction` subclasses `BasePrediction`; its signal deviations explicitly state
  that they are association evidence, not causal diagnosis.
- `MaintenanceDecisionService` retains failure, health and anomaly outputs, applies
  50/30/20 default weights, normalizes missing weights, and returns both `coverage` and
  `missing_modules`.
- Guardrails floor risk bands without rewriting component scores. Data quality and drift
  affect confidence only.
- Four deterministic advisory actions map to routine cadence, 7 days, 48 hours and same
  shift.
- `/anomaly` and `/maintenance` routers, the root registry and `/health` are mounted.
- The Streamlit “Anomaly & Maintenance” page calls the same services as the API.
- `scripts/run_all_pipelines.py` runs failure → health → anomaly against one raw fleet;
  Docker's pipeline profile uses it.

The anomaly calibration and results are documented in
[`MODEL_CARD_ANOMALY.md`](MODEL_CARD_ANOMALY.md). Generated datasets and artifacts remain
git-ignored and reproducible.
