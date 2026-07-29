# Model Card — Turbine Health Monitoring

**Model name:** `health_score`
**Version:** 1.0.0
**Module:** Turbine Health Monitoring (Stage 2 of 3)
**Algorithm:** `Ridge` (scikit-learn), median-imputed and standardised in a pipeline
**Task:** Regression — what condition is this turbine in, on a 0-100 scale?
**Status:** Portfolio / demonstration project trained on **synthetic data**
**License:** MIT

> ## ⚠️ Critical statement of scope
>
> **This model must not autonomously stop, start, derate, operate, or schedule
> maintenance for industrial equipment, under any circumstances.**
>
> It produces advisory decision support for qualified human maintenance staff. It is not a
> certified industrial safety system, not a substitute for engineering inspection, and not
> a validated real-world condition-monitoring solution. It has never been evaluated
> against measured turbine condition on operating machines.

The companion card for the Failure Prediction Module is [`../MODEL_CARD.md`](../MODEL_CARD.md).
The two models answer different questions and must not be substituted for one another.

---

## 1. Model overview

Given a recent window of SCADA sensor history for one turbine, the module outputs:

| Output | Description |
|---|---|
| `health_score` | Published 0-100 condition score (100 = as-new) |
| `raw_health_score` | The model's own estimate, before the drift deduction |
| `drift_penalty` | Points deducted for detected sensor drift, capped at 15 |
| `health_class` | `healthy` / `monitor` / `degraded` / `critical` |
| `operating_regime` | Operating point the assessment was made in |
| `data_quality` | Share of the assessed window that passed the sensor validity checks |
| `component_health` | Per-component score and the evidence behind it, worst first |
| `drift_signals` | Per-channel drift verdicts; empty means none fired |
| `rule_violations` | Envelope and validity findings, with the limit that was crossed |
| `top_factors` | Largest condition deviations, each traceable to a measured value |
| `explanation` | Narrative derived only from the values above |
| `recommendation` | Advisory message carrying the standing disclaimer |

The published score is always `max(raw_health_score − drift_penalty, 0)`, and the output
contract enforces it, so the deduction can never be silently absorbed.

### What it is not

It is not a failure predictor. It carries no horizon and makes no claim about *when*
anything will fail. A turbine can be Healthy with a high failure probability (a sudden fault
signature on an otherwise sound machine) or Degraded with a low one (worn but not near
failure). Both are operationally meaningful; see
[`OZAN_HANDOFF.md`](OZAN_HANDOFF.md) §12.3 before combining the two.

---

## 2. Intended use

**Intended users:** wind-farm reliability engineers, O&M planners and condition-monitoring
analysts, working with the raw sensor trends alongside the output.

**Intended uses:**

- Ranking a fleet by condition to inform inspection scheduling
- Narrowing an investigation to a named component before dispatching an engineer
- Reviewing a turbine's condition trend across weeks
- Flagging channels whose calibration may need checking
- Demonstrating a condition-monitoring pipeline end to end

**Every use requires a human in the loop.** The output is one input to a maintenance
decision, never the decision.

---

## 3. Out-of-scope use

- **Automated control or dispatch of any kind.**
- **Safety functions.** Not a protection system; not certified to any functional-safety
  standard.
- **Warranty, insurance or contractual determinations** about machine condition.
- **Replacing physical inspection**, oil analysis, borescope inspection or certified
  condition-monitoring hardware.
- **Predicting time to failure.** No horizon is modelled.
- **Transfer to a different turbine type** without re-deriving the envelope limits, the
  regime boundaries and the class bands, and retraining.
- **Reading component scores as a physical diagnosis.** They localise evidence; they do not
  identify a fault mode.

---

## 4. Training data

### Synthetic-data disclosure

The model is trained on data from `scripts/generate_synthetic_data.py`, the same generator
the Failure Prediction Module uses. It is physics-flavoured, not physics-validated: it
contains the structure that was programmed into it, including the degradation process the
health target is derived from. **No metric in this card is evidence of real-world
performance.**

### Dataset characteristics

| Property | Value |
|---|---|
| Turbines | 20 |
| Sampling interval | 1 hour |
| Coverage | 12 months |
| Rows used for modelling | 173,558 |
| Train / validation / test rows | 104,207 / 33,779 / 33,675 |
| Observations below the Monitor boundary | 7.59% |
| Sensor validity across the dataset | 96.02% |

### Operating-regime distribution

| Regime | Share |
|---|---|
| `medium_load` | 48.5% |
| `low_load` | 23.5% |
| `idle` | 16.9% |
| `high_load` | 9.9% |
| `curtailed` | 1.2% |
| `offline` | 0.0% |

`offline` is 0% by construction: rows the controller reports as `maintenance` or `fault` are
excluded from the labelled set, because at that point the machine's condition is already
known to the operator.

### Known data limitations

- Degradation is monotonic within an episode and reset by repair; real wear is noisier.
- No injected **sensor calibration drift** — see §9 for why this matters.
- One site, one year, one turbine class. No seasonal or cross-site generalisation.
- The label and the features come from the same simulator, so their relationship is
  cleaner than reality.

---

## 5. Target

```
health_score_target = 100 × (1 − degradation_level)
```

`degradation_level ∈ [0, 1]` comes from the synthetic generator. Rows whose
`operational_status` is `maintenance` or `fault` are marked ineligible for labelling (but
retained, so feature windows are not truncated).

**On a real fleet this label must come from a source independent of the SCADA channels the
features are built from** — inspection reports, oil-analysis results, borescope findings or
a certified condition-monitoring index. Deriving it from the same signals that produce the
features would make the model learn its own input. `build_health_target` raises with that
instruction rather than inventing a target, and `degradation_level` is listed in
`health.features.exclude_columns` with a test asserting it never reaches the matrix.

---

## 6. Feature groups

239 features in 8 groups, from one entry point (`build_health_features`) used identically by
training and serving.

| Group | n | Content |
|---|---|---|
| `condition` | 15 | Current condition-channel values, one-hot operating regime |
| `rolling` | 81 | Trailing mean/std/max over 6/24/72 h |
| `trend` | 36 | Trailing OLS slopes, differences, deviation from a 168 h baseline |
| `signal_shape` | 21 | RMS, crest factor, excess kurtosis, skewness, peak-to-peak, high-frequency ratio, zero-crossing rate |
| `regime_relative` | 12 | Deviation and robust z-score against the turbine's own producing-regime baseline |
| `rule` | 36 | Normalised distance into the operating envelope, recent violation rates |
| `drift` | 25 | CUSUM arms, trailing signal counts, EWMA, standardised residuals |
| `physical` | 13 | Temperature rise above ambient, power-curve ratio, vibration per load, lubrication interactions |

Every temporal operation is grouped by `turbine_id` and reads current-or-past rows only.
Leakage is asserted empirically: perturbing the final row must not change any earlier row's
features, and one turbine's history must not affect another's.

---

## 7. Validation strategy

Chronological split with a 48-hour embargo at each boundary, identical to the Failure
Prediction Module and using the same code. Train fits, validation ranks and selects, test is
scored once and never fed back.

The drift thresholds and the multivariate detector are fitted on the **training split
only**, so no threshold can be derived from data it will later judge.

---

## 8. Metrics

### Selected model — test split

| Metric | Value |
|---|---|
| MAE | 6.4577 |
| **MAE on degraded (< 60)** | **6.2673** |
| RMSE | 7.1611 |
| Spearman | 0.5991 |
| Class agreement | 0.9725 |
| Median absolute error | see `artifacts/metrics/health_metrics.json` |

### Error by true health band (test)

| Band | n | MAE | Class agreement | Optimistic rate |
|---|---|---|---|---|
| `healthy` | 29,835 | 6.53 | 0.988 | 0.000 |
| `monitor` | 1,151 | 5.08 | 0.756 | 0.043 |
| `degraded` | 1,147 | 5.44 | 0.782 | 0.041 |
| `critical` | 1,542 | 6.88 | 0.972 | 0.029 |

`optimistic_rate` is the share placed in a **healthier** band than the truth — the dangerous
direction. It is reported per band because an aggregate figure hides it.

### Candidate comparison (validation)

| Model | MAE | MAE degraded | RMSE | Spearman | Class agreement |
|---|---|---|---|---|---|
| **Ridge** ✅ | 4.2115 | **6.2282** | 5.2948 | 0.6460 | 0.9654 |
| Histogram gradient boosting | **2.0650** | 8.0044 | **4.3247** | 0.6276 | 0.9524 |
| Random forest | 2.8870 | 13.4365 | 6.2527 | 0.5946 | 0.9356 |
| Mean baseline | 11.5791 | 58.6659 | 19.3206 | n/a | 0.8677 |

### Selection rationale

Ridge was selected on `mae_degraded`, **not** on overall MAE. Histogram gradient boosting is
clearly better overall (2.07 versus 4.21) and clearly worse on the observations that matter
(8.00 versus 6.23). Since 88% of observations are healthy, overall MAE is dominated by that
majority and a model can win it while being least reliable exactly where the score has to be
trusted.

Two guard rails also apply, and reject a candidate outright:

- rank correlation with true condition below **0.50** — a score that does not order turbines
  correctly cannot prioritise maintenance, whatever its error;
- overall MAE worse than **85%** of the mean-prediction baseline.

The full rationale string is persisted in `artifacts/metadata/health_model_metadata.json`.

---

## 9. Sensor drift detection

Three detectors run per configured channel: CUSUM (small persistent shifts), EWMA (faster on
moderate steps), and an Isolation Forest over all channels jointly (broken relationships
between channels).

### The calibration problem, and why it is documented here

The textbook CUSUM constants (`k=0.5`, `h=5`) and the asymptotic EWMA control limit assume
**statistically independent** residuals. Hourly SCADA channels are strongly autocorrelated,
so those constants have no defined false-alarm rate on this data. Measured before and after
the fix:

| | Before | After |
|---|---|---|
| Peak CUSUM statistic | ~1,900 σ (unbounded) | bounded by `h` + one residual |
| Rows at the maximum drift penalty | **92%** | 2.1% |
| Rows with no penalty | 3.8% | 68.5% |
| Flag rate, healthy vs degraded | 24% vs 17% *(inverted)* | 28% vs 66% *(2.37×)* |

Three changes were required:

1. **Page's restart** — an arm that reaches its limit is reported and reset, so the statistic
   stays bounded and persistence becomes countable.
2. **Severity from persistence**, not magnitude, since the statistic is now bounded.
3. **Empirical calibration** of every threshold against the fleet's own healthy population
   on the training split, with a chosen false-alarm rate of 5% (warning) and 1% (alarm) per
   detector per channel.

The distance between the calibrated and textbook limits is the size of the problem: EWMA
limits came out at 1.67–9.60 against a theoretical 0.688, and the Isolation Forest threshold
at 0.915 against a configured 0.62. The fitted thresholds are published in the artifact and
on `GET /health-monitoring/model-info`.

### The penalty

Capped at **15 points**, so drift alone can never manufacture a Critical classification.
Drift means the *measurement* or the baseline is suspect, which justifies lowering confidence
in the score — not asserting that the machine has failed. One deduction per channel at its
worst severity: CUSUM and EWMA firing on the same channel is one drifting sensor, not two.

### The honest limitation

**This dataset contains no injected calibration drift**, so the detector has nothing genuine
to find here, and the crossing counts carry no information about machine degradation — which
is correct, since drift targets the instrument rather than the machine. Its thresholds are
calibrated only to control the false-alarm rate on healthy machines. **Its ability to catch a
real drifting instrument is untested and must not be claimed.**

---

## 10. Health classification

| Class | Score | Operational meaning |
|---|---|---|
| `healthy` | ≥ 80 | No action beyond routine monitoring |
| `monitor` | 60–80 | Watch the trend; review at the next planned visit |
| `degraded` | 40–60 | Schedule an inspection; the machine is off-baseline |
| `critical` | < 40 | Prioritise inspection before continued operation |

The boundaries are an **operational** choice rather than a property of the model: they encode
how much degradation justifies an inspection, which is the operator's decision. They live in
`configs/health_model.yaml` and can be re-tuned without retraining.

Optional adaptive bands shift with the fleet's own score distribution, so a fleet-wide change
(a firmware update, a season) does not silently move every turbine into Monitor. **Off by
default**, because a moving boundary is harder to explain to an operator than a fixed one,
and clamped by `max_shift` so the bands can never invert.

---

## 11. Explainability

Attribution is **rule- and baseline-driven, not model-driven**. Every reported factor and
every component deduction traces to one of two checkable statements:

- "this channel is *N* of the way from its warning limit to its alarm limit", or
- "this channel is *N* sigma from this turbine's own baseline for the current regime".

Both can be verified against the raw trend, which the dashboard plots directly beneath the
assessment together with the exact limits applied and their provenance.

This was chosen over a SHAP-style decomposition deliberately. A regression trained on a
fleet-level target cannot be decomposed into per-component contributions without inventing an
attribution the training objective never constrained. A number an engineer can check beats a
number that merely looks principled.

**Consequence to be aware of:** a component score can disagree with the overall score, since
one is this turbine's own rule evidence and the other is a fleet-trained estimate. The
advisory text names the component explicitly even when the overall class is Healthy, rather
than letting an operator read "no action required" while a subsystem sits in its alarm band.

Narratives never fabricate. With no evidence they say so, rather than producing plausible
filler.

---

## 12. Limitations

1. **Synthetic data.** No result here is evidence of real-world performance.
2. **The target is simulator-derived.** See §5; on a real fleet it must be independent of the
   feature channels.
3. **The drift detector is unvalidated against real drift.** See §9.
4. **Rank correlation is moderate** (Spearman 0.60 on test). Sufficient to inform
   prioritisation, insufficient to be a strict priority queue on its own.
5. **Error is worst on healthy machines** (MAE 6.53 versus 5.08 on Monitor): the model is
   mildly pessimistic about sound turbines. Class agreement on that band is 0.988, so the
   decision is rarely wrong even where the number is.
6. **Ridge is linear over 239 correlated features** and reported an ill-conditioned design
   matrix during fitting. The L2 penalty handles it, but coefficients must not be read as
   independent effect sizes.
7. **Envelope limits are advisory defaults** for a 2 MW geared machine, from ISO vibration
   bands and typical O&M alarm practice. They must be re-derived per turbine type and alarm
   history.
8. **Class boundaries are unvalidated** against real inspection outcomes.
9. **Assessment needs ≥72 hours of history**, so a newly commissioned turbine cannot be
   scored until it has that. Expanding baselines keep improving well beyond it, so a short
   window is less informed rather than wrong — the two entry points can legitimately return
   different scores for the same timestamp, and this is documented rather than hidden.
10. **No uncertainty quantification.** A score of 62 has no confidence interval.
11. **Feature build is ~90 s for 174k rows.** Fine in batch and fast for a single window, but
    do not call it per row.

---

## 13. Risks

| Risk | Consequence | Mitigation in place |
|---|---|---|
| Over-trust in the score | A degraded machine dismissed because the number looks fine | Component roll-up surfaces disagreement; advisory names it explicitly; raw evidence plotted alongside |
| Drift flags misread as machine condition | Unnecessary teardown of a healthy machine | Drift is reported separately, capped, and described as an instrument concern in the advisory text |
| Uncalibrated thresholds on new data | Alert flood, then alert fatigue, then a real signal ignored | Thresholds calibrated per fleet from the healthy population; achieved rates recorded in the artifact |
| Class boundary treated as physics | Inspection scheduling anchored to an arbitrary number | Boundaries documented as an operational choice and kept in configuration |
| Synthetic metrics quoted as validated | Unwarranted confidence in a real deployment | Every metric surface carries a synthetic disclaimer; the API response includes one |
| Silent artifact/runtime mismatch | Scores produced by a differently-versioned estimator | Library versions recorded and checked at load; warning surfaced on `/health` |

---

## 14. Human oversight

Required at every point. The module produces text that names the evidence and the limit that
was crossed so that a reviewer can disagree with it on specifics rather than in general.
Every recommendation carries:

> Advisory output only - requires review by a qualified maintenance engineer and does not
> replace inspection or certified safety systems.

---

## 15. Monitoring recommendations

If this were deployed:

- **Recalibrate the drift thresholds** whenever the fleet, the sensor set or the sampling
  interval changes. They are fleet-specific by construction.
- **Track the achieved alert rate** against the 5%/1% targets. Divergence means the healthy
  reference period no longer represents the fleet.
- **Track the score distribution per turbine and fleet-wide.** A fleet-wide shift is a data
  or instrumentation change far more often than a fleet-wide degradation.
- **Track the class transition rate.** Churn across a boundary means the boundary is in the
  wrong place, not that the machines are oscillating.
- **Audit assessments against inspection outcomes** and use those outcomes to re-derive the
  class bands — that is the only way the bands stop being a guess.
- **Watch `data_quality`.** A falling value is an instrumentation problem, and every score
  computed from that window is correspondingly less trustworthy.

---

## 16. Retraining recommendations

Retrain when any of these hold:

- A real ground-truth source becomes available (**highest priority** — it replaces the
  simulator-derived label).
- Turbines are added, replaced or retrofitted.
- The sensor set or sampling interval changes.
- The score distribution drifts materially against the training period.
- Twelve months elapse.

Retrain the whole bundle together: the estimator, the multivariate detector and the drift
calibration are fitted on the same training split and must stay consistent.

---

## 17. Real-world validation requirements

Before any operational use:

1. **Independent ground truth.** Inspection reports, oil analysis, borescope findings or a
   certified CMS index — not derived from the SCADA feature channels.
2. **Re-derived envelope limits** from the operator's turbine type, control strategy and
   historical alarm log.
3. **Re-derived class bands** against real inspection outcomes and their costs.
4. **Recalibrated drift thresholds** on a genuinely in-control period of the real fleet.
5. **Validated drift detection** against known calibration events — the one claim this card
   currently cannot make.
6. **Prospective evaluation** over at least one full seasonal cycle.
7. **A documented human review workflow**, with the authority to override.
8. **Sign-off from the operating organisation's reliability engineering function.**

---

## 18. Reproducibility

```bash
make pipeline-all       # both modules, one shared dataset
# or just this module, once the raw data exists:
make health-pipeline
```

Everything derives from `random_seed: 42`. Library versions are recorded in the artifact and
checked against the runtime at load, because a serialised scikit-learn estimator is only
guaranteed to load under the version that wrote it.

| Artifact | Path |
|---|---|
| Estimator | `artifacts/models/health_model.joblib` |
| Drift detector | `artifacts/models/health_drift_detector.joblib` |
| Metadata | `artifacts/metadata/health_model_metadata.json` |
| Metrics | `artifacts/metrics/health_metrics.json` |
| Candidate comparison | `artifacts/metrics/health_model_comparison.csv` |
| Error by band | `artifacts/metrics/health_error_by_band.csv` |
| Data quality | `artifacts/metrics/health_data_quality.json` |
| Figures | `artifacts/figures/health_*.png` |

---

## 19. Contact and governance

**Module owner:** Turbine Health Monitoring Module (Stage 2) —
[@SBRKBNL](https://github.com/SBRKBNL).
**Repository:** https://github.com/SEO-Dynamics/Wind-Turbine-Predictive-Maintenance-Platform
**Handoff documentation:** [`OZAN_HANDOFF.md`](OZAN_HANDOFF.md) — §12 is this module's contract

Report issues with the model, its documentation or its outputs through the repository's issue
tracker. Concerns about unsafe or out-of-scope use should be escalated to the module owner and
the operating organisation's reliability engineering function.

---

*Last updated: 2026-07-30 · Model version 1.0.0 · Trained on synthetic data*
