# Model Card — Anomaly Detection & Maintenance Decision Support

## Model details

| Field | Value |
|---|---|
| Module | Anomaly Detection |
| Version | 1.0.0 |
| Selected algorithm | Local Outlier Factor (`novelty=True`) |
| Alternatives compared | Isolation Forest, One-Class SVM |
| Input | Hourly synthetic SCADA, maximum 72-hour lookback |
| Training reference | Valid, healthy train rows outside controller `fault` and `maintenance` |
| Offline positive label | `degradation_level >= 0.20` |
| Output | Raw novelty margin, empirical `[0,1]` percentile, severity, regime, data quality, associated signal deviations |
| Owner | [@emirsseven](https://github.com/emirsseven) |

The offline label is used only for model comparison and held-out reporting. It is not an
input feature. `degradation_level`, events, failure mode, episode id and maintenance/failure
truth are explicitly excluded from the feature matrix.

## Intended use

The detector identifies SCADA behaviour that differs from a healthy synthetic reference.
It supports a qualified maintenance engineer deciding what to inspect; it is not a failure
diagnosis. Its score is one independent input to the deterministic maintenance policy.

Appropriate uses:

- ranking turbines for human review;
- surfacing unusual sensor relationships and trends;
- supplementing failure probability and condition health;
- testing API, monitoring and maintenance decision-support architecture.

Out-of-scope uses:

- automatic shutdown, derating or controller action;
- automatic work-order creation;
- safety certification;
- deployment to a real fleet without retraining, validation and threshold calibration;
- claiming that a contributing signal caused the anomaly.

## Data and features

The shipped data is entirely synthetic and hourly. Feature families include current sensor
and operating-regime values, 6/24/72-hour rolling statistics, 6/24-hour differences,
24/72-hour trends, physical residuals and past-only robust deviations conditioned by
turbine and operating regime.

All temporal operations are turbine-grouped and current/past-only. A future-row
perturbation test verifies that changing the future cannot alter earlier feature rows.

The healthy training sample is bounded at 20,000 rows and sampled deterministically while
retaining turbine/regime coverage. All three candidate estimators receive the same sample.

## Calibration and thresholds

Each estimator's raw novelty margin is oriented so larger means more anomalous. The margin
is mapped through the empirical CDF of healthy validation scores.

| Severity | Percentile | Target healthy reference alert rate | Measured |
|---|---:|---:|---:|
| Warning | 0.95 | 5% | 5.000% |
| Alarm | 0.99 | 1% | 1.003% |

Ties and finite-sample quantiles mean the measured rate need not equal the target exactly.
CI permits bounded tolerances and fails when calibration drifts outside them.

## Selection and evaluation

Model selection used validation recall at the calibrated warning threshold. PR-AUC, F2 and
lower inference latency were tie-breakers. Test data was scored only after selection.

| Candidate | Validation PR-AUC | Recall | F2 | Selected |
|---|---:|---:|---:|---|
| Isolation Forest | 0.212 | 0.072 | 0.082 | No |
| Local Outlier Factor | 0.725 | 0.693 | 0.685 | Yes |
| One-Class SVM | 0.220 | 0.044 | 0.051 | No |

| Selected LOF | PR-AUC | Recall | Precision | F2 |
|---|---:|---:|---:|---:|
| Validation | 0.725 | 0.693 | 0.657 | 0.685 |
| Test | 0.794 | 0.826 | 0.574 | 0.759 |

All numbers above come from `artifacts/metrics/anomaly_metrics.json` generated on the
current 20-turbine synthetic run.

## Explainability

`contributing_signals` ranks the final observation's absolute distance from healthy
training median in IQR units. This is a reproducible association explanation. It is not
SHAP, a causal attribution or a component diagnosis.

## Unified maintenance policy

The separate policy is not a learned model:

- failure risk: `failure_probability`, weight 0.50;
- anomaly risk: calibrated `anomaly_score`, weight 0.30;
- health risk: `1 - health_score / 100`, weight 0.20.

Missing weights are normalized across ready models, while original evidence weight is
reported as `coverage`. A serious source verdict floors the risk band; controller fault or
two serious signals produces a same-shift engineering review. Data quality and drift lower
decision confidence only.

## Limitations and risks

1. The dataset and evaluation truth are synthetic. Performance does not transfer to real
   turbines by default.
2. `degradation_level` is simulator state, not inspection or CMMS ground truth.
3. Healthy-reference novelty can flag benign operating changes and can miss degradation
   that resembles its training population.
4. The empirical percentile is fleet- and time-period-specific; it must be monitored and
   recalibrated after drift or retraining.
5. Operating regimes are coarse and based on synthetic 2 MW turbine assumptions.
6. Signal deviations are correlations, not causes.
7. Partial-coverage assessments carry less evidence. Consumers must display coverage,
   confidence and missing modules.

## Oversight and monitoring

- Require qualified engineering review for every maintenance recommendation.
- Track healthy warning/alarm rates by turbine and regime.
- Review score and feature distribution drift.
- Revalidate against real inspection and work-order outcomes before operational use.
- Retrain and recalibrate when turbine type, sensor mapping or operating policy changes.
- Keep source assessments visible; never display only the unified score.
