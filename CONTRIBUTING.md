# Contributing

This is a three-stage collaborative project. All three modules are implemented; changes
must preserve their shared contracts and independent artifact readiness.

| Stage | Module | Author | Branch | Status |
|---|---|---|---|---|
| 1 | Failure Prediction | [@onurozansunger](https://github.com/onurozansunger) | `feature/ozan-failure-prediction` | ✅ Complete |
| 2 | Turbine Health Monitoring | [@SBRKBNL](https://github.com/SBRKBNL) | `feature/turbine-health-monitoring` | ✅ Complete |
| 3 | Anomaly Detection & Decision Support | [@emirsseven](https://github.com/emirsseven) | `feature/emir-anomaly-detection` | ✅ Implemented |

**Before writing any code, read [`docs/OZAN_HANDOFF.md`](docs/OZAN_HANDOFF.md).** It
documents the data, feature, artifact and API contracts your module must respect, the
extension points to build on, and the files that must not change without coordination.
§12 is the Turbine Health Monitoring contract, including how to combine a health
assessment with a failure probability without collapsing the two into one number.

---

## Setup

Requires **Python 3.12+**.

```bash
python -m venv .venv && source .venv/bin/activate
make install
make pipeline-all      # failure -> health -> anomaly on one shared raw fleet
make test
```

`make pipeline-all` runs failure, health and anomaly pipelines in order against **one**
shared raw dataset. Downstream pipelines deliberately do not regenerate the data. Run
them separately with `make pipeline`, `make health-pipeline` or `make anomaly-pipeline`
if you only need one.

If `import wind_turbine_pm` fails after the editable install (happens on some sandboxed
macOS shells), use `export PYTHONPATH=src`.

---

## Branching

- `main` — integration branch. Always green: tests and lint must pass.
- `feature/<name>-<module>` — one branch per module.

Branch from `main`, keep your branch rebased on it, and open a pull request when the
module is complete.

---

## Definition of done

A module is not finished until all of these hold:

- [ ] `make lint` and `make test` pass
- [ ] New temporal features have a **leakage test** (see
      `tests/test_failure_features.py::test_no_future_leakage` and
      `tests/test_health_features.py::test_no_feature_reads_the_future`)
- [ ] Nothing in [`docs/OZAN_HANDOFF.md`](docs/OZAN_HANDOFF.md) §9 ("files you should not
      change unnecessarily") was modified without agreement
- [ ] Model outputs subclass `BasePrediction` and keep `advisory_only: true`
- [ ] Configuration lives under a single top-level namespace for your module, so a
      deep-merge cannot overwrite another module's keys (see `configs/health_model.yaml`)
- [ ] Artifacts are prefixed with your module name and written via a persistence-module
      path accessor, not literal paths
- [ ] Missing artifacts degrade gracefully — a message and the command to fix, never a
      traceback
- [ ] Any detector with a decision threshold states where the threshold came from, and
      **its alert rate on healthy data is measured, not assumed** (see non-negotiable 7)
- [ ] Metrics in documentation come from executed code, never from memory or estimation
- [ ] Your module is documented and added to the roadmap tables

---

## Non-negotiables

These are properties of the platform, not preferences.

**1. No leakage.** Every temporal operation is grouped by `turbine_id` and reads only
current or past rows. Use the primitives in `features/transformers.py` rather than
writing your own rolling logic — they carry this guarantee.

**2. No random splits.** Time-series data with a forward-looking label requires a
chronological split with an embargo. Use `data/splitting.py`.

**3. Never tune on test.** Train fits, validation selects and calibrates, test is scored
once and never feeds back. `optimise_threshold()` raises if handed the test split — keep
that property in anything you add.

**4. Accuracy is not a metric here.** The positive rate is ~2%, so "never fails" scores
~98%. Lead with PR-AUC, recall, F2 and the false-negative rate.

**5. Advisory only.** No model output may trigger an automated action on plant. Every
prediction carries the human-review disclaimer. `advisory_only` is a structural constant,
not a configurable field.

**6. No invented numbers.** Every metric in a README, model card or comment must come
from a file in `artifacts/`. If you have not run it, do not write it.

**7. A detector's alert rate must be measured on healthy data.** Textbook control-chart
constants (CUSUM `h=5`, `k=0.5`; EWMA `L=3`) are derived for *independent* residuals.
Hourly SCADA channels are strongly autocorrelated, so those constants produce far more
alerts than their nominal rate. Measured here before it was fixed: 92% of all
observations took the maximum drift penalty, which is a constant offset on every score
rather than a signal. Set your limits from a healthy reference period and record the
achieved rate — see `health/drift.py::DriftCalibration`.

**8. Two scores that describe the same turbine must not disagree.** If a page shows a
fleet table and a detail panel, both must be derived from the same feature matrix. Where
two entry points can legitimately differ (a short serving window versus full history), say
so in the docstring rather than leaving a reader to discover it.

---

## Code style

Enforced by `ruff` (config in `pyproject.toml`); run `make format` before pushing.

- Type hints and docstrings on public functions
- `pathlib`, never string path concatenation
- Configuration over constants — add a YAML key rather than a literal
- No mutable default arguments
- No broad `except:` and no silently swallowed errors
- Notebooks import from `src/`; production logic never lives in a notebook

Install the pre-commit hooks so this is automatic:

```bash
make install-dev
pre-commit install
```

Runtime and development dependencies are fully and separately locked. Change exact
direct pins in `requirements-runtime.in` or `requirements-dev.in`, then run `make lock`.
Never hand-edit the generated `requirements.txt` or `requirements-dev.txt`. Before
pushing, run both `make lint` and `make security`.

---

## Data and artifacts

**Never commit** generated datasets or model artifacts — they are reproducible from a
seed and are git-ignored. `data/samples/` is the one exception: a small extract that
documents the schema.

Regenerate everything with:

```bash
make pipeline-all
```

---

## Pull requests

Keep them scoped to one module or one concern. In the description, state what changed,
which contracts you touched (if any), and paste the actual `make test` output.

If you changed a shared contract, say so explicitly in the title — those changes affect
every module.
