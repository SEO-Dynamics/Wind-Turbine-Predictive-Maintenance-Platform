# Result figures

Committed copies of selected plots from `artifacts/figures/`, so the results are visible
on GitHub without cloning and running the pipeline.

**These are generated artifacts, not hand-made.** They correspond to the model described
in [`MODEL_CARD.md`](../../MODEL_CARD.md) and are produced on **synthetic data**.

Refresh them after retraining:

```bash
make docs-figures
```

The full set (including validation-split curves, the SHAP bar plot, anomaly
comparison/calibration and the maintenance-policy figure) lives in
`artifacts/figures/` after running `make pipeline-all`.
