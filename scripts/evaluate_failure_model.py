#!/usr/bin/env python
"""Evaluate the published model and produce evaluation + explainability artifacts.

Writes confusion-matrix, PR, ROC, SHAP summary/bar and local-explanation figures
to ``artifacts/figures/``, and global importance to ``artifacts/metrics/``.

Usage:
    python scripts/evaluate_failure_model.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from wind_turbine_pm.config import load_config
from wind_turbine_pm.constants import SPLIT_COLUMN, TARGET_COLUMN, TIMESTAMP, TURBINE_ID, SplitName
from wind_turbine_pm.explainability.narratives import build_advisory
from wind_turbine_pm.explainability.shap_explainer import (
    ExplainerUnavailableError,
    build_explainer,
    plot_global_importance,
    plot_local_explanation,
    plot_shap_bar,
    plot_shap_summary,
)
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.models.evaluation import (
    classification_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from wind_turbine_pm.models.persistence import figures_dir, load_bundle, metrics_path
from wind_turbine_pm.models.threshold import map_risk_level
from wind_turbine_pm.services.failure_prediction_service import _load_background
from wind_turbine_pm.utils.io import read_json, write_json, write_table
from wind_turbine_pm.utils.paths import resolve
from wind_turbine_pm.utils.reproducibility import set_global_seed

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from prepare_data import load_prepared  # noqa: E402

logger = get_logger("scripts.evaluate_failure_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explain-samples",
        type=int,
        default=None,
        help="Override explainability.explain_samples.",
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP figures entirely.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation step.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    cfg = load_config()
    configure_from_config(cfg)
    seed = int(cfg.get("random_seed", 42))
    set_global_seed(seed)

    bundle = load_bundle(cfg)
    dataset, features, _ = load_prepared(cfg)
    figures = figures_dir(cfg)

    aligned = features.loc[:, bundle.features]
    results: dict[str, dict[str, float]] = {}

    for split in (SplitName.VALID, SplitName.TEST):
        mask = dataset[SPLIT_COLUMN] == str(split)
        x = aligned.loc[mask]
        y = dataset.loc[mask, TARGET_COLUMN].to_numpy(dtype=int)
        probabilities = bundle.estimator.predict_proba(x)[:, 1]
        metrics = classification_metrics(y, probabilities, bundle.threshold)
        results[str(split)] = metrics

        label = str(split).capitalize()
        plot_confusion_matrix(
            metrics, figures / f"confusion_matrix_{split}.png", f"{label} confusion matrix"
        )
        plot_precision_recall_curve(
            y,
            probabilities,
            figures / f"pr_curve_{split}.png",
            f"{label} precision-recall curve",
            bundle.threshold,
        )
        plot_roc_curve(y, probabilities, figures / f"roc_curve_{split}.png", f"{label} ROC curve")
        logger.info(
            "Evaluated split",
            extra={
                "split": str(split),
                "pr_auc": round(metrics["pr_auc"], 4),
                "recall": round(metrics["recall"], 4),
                "precision": round(metrics["precision"], 4),
                "f2": round(metrics["f2"], 4),
            },
        )

    # --- Explainability ---------------------------------------------------
    explainability: dict[str, object] = {"enabled": False}
    if not args.skip_shap and bool(cfg.get("explainability.enabled", True)):
        explainability = _run_explainability(cfg, bundle, dataset, aligned, args, seed)

    # --- Merge into the metrics document ---------------------------------
    document = read_json(metrics_path(cfg), hint="python scripts/train_failure_model.py")
    document["evaluation"] = results
    document["explainability"] = explainability
    document["figures"] = sorted(path.name for path in figures.glob("*.png"))
    write_json(document, metrics_path(cfg))

    print("\n=== Final evaluation ===")  # noqa: T201
    for split, metrics in results.items():
        print(  # noqa: T201
            f"{split:<6} "
            + " ".join(
                f"{key}={metrics[key]:.4f}"
                for key in (
                    "pr_auc",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "false_negative_rate",
                )
            )
        )
    return 0


def _run_explainability(cfg, bundle, dataset, aligned, args, seed) -> dict[str, object]:
    """Produce SHAP/permutation artifacts and a worked local explanation."""
    test_mask = dataset[SPLIT_COLUMN] == str(SplitName.TEST)
    test_x = aligned.loc[test_mask]
    test_y = dataset.loc[test_mask, TARGET_COLUMN].to_numpy(dtype=int)

    background = _load_background(cfg, tuple(bundle.features))
    explainer = build_explainer(bundle.estimator, bundle.features, background, cfg)
    logger.info("Explainer ready", extra={"method": explainer.method})

    n_samples = int(args.explain_samples or cfg.get("explainability.explain_samples", 1000))
    if explainer.method == "shap_permutation":
        # The permutation explainer costs 2 * n_features + 1 model evaluations
        # per row, so the full sample would take hours. Tree models are exact
        # and cheap and keep the full sample.
        capped = int(cfg.get("explainability.explain_samples_permutation", 150))
        if capped < n_samples:
            logger.info(
                "Reducing the explanation sample for the permutation explainer",
                extra={"from": n_samples, "to": capped, "n_features": len(bundle.features)},
            )
            n_samples = capped
    sample_index = (
        test_x.sample(min(n_samples, len(test_x)), random_state=seed).index
        if len(test_x) > n_samples
        else test_x.index
    )
    sample_x = test_x.loc[sample_index]

    figures = figures_dir(cfg)
    try:
        importance = explainer.global_importance(
            sample_x,
            y=test_y[[test_x.index.get_loc(i) for i in sample_index]],
            repeats=int(cfg.get("explainability.permutation_repeats", 5)),
        )
    except ExplainerUnavailableError as exc:
        logger.error("Global importance unavailable", extra={"error": str(exc)})
        return {"enabled": True, "method": explainer.method, "error": str(exc)}

    write_table(
        importance,
        resolve(str(cfg.require("paths.artifacts_metrics"))) / "global_feature_importance.csv",
    )
    plot_global_importance(
        importance,
        figures / "global_feature_importance.png",
        title=f"Global feature importance ({explainer.method})",
    )
    plot_shap_summary(explainer, sample_x, figures / "shap_summary.png")
    plot_shap_bar(explainer, sample_x, figures / "shap_bar.png")

    # --- Worked local explanation for the highest-risk test observation ---
    probabilities = bundle.estimator.predict_proba(test_x)[:, 1]
    highest = int(np.argmax(probabilities))
    row = test_x.iloc[[highest]]
    probability = float(probabilities[highest])
    attributions = explainer.local_attributions(
        row, top_k=int(cfg.get("explainability.top_k_factors", 5))
    )[0]

    low_max = float(cfg.require("risk_levels.low_max"))
    medium_max = float(cfg.require("risk_levels.medium_max"))
    risk_level = map_risk_level(probability, low_max, medium_max)
    advisory = build_advisory(attributions, probability, risk_level, explainer.method)

    meta = dataset.loc[test_mask].iloc[highest]
    plot_local_explanation(
        attributions,
        figures / "local_explanation_high_risk.png",
        f"Local explanation - {meta[TURBINE_ID]} @ {pd.Timestamp(meta[TIMESTAMP])} "
        f"(p={probability:.3f}, {risk_level} risk)",
    )

    logger.info("Local explanation: %s", advisory.explanation)
    return {
        "enabled": True,
        "method": explainer.method,
        "n_explained": int(len(sample_x)),
        "top_global_features": importance.head(15).to_dict(orient="records"),
        "example_high_risk": {
            "turbine_id": str(meta[TURBINE_ID]),
            "timestamp": str(pd.Timestamp(meta[TIMESTAMP])),
            "failure_probability": round(probability, 6),
            "actual_label": int(meta[TARGET_COLUMN]),
            "risk_level": risk_level,
            "top_risk_factors": [
                {
                    "feature": attribution.feature,
                    "impact": round(attribution.impact, 6),
                    "direction": attribution.direction,
                    "value": None
                    if attribution.value is None or not np.isfinite(attribution.value)
                    else round(attribution.value, 4),
                }
                for attribution in attributions
            ],
            "explanation": advisory.explanation,
            "recommendation": advisory.recommendation,
        },
    }


if __name__ == "__main__":
    sys.exit(main())
