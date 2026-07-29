"""SHAP-based explanations with a permutation-importance fallback.

The explainer is constructed once and reused, because building a SHAP
background set is the expensive part.  For tree ensembles ``shap.TreeExplainer``
is exact and fast; for anything else (including a calibrated wrapper around a
tree model, where the calibrator sits between SHAP and the trees) the code falls
back to ``shap.PermutationExplainer`` on a small background sample, and finally
to scikit-learn's permutation importance if SHAP is unavailable altogether.

Local attributions are always returned in the *log-odds* space of the underlying
model, which is what makes "this feature added +0.27 to the risk" meaningful and
additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_pm.config import Config
from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.utils.paths import ensure_parent

logger = get_logger(__name__)


class ExplainerUnavailableError(RuntimeError):
    """Raised when no explanation method could be constructed."""


@dataclass
class LocalAttribution:
    """One feature's contribution to a single prediction."""

    feature: str
    impact: float
    value: float | None

    @property
    def direction(self) -> str:
        """``'increases_risk'`` when the contribution is positive."""
        return "increases_risk" if self.impact >= 0 else "decreases_risk"


def _unwrap_pipeline(estimator: Any) -> Any:
    """Strip calibration and freezing wrappers to reach the underlying pipeline.

    Args:
        estimator: Possibly wrapped fitted estimator.

    Returns:
        The innermost pipeline or estimator.
    """
    from sklearn.calibration import CalibratedClassifierCV

    inner = estimator
    for _ in range(4):  # bounded: calibrator -> frozen -> pipeline
        if isinstance(inner, CalibratedClassifierCV):
            inner = getattr(inner, "estimator", inner)
            continue
        if type(inner).__name__ == "FrozenEstimator":
            inner = getattr(inner, "estimator", inner)
            continue
        break
    return inner


def _unwrap_estimator(estimator: Any) -> tuple[Any, Any]:
    """Return ``(pipeline_or_model, final_tree_model_or_None)``.

    Digs through calibration, freezing and pipeline wrappers to find a tree
    ensemble that ``shap.TreeExplainer`` can consume directly.

    Args:
        estimator: The fitted estimator, possibly wrapped.

    Returns:
        The unwrapped estimator and, when present, its tree model.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.pipeline import Pipeline

    inner = _unwrap_pipeline(estimator)
    final = inner.steps[-1][1] if isinstance(inner, Pipeline) else inner
    tree_model = (
        final
        if isinstance(final, RandomForestClassifier | HistGradientBoostingClassifier)
        else None
    )
    return inner, tree_model


def _transform_features(estimator: Any, x: pd.DataFrame) -> pd.DataFrame:
    """Apply a pipeline's preprocessing steps without its final estimator.

    Args:
        estimator: The fitted estimator, possibly wrapped.
        x: Raw feature frame.

    Returns:
        The transformed frame, or ``x`` when there is nothing to apply.
    """
    from sklearn.pipeline import Pipeline

    inner = _unwrap_pipeline(estimator)
    if not isinstance(inner, Pipeline) or len(inner.steps) == 1:
        return x
    transformed = inner[:-1].transform(x)
    return pd.DataFrame(np.asarray(transformed), columns=x.columns, index=x.index)


class FailureExplainer:
    """Explains failure-probability predictions for a fitted model bundle."""

    def __init__(
        self,
        estimator: Any,
        feature_names: list[str],
        background: pd.DataFrame | None = None,
        max_background: int = 200,
        seed: int = 42,
    ) -> None:
        """Build an explainer for ``estimator``.

        Args:
            estimator: The fitted pipeline (possibly calibrated).
            feature_names: Feature order the estimator expects.
            background: Rows sampled from the training split, used as the SHAP
                background distribution. Required for the permutation path.
            max_background: Cap on background rows, for runtime.
            seed: Seed for background sampling.

        Raises:
            ExplainerUnavailableError: If SHAP is installed but no explainer
                could be constructed for this estimator.
        """
        self.estimator = estimator
        self.feature_names = list(feature_names)
        self.method = "unavailable"
        self._explainer: Any = None
        self._background = None

        if background is not None and len(background) > max_background:
            background = background.sample(max_background, random_state=seed)
        self._background = background

        try:
            import shap
        except ImportError:
            logger.warning("SHAP is not installed; falling back to permutation importance")
            self.method = "permutation_importance"
            return

        _, tree_model = _unwrap_estimator(estimator)
        if tree_model is not None:
            try:
                self._explainer = shap.TreeExplainer(tree_model)
                self.method = "shap_tree"
                logger.info("Built SHAP TreeExplainer", extra={"model": type(tree_model).__name__})
                return
            except (TypeError, ValueError, NotImplementedError) as exc:
                logger.warning("TreeExplainer unavailable, falling back", extra={"error": str(exc)})

        if background is None or background.empty:
            self.method = "permutation_importance"
            logger.warning("No background data supplied; using permutation importance")
            return

        def score(matrix: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(matrix, columns=self.feature_names)
            probabilities = np.clip(estimator.predict_proba(frame)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(probabilities / (1 - probabilities))

        self._explainer = shap.PermutationExplainer(
            score, background[self.feature_names].to_numpy()
        )
        self.method = "shap_permutation"
        logger.info("Built SHAP PermutationExplainer")

    @property
    def uses_shap(self) -> bool:
        """Whether a genuine SHAP explainer backs this instance."""
        return self.method.startswith("shap")

    def shap_values(self, x: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the positive class.

        Args:
            x: Feature frame in the model's feature order.

        Returns:
            An array of shape ``(n_rows, n_features)``.

        Raises:
            ExplainerUnavailableError: If no SHAP explainer is available.
        """
        if not self.uses_shap or self._explainer is None:
            raise ExplainerUnavailableError(
                f"No SHAP explainer available (method={self.method}); use global_importance instead"
            )
        if self.method == "shap_tree":
            values = self._explainer.shap_values(
                _transform_features(self.estimator, x), check_additivity=False
            )
        else:
            # PermutationExplainer needs at least one forward and one reverse
            # pass over every feature, i.e. 2 * n_features + 1 model
            # evaluations per row. Anything less raises rather than degrading.
            values = self._explainer(
                x[self.feature_names].to_numpy(),
                max_evals=2 * len(self.feature_names) + 1,
                silent=True,
            ).values

        values = np.asarray(values)
        if values.ndim == 3:
            # (n_rows, n_features, n_classes) -> positive class
            values = values[:, :, -1]
        return values

    def local_attributions(self, x: pd.DataFrame, top_k: int = 5) -> list[list[LocalAttribution]]:
        """Compute per-row top contributions.

        Args:
            x: Feature frame in the model's feature order.
            top_k: Number of factors to return per row.

        Returns:
            One ranked list of :class:`LocalAttribution` per input row.
        """
        if self.uses_shap:
            values = self.shap_values(x)
        else:
            # Fallback: weight the global importance by how unusual each value
            # is relative to the background, giving a per-row ranking that is
            # honest about its coarser provenance.
            importance = self.global_importance(x).set_index("feature")["importance"]
            reference = self._background if self._background is not None else x
            centre = reference[self.feature_names].median()
            scale = reference[self.feature_names].std().replace(0.0, 1.0).fillna(1.0)
            deviation = (x[self.feature_names] - centre) / scale
            values = (deviation * importance.reindex(self.feature_names).to_numpy()).to_numpy()

        results: list[list[LocalAttribution]] = []
        for position in range(len(x)):
            row = values[position]
            order = np.argsort(-np.abs(row))[:top_k]
            results.append(
                [
                    LocalAttribution(
                        feature=self.feature_names[index],
                        impact=float(row[index]),
                        value=float(x.iloc[position, x.columns.get_loc(self.feature_names[index])]),
                    )
                    for index in order
                ]
            )
        return results

    def global_importance(
        self, x: pd.DataFrame, y: np.ndarray | None = None, repeats: int = 5
    ) -> pd.DataFrame:
        """Compute global feature importance.

        Uses mean absolute SHAP value when SHAP is available, otherwise
        scikit-learn permutation importance (which requires labels).

        Args:
            x: Feature frame.
            y: Labels, required only for the permutation fallback.
            repeats: Permutation repeats for the fallback.

        Returns:
            A dataframe of ``feature`` / ``importance``, sorted descending.

        Raises:
            ExplainerUnavailableError: If the fallback is needed but no labels
                were supplied.
        """
        if self.uses_shap:
            values = np.abs(self.shap_values(x)).mean(axis=0)
            frame = pd.DataFrame({"feature": self.feature_names, "importance": values})
        else:
            if y is None:
                raise ExplainerUnavailableError(
                    "Permutation importance requires labels; pass y= to global_importance"
                )
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                self.estimator,
                x,
                y,
                n_repeats=repeats,
                random_state=42,
                scoring="average_precision",
                n_jobs=1,
            )
            frame = pd.DataFrame(
                {"feature": self.feature_names, "importance": result.importances_mean}
            )
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_global_importance(
    importance: pd.DataFrame, path: str | Path, top_n: int = 20, title: str = ""
) -> Path:
    """Render a horizontal bar chart of global feature importance.

    Args:
        importance: Output of :meth:`FailureExplainer.global_importance`.
        path: Destination image path.
        top_n: Number of features to show.
        title: Figure title.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    subset = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.32 * len(subset))))
    ax.barh(subset["feature"], subset["importance"], color="#2b6cb0")
    ax.set_xlabel("Mean |contribution|")
    ax.set_title(title or f"Top {top_n} features by global importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def plot_shap_summary(
    explainer: FailureExplainer, x: pd.DataFrame, path: str | Path, max_display: int = 20
) -> Path | None:
    """Render the SHAP beeswarm summary plot.

    Args:
        explainer: A SHAP-backed explainer.
        x: Rows to explain.
        path: Destination image path.
        max_display: Number of features to display.

    Returns:
        The written path, or ``None`` when SHAP is unavailable.
    """
    if not explainer.uses_shap:
        logger.warning("Skipping SHAP summary plot: no SHAP explainer available")
        return None

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import shap

    values = explainer.shap_values(x)
    fig = plt.figure(figsize=(9.0, 7.0))
    shap.summary_plot(
        values, x[explainer.feature_names], max_display=max_display, show=False, plot_size=None
    )
    plt.title("SHAP summary (test split)")
    plt.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_shap_bar(
    explainer: FailureExplainer, x: pd.DataFrame, path: str | Path, max_display: int = 20
) -> Path | None:
    """Render the SHAP mean-absolute bar plot.

    Args:
        explainer: A SHAP-backed explainer.
        x: Rows to explain.
        path: Destination image path.
        max_display: Number of features to display.

    Returns:
        The written path, or ``None`` when SHAP is unavailable.
    """
    if not explainer.uses_shap:
        return None

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import shap

    values = explainer.shap_values(x)
    fig = plt.figure(figsize=(9.0, 7.0))
    shap.summary_plot(
        values,
        x[explainer.feature_names],
        plot_type="bar",
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("Mean |SHAP value| (test split)")
    plt.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_local_explanation(
    attributions: list[LocalAttribution], path: str | Path, title: str
) -> Path:
    """Render a signed bar chart for one observation's top factors.

    Args:
        attributions: Ranked local attributions.
        path: Destination image path.
        title: Figure title.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    ordered = sorted(attributions, key=lambda a: a.impact)
    colours = ["#c53030" if a.impact >= 0 else "#2f855a" for a in ordered]
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.46 * len(ordered))))
    ax.barh([a.feature for a in ordered], [a.impact for a in ordered], color=colours)
    ax.axvline(0, c="black", lw=0.9)
    ax.set_xlabel("Contribution to predicted risk (log-odds)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    target = ensure_parent(path)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def build_explainer(
    estimator: Any, feature_names: list[str], background: pd.DataFrame | None, cfg: Config
) -> FailureExplainer:
    """Construct a :class:`FailureExplainer` from configuration.

    Args:
        estimator: The fitted pipeline.
        feature_names: Model feature order.
        background: Background rows sampled from the training split.
        cfg: Merged configuration.

    Returns:
        The explainer.
    """
    return FailureExplainer(
        estimator=estimator,
        feature_names=feature_names,
        background=background,
        max_background=int(cfg.get("explainability.background_samples", 200)),
        seed=int(cfg.get("random_seed", 42)),
    )
