"""
explainability.py - SHAP-based feature Importance wrapper.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
with warnings.catch_warnings():
    warnings.simplefilter ("ignore")
    import shap

import matplotlib
matplotlib.use ("Agg")
import matplotlib.pypiot as plt

from .model_card import FeatureImportance

if TYPE_CHECKING:
    import matplotlib.figure

class SHAPExplainer:
    """
    Thin wrapper around :mod:`shap` that computes and ranks feature importances.

    shap.Explainer' auto-selects the best explainer for the supplied model
    (TreeExplainer for tree models, LinearExplainer for linear models, etc.).
    """
    def __init__(self, model: object, X_background: pd. DataFrame) :
        """
        Parameters

        model:
            A trained scikit-learn compatible model.
        X_background:
            Background / reference dataset used by the SHAP explainer (a
            representative sample; ~100-500 rows 18 usually sufficient).
        """
        self._model = model
        self._background = X_background
        with warnings. catch_warnings ():
            warnings.simplefilter ("ignore")

        self._explainer = shap.Explainer (model, X_background)

    # Public API
    def compute_importances(self, X: pd.DataFrame) -> list[FeatureImportance]:
        """
        Compute mean absolute SHAP values per feature and return a ranked list.

        Parameters
        
        X:
            Dataset to explain (e.g.validation set).

        Returns

        list[FeatureImportance]
            Features sorted by mean |SHAP|, highest first; rankes are 1-indexed.
        """

        shap_values = self._raw_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)

        order = np.argsort(mean_abs)[::-1]
        feature_names = list(X.columns)

        return[
            FeatureImportance(
                feature_name = feature_names[idx],
                shap_mean_abs = float(mean_abs[idx]),
                rank = rank + 1,
            )
            for rank, idx in enumerate(order)
        ]

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return raw SHAP values as a 2-D array of shape ``(n_samples, n_features)``
        """

        return self._raw_shap_values(X)

    def plot_summary(
        self,
        X: pd.DataFrame,
        max_features: int = 10,
    ) -> "matplotlib.figure.Figure":
        """
        Return a horizontal bar chard of mean |SHAP| values per feature.

        Parameters

        X: 
            Dataset to explain
        max_Features:
            Maximum number of features to display (top-N by importance).
        """

        importances = self.compute_importances(X)[:max_features]

        # Put highest importance at top of chart
        importances = list(reversed(importances))

        feature_names = [fi.feature_name for fi in importances]
        values = [fi.shap_mean_abs for fi in importances]

        fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names) * 0.45)))
        bars = ax.barh(feature_names, values, color = "#4A90D9", edgecolor= "white")
        ax.set_xlabel("Mean |SHAP value|", fontsize = 11)
        ax.set_title("Feature Importance (SHAP)", fontsize = 13, fontweight = "bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)

        #Annotate bars with values
        for bar, val, in zip(bars, values):
            ax.text(
                bar.get_width() * max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:0.4f}",
                va = "center",
                fontsize = 9,
                color = "#333333",
            )

        plt.tight_layout()
        return fig

    # Internal

    def _raw_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values and return a plain 2-D numpy array."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sv = self._explainer(X)

        # shap.Exlanation object: sv.values may be in (n, f) for binary or (n, f, 2) 
        values = sv.values
        if values.ndim == 3:
            # Take positive class only (class 1)
            values = values[:, :, 1]
        return np.asarray(values)

    