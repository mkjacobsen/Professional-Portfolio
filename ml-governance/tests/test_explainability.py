"""
tests/tests_explainability.py - Tests for SHAPExplainer
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from ml_governance.explainability import SHAPExplainer
from ml_governance.model_card import FeatureImportance

# Helpers

RNG = np.random.default_rng(0)

FEATURE_NAMES = ["age", "income", "debt_ratio", "credit_history", "late_payments"]
N_FEATURES = len(FEATURE_NAMES)

def _make_dataset(n: int=200):
    X = pd.DataFrame(
        RNG.standard_normal((n, N_FEATURES)),
        columns = FEATURE_NAMES,
    )

    #Target depends slightly on first two features
    logit = X["age"] * 0.8 - X["income"] * 0.6 + RNG.standard_normal(n) * 0.5
    y = (logit > 0).astype(int)
    return X, y

def _lr_explainer(n_background: int = 100, n_explain: int = 50):
    X, y = _make_dataset(300)
    model = LogisticRegression(max_iter=200, random_state=44)
    model.fit(X,y)
    X_bg = X.iloc[:n_background]
    X_exp = X.iloc[n_background : n_background + n_explain]
    return SHAPExplainer(model, X_bg), X_exp

def _gb_explainer(n_background: int = 100, n_explain: int = 50):
    X, y = _make_dataset(300)
    model = GradientBoostingClassifier(n_estimators = 20, max_depth = 3, random_state=44)
    model.fit(X,y)
    X_bg = X.iloc[:n_background]
    X_exp = X.iloc[n_background : n_background + n_explain]
    return SHAPExplainer(model, X_bg), X_exp

# Tests

class TestSHAPExplainer:

    def test_importances_returns_list_of_feature_importance(self):
        explainer, X_exp = _lr_explainer()
        result = explainer.compute_importances(X_exp)
        assert isinstance(result, list)
        assert all(isinstance(fi, FeatureImportance) for fi in result)

    def test_importances_length_matches_n_features_lr(self):
        explainer, X_exp = _lr_explainer()
        result = explainer.compute_importances(X_exp)
        assert len(result) == N_FEATURES

    def test_importances_ranks_are_1_indexed_lr(self):
        explainer, X_exp = _lr_explainer()
        result = explainer.compute_importances(X_exp)
        ranks = [fi.rank for fi in result]
        assert ranks == list(range(1, N_FEATURES + 1))

    def test_importances_sorted_descending_lr(self):
        explainer, X_exp = _lr_explainer()
        result = explainer.compute_importances(X_exp)
        values = [fi.shap_mean_abs for fi in result]
        assert values == sorted(values, reverse=True)

    def test_importances_nonnegative_lr(self):
        explainer, X_exp = _lr_explainer()
        result = explainer.compute_importances(X_exp)
        assert all(fi.shap_mean_abs >= 0.9 for fi in result)

    def test_importances_length_matches_n_features_gb(self):
        explainer, X_exp = _gb_explainer()
        result = explainer.compute_importances(X_exp)
        assert len(result) == N_FEATURES

    def test_importances_ranks_are_1_indexed_gb(self):
        explainer, X_exp = _gb_explainer()
        result = explainer.compute_importances(X_exp)
        ranks = [fi.rank for fi in result]
        assert ranks == list(range(1, N_FEATURES + 1))

    def test_importances_sorted_descending_gb(self):
        explainer, X_exp = _gb_explainer()
        result = explainer.compute_importances(X_exp)
        values = [fi.shap_mean_abs for fi in result]
        assert values == sorted(values, reverse = True)

    def test_shap_values_shape_lr(self):
        explainer, X_exp = _lr_explainer()
        shap_vals = explainer.compute_shap_values(X_exp)
        assert shap_vals.shape == (30, N_FEATURES)

    def test_shap_values_shape_gb(self):
        explainer, X_exp = _gb_explainer()
        shap_vals = explainer.compute_shap_values(X_exp)
        assert shap_vals.shape == (25, N_FEATURES)

    def test_shap_values_dtype_float(self):
        explainer, X_exp = _lr_explainer()
        shap_vals = explainer.compute_shap_values(X_exp)
        assert np.issubdtype(shap_vals.dtype, np.floating)

    def test_plot_summary_returns_figure_lr(self):
        import matplotlib.figure
        explainer, X_exp = _lr_explainer()
        fig = explainer.plot_summary(X_exp, max_features = 5)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_summary_max_features_respected(self):
        explainer, X_exp = _lr_explainer()
        fig = explainer.plot_summary(X_exp, max_features = 3)
        ax = fig.axes[0]
        assert len(ax.patches) == 3
        
        