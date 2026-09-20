"""
tests/test_evaluation.py - Tests for ModelEvaluator
"""

import numpy as np
import pandas as pd
import pytest

from ml_governance.evaluation import ModelEvaluator

# Helpers

RNG = np.random.default_rng(33)

def _synthetic_binary(n: int = 200, positive_rate: float = 0.3):
    y_true = RNG.binomial(1, positive_rate, n)

    # Biased probabilities that strongly correlate with y_true
    y_prob = np.clip(y_true * 0.6 + RNG.uniform(0.0, 0.4, n), 0.0, 1.0)
    y_pred = (y_prob > 0.5).astype(int)
    return y_true, y_pred, y_prob

# compute_performance

class TestComputePerformance:
    def setup_method(self):
        self.evaluator = ModelEvaluator()

    def test_returns_performance_metrics(self):
        from ml_governance.model_card import PerformanceMetrics
        y_true, y_pred, y_prob = _synthetic_binary()
        result = self.evaluator.compute_performance(y_true, y_pred, y_prob)
        assert isinstance(result, PerformanceMetrics)

    def test_gini_equals_2_auc_minus_1(self):
        y_true, y_pred, y_prob = _synthetic_binary()
        result = self.evaluator.compute_performance(y_true, y_pred, y_prob)
        assert result.gini == pytest.approx(2.0 * result.roc_auc - 1.0, abs=1e-9)

    def test_perfect_classifier(self):
        y_true = np.array([0,0,1,1,0,1])
        y_prob = np.array([0.1,0.2,0.9,0.85,0.15,0.95])
        y_pred = (y_prob > 0.5).astype(int)
        result = self.evaluator.compute_performance(y_true, y_pred, y_prob)
        assert result.roc_auc == pytest.approx(1.0)
        assert result.gini == pytest.approx(1.0)

    def test_ks_statistic_positive(self):
        y_true, y_pred, y_prob = _synthetic_binary()
        result = self.evaluator.compute_performance(y_true, y_pred, y_prob)
        assert result.ks_statistic > 0.0

    def test_ks_statistic_range(self):
        y_true, y_pred, y_prob = _synthetic_binary()
        result = self.evaluator.compute_performance(y_true, y_pred, y_prob)
        assert 0.0 <= result.ks_statistic <= 1.0

    def test_accuracy_in_range(self):
        y_true, y_pred, y_prob = _synthetic_binary()
        result = self.evaluator.compute_performance(y_true, y_pred, y_prob)
        assert 0.0 <= result.accuracy <= 1.0

    def test_known_values(self):
        #Deterministic small sample
        y_true = np.array([1,0,1,0,1,0,1,0])
        y_pred = np.array([1,0,1,0,1,0,0,1])
        y_prob = np.array([0.9,0.1,0.8,0.2,0.7,0.3,0.4,0.6])

        result = self.evaluator.compute_performance(y_true, y_pred, y_prob)
        assert result.accuracy == pytest.approx(0.75, abs=1e-6)

# compute_fairness_slices

class TestComputeFairnessSlices:
    def setup_method(self):
        self.evaluator = ModelEvaluator()

    def _make_biased_dataset(self):
        """Group A gets approved 90%; Group B gets approved 50%"""
        n = 200
        group = np.array(["A"] * 100 + ["B"] * 100)
        y_true = np.zeros(n, dtype=int)
        y_pred = np.zeros(n, dtype=int)

        y_pred[:100] = RNG.choice([0,1], 100, p=[0.9,0.1])
        y_pred[100:] = RNG.choice([0,1], 100, p=[0.5,0.5])

        y_prob = 1.0 - y_pred.astype(float) + RNG.uniform(-0.05, 0.05, n)
        y_prob = np.clip(y_prob, 0.0, 1.0)

        X = pd.DataFrame({"group":group, "income": RNG.uniform(20,80,n)})
        return X, y_true, y_pred, y_prob

    def test_returns_slices(self):
        X, y_true, y_pred, y_prob = self._make_biased_dataset()
        slices = self.evaluator.compute_fairness_slices(X, y_true, y_pred, y_prob, ["group"])
        assert len(slices) == 2

    def test_slice_names(self):
        X, y_true, y_pred, y_prob = self._make_biased_dataset()
        slices = self.evaluator.compute_fairness_slices(X, y_true, y_pred, y_prob, ["group"])
        names = {s.slice_name for s in slices}
        assert "group=A" in names
        assert "group=B" in names

    def test_biased_group_low_disparate_impact(self):
        X, y_true, y_pred, y_prob = self._make_biased_dataset()
        slices = self.evaluator.compute_fairness_slices(X, y_true, y_pred, y_prob, ["group"])
        b_slice = next(s for s in slices if s.value =="B")
        assert b_slice.disparate_impact < 0.9

    def test_small_slice_excluded(self):
        """ Fewer than 30 samples should be dropped """
        X = pd.DataFrame({"rare":["X"] * 10 + ["Y"] * 200})
        y_true = np.zeros(210, dtype=int)
        y_pred = np.zeros(210, dtype=int)
        y_prob = np.zeros(210)
        slices = self.evaluator.compute_fairness_slices(X, y_true, y_pred, y_prob, ["rare"])
        assert all(s.n_samples >= 30 for s in slices)

    def test_missing_feature_ignored(self):
        X, y_true, y_pred, y_prob = self._make_biased_dataset()
        # non existent column skipped
        slices = self.evaluator.compute_fairness_slices(
            X, y_true, y_pred, y_prob, ["nonexistent"]
        )
        assert slices == []

    def test_disparate_impact_values_in_range(self):
        X, y_true, y_pred, y_prob = self._make_biased_dataset()
        slices = self.evaluator.compute_fairness_slices(X, y_true, y_pred, y_prob, ["group"])
        for s in slices:
            assert s.disparate_impact >= 0.0

# compute ks curve

class TestComputeKsCurve:
    def setup_method(self):
        self.evaluator = ModelEvaluator()

    def test_returns_expected_keys(self):
        y_true, _, y_prob = _synthetic_binary()
        result = self.evaluator.compute_ks_curve(y_true, y_prob)
        assert set(result.keys()) == {"thresholds", "cumulative_good", "cumulative_bad", "ks_values"}

    def test_returns_10_deciles(self):
        y_true, _, y_prob = _synthetic_binary()
        result = self.evaluator.compute_ks_curve(y_true, y_prob)
        assert len(result["thresholds"]) == 10
        assert len(result["cumulative_good"]) == 10
        assert len(result["cumulative_bad"]) == 10
        assert len(result["ks_values"]) == 10

    def test_thresholds_are_deciles(self):
        y_true, _, y_prob = _synthetic_binary()
        result = self.evaluator.compute_ks_curve(y_true, y_prob)
        assert result["thresholds"] == list(range(10,110,10))

    def test_cumulative_rates_monotone(self):
        y_true, _, y_prob = _synthetic_binary()
        result = self.evaluator.compute_ks_curve(y_true, y_prob)
        cumulative_bad = result["cumulative_bad"]
        assert all(cumulative_bad[i] <= cumulative_bad[i+1] for i in range(len(cumulative_bad) - 1))


                
                
