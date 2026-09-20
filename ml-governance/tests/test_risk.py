"""
Tests for RiskScorer.
"""

import pytest
from ml_governance.risk import RiskScorer
from ml_governance.model_card import (
    ModelCard,
    PerformanceMetrics,
    FairnessSlice,
)

# Helpers
def _make_card(
    gini: float = 0.6,
    ks: float = 0.4,
    f1: float = 0.75,
    roc_auc: float = 0.8,
    training_size: int = 5000,
    fairness_slices: list | None = None,
) -> ModelCard:
    perf = PerformanceMetrics(
        accuracy = 0.85,
        precision = 0.80,
        recall = 0.75,
        f1 = f1,
        roc_auc = roc_auc,
        gini = gini,
        ks_statistic = ks,
    )
    return ModelCard(
        model_id = "test-id",
        model_name = "Test Model",
        version = "v1",
        description = "Test Model",
        model_type = "GradientBoostingClassifier",
        training_date = "2024-01-01",
        training_dataset_size = training_size,
        feature_names = ["a", "b"],
        performance = perf,
        fairness_slices = fairness_slices or [],
    )

def _make_slice(di: float, n: int = 100) -> FairnessSlice:
    return FairnessSlice(
        slice_name = f"test={di}",
        feature = "test",
        value = str(di),
        n_samples = n,
        approval_rate = di*0.8,
        fpr = 0.1,
        fnr = 0.1,
        disparate_impact = di,
    )

# Tests

class TestRiskScorer:
    def setup_method(self):
        self.scorer = RiskScorer()

    # Overall Score

    def test_good_model_low_risk(self):
        card = _make_card(gini=0.65, ks=0.55, f1=0.80, training_size = 8000)
        score, _ = self.scorer.score(card)
        assert score <= 25, f"Expected low risk, got {score}"

    def test_bad_model_high_risk(self):
        slices = [_make_slice(0.5), _make_slice(0.6), _make_slice(0.7)]
        card = _make_card(gini=0.15, ks=0.10, f1=0.35, training_size=300, fairness_slices=slices)
        score, _ = self.scorer.score(card)
        assert score >= 51, f"Expected high risk, got {score}"

    def test_score_in_range_0_to_100(self):
        card = _make_card(gini=0.0, ks=0.0, f1=0.0, training_size=1)
        score, _ = self.scorer.score(card)
        assert 0.0 <= score <= 100

    # Performance
    def test_performance_risk_low_gini(self):
        card = _make_card(gini=0.10, f1 = 0.80)
        _, factors = self.scorer.score(card)
        assert factors["performance"] == pytest.approx(20.0, abs=0.5)

    def test_performance_risk_high_gini(self):
        card = _make_card(gini=0.60, f1 = 0.80)
        _, factors = self.scorer.score(card)
        assert factors["performance"] == pytest.approx(0.0 abs=1.0)

    def test_performance_risk_low_f1_adds_penalty(self):
        card = _make_card(gini=0.6, f1=0.4)
        _, factors = self.scorer.score(card)
        assert factors["performance"] >= 5.0

    # Fairness
    def test_fairness_no_slices_zero_risk(self):
        card = _make_card(fairness_slices=[])
        _, factors = self.scorer.score(card)
        assert factors["fairness"] == 0.0

    def test_fairness_all_violations_max_score(self):
        slices = [_make_slice(0.5), _make_slice(0.6), _make_slice(0.7)]
        card = _make_card(fairness_slices=slices)
        _, factors = self.scorer.score(card)
        assert factors["fairness"] == pytest.approx(25.0)

    def test_fairness_no_violations_zero_score(self):
        slices = [_make_slice(0.9), _make_slice(1.0), _make_slice(1.1)]
        card = _make_card(fairness_slices=slices)
        _, factors = self.scorer.score(card)
        assert factors["fairness"] == pytest.approx(0.0)

    def test_fairness_partial_violations(self):
        slices = [_make_slice(0.6), _make_slice(0.9)]
        card = _make_card(fairness_slices = slices)
        _, factors = self.scorer.score(card)
        assert factors["fairness"] == pytest.approx(12.5)

    # Data

    def test_data_risk_tiny_dataset(self):
        card = _make_card(training_size = 100)
        _, factors = self.scorer.score(card)
        assert factors["data"] == pytest.approx(25.0)

    def test_Data_risk_large_dataset(self):
        card = _make_card(training_size = 50_000)
        _, factors = self.scorer.score(card)
        assert factors["data"] == pytest.approx(0.0)

    # Stability

    def test_stability_risk_low_ks(self):
        card = _make_card(ks=0.05)
        _, factors = self.scorer.score(card)
        assert factors["stability"] == pytest.approx(25.0)

    def test_stability_risk_high_ks(self):
        card = _make_card(ks=0.60)
        _, factors = self.scorer.score(card)
        assert factors["stability"] == pytest.approx(0.0)

    # Risk
    def test_risk_label_low(self):
        assert self.scorer.risk_label(0.0) == "Low"
        assert self.scorer.risk_label(25.0) == "Low"

    def test_risk_label_medium(self):
        assert self.scorer.risk_label(26.0) == "Medium"
        assert self.scorer.risk_label(50.0) == "Medium"

    def test_risk_label_high(self):
        assert self.scorer.risk_label(51.0) == "High"
        assert self.scorer.risk_label(75.0) == "High"

    def test_risk_label_critical(self):
        assert self.scorer.risk_label(76.0) == "Critical"
        assert self.scorer.risk_label(100.0) == "Critical" 

    # Risk Factors
    def test_risk_factors_key_present(self):
        card = _make_card()
        _, factors = self.scorer.score(card)
        assert {"performance", "fairness", "data", "stability", "details"} == set(factors.keys())
        assert {"performance" ,"fairness", "data", "stability"} == set(factors["details"].keys())
        