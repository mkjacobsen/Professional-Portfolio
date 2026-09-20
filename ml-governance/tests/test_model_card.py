"""
Serialization round-trip tests for ModelCard
"""

import json
import pytest

from ml_governance.model_card import (
    ModelCard,
    PerformanceMetrics,
    FairnessSlice,
    FeatureImportance,
)

# Fixtures

def _make_full_card() -> ModelCard:
    perf = PerformanceMetrics(
        accuracy = 0.87,
        precision = 0.82,
        recall = 0.78,
        f1 = 0.80,
        roc_auc = 0.91,
        gini = 0.82,
        ks_statistic = 0.52,
    )

    slices = [
        FairnessSlice(
            slice_name = "age_group=18-25",
            feature = "age_group",
            value = "18-25" ,
            n_samples = 120,
            approval_rate = 0.70,
            fpr = 0.10,
            fnr = 0.15,
            disparate_impact = 0.85,
        ),
        FairnessSlice(
            slice_name = "age_group55+",
            feature = "age_group",
            value = "55+",
            n_samples = 95,
            approval_rate = 0.60,
            fpr = 0.12,
            fnr = 0.18,
            disparate_impact = 0.73,
        ),
    ]

    importances = [
        FeatureImportance(feature_name="debt_ratio", shap_mean_abs=0.23, rank=1),
        FeatureImportance(feature_name="credit_history_length", shap_mean_abs=0.18, rank=2),
    ]

    return ModelCard(
        model_id="abc-123",
        model_name="CreditScorer",
        version="v1.2",
        description="Logistic regression credit default model.",
        model_type="LogisticRegression",
        training_date="2024-03-01",
        training_dataset_size = 5000,
        feature_names = ["age", "income", "debt_ratio", "credit_history_length"],
        performance = perf,
        fairness_slices = slices,
        feature_importances = importances,
        risk_score = 22.5,
        risk_factors = {"performance": 5.0, "fairness": 8.5, "data": 4.0, "stability": 5.0}
        intended_use = "Approve/decline consumer credit applications.",
        limitations="Not validated for commercial real estate loans.",
    )

# Dict Round-Trip
class TestToDictFromDict:
    def test_roundtrip_preserves_model_name(self):
        card = _make_full_card()
        assert ModelCard.from_dict(card.to_dict()).model_name == card.model_name

    def test_roundtrip_preserves_performance(self):
        card = _make_full_card()
        restored = ModelCard.from_dict(card.to_dict())
        assert restored.performance.gini == pytest.approx(card.performance.gini)
        assert restored.performance.ks_statistic == pytest.approx(card.performance.ks_statistic)

    def test_roundtrip_preserves_fairness_slices(self):
        card = _make_full_card()
        restored = ModelCard.from_dict(card.to_dict())
        assert len(restored.fairness_slices) == 2
        assert restored.fairness_slices[1].disparate_impact == pytest.approx(0.73)

    def test_roundtrip_preserves_feature_importances(self):
        card = _make_full_card()
        restored = ModelCard.from_dict(card.to_dict())
        assert len(restored.feature_importances) == 2
        assert restored.feature_importances[0].feature_name == "debt_ratio"

    def test_roundtrip_preserves_risk_score(self):
        card = _make_full_card()
        restored = ModelCard.from_dict(card.to_dict())
        assert restored.risk_score == pytest.approx(22.5)

    def test_to_dict_returns_pain_dict(self):
        card = _make_full_card()
        d = card.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["performance"], dict)
        assert isinstance(d["fairness_slices"], list)

    def test_empty_slices_roundtrip(self):
        card = _make_full_card()
        card.fairness_slices = []
        card.feature_importances = []
        restored = ModelCard.from_dict(card.to_dict())
        assert restored.fairness_slices == []
        assert restored.feature_importances == []

# JSON round trip
class TestToJsonFromJson:
    def test_to_json_is_valid_json(self):
        card = _make_full_card()
        raw = card.to_json()
        parsed = json.loads(raw)
        assert parsed["model_name"] == "CreditScorer"

    def test_json_roundtrip_full(self):
        card = _make_full_card()
        restored = ModelCard.from_json(card.to_json())
        assert restored.model_id == card.model_id
        assert restored.version == card.version
        assert restored.performance.roc_auc == pytest.approx(card.performance.roc_auc)
        assert len(restored.fairness_slices) == len(card.fairness_slices)
        assert len(restored.feature_importances) == len(card.feature_importances)

    def test_json_roundtrip_preserves_risk_factors(self):
        card = _make_full_card()
        restored = ModelCard.from_json(card.to_json())
        assert restored.risk_factors["performance"] == pytest.approx(0.5)

    def test_json_roundtrip_optional_none(self):
        card = _make_full_card()
        card.risk_score = None
        restored = ModelCard.from_json(card.to_json())
        assert restored.risk_score is None

    
