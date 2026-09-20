"""
Tests for ModelRegistry
"""

import pytest
import tempfile
from pathlib import Path
from ml_governance.registry import ModelRegistry
from ml_governance.model_card import ModelCard, PerformanceMetrics

# Helpers

def _make_performance(**overrides) -> PerformanceMetrics:
    defaults = dict (
        accuracy = 0.85, precision = 0.80, recall = 0.75,
        f1 = 0.77, roc_auc = 0.88, gini = 0.76, ks_statistic = 0.45,
    )
    defaults.update(overrides)
    return PerformanceMetrics(**defaults)

def _make_card (
        model_id: str = "tmp",
        name: str = "TestModel",
        version: str = "v1.0",
) -> ModelCard:
    return ModelCard(
        model_id = model_id,
        model_name = name,
        version = version,
        description = "A test model",
        model_type = "LogisticRegression",
        training_date = "2024-01-01",
        training_dataset_size = 2000,
        feature_names = ["age", "income"],
        performance = _make_performance(),
        intended_use = "Testing",
        limitations = "None",
    )

class _DummyModel:
    """Trivial picklable object to stand in for model"""

    def predict(self, X):
        return [0] * len(X)

# Tests

def test_register_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        card = _make_card()
        model = _DummyModel()

        model_id = registry.register(model, "TestModel", "v1.0", card)

        assert isinstance(model_id, str) and len(model_id) > 0

        retrieved_model, retrieved_card = registry.get(model_id)

        assert isinstance(retrieved_model, _DummyModel)
        assert retrieved_card.model_name == "TestModel"
        assert retrieved_card.version == "v1.0"
        assert retrieved_card.model_id == model_id

def test_register_persists_tags():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        card = _make_card()
        model_id = registry.register(
            _DummyModel(), "TagModel", "v2.0", card, tags = {"env": "staging", "team": "risk"}
        )
        entries = registry.list_models()
        entry = next(e for e in entries if e.model_id == model_id)
        assert entry.tags["env"] == "staging"
        assert entry.tags["team"] == "risk"

def test_list_models_no_filter():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        for i in range (3):
            registry.register(_DummyModel(), f"Model {i}", "v1.0", _make_card())
        entries = registry.list_models()
        assert len(entries) == 3

def test_list_models_status_filter():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        id1 = registry.register(_DummyModel(), "M1", "v1", _make_card())
        id2 = registry.register(_DummyModel(), "M2", "v1", _make_card())

        registry.update_status(id1, "production")
        prod = registry.list_models(status="production")
        staging = registry.list_models(status="staging")

        assert len(prod) == 1 and prod[0].model_id == id1
        assert len(staging) == 1 and staging[0].model_id == id2

def test_update_status_valid():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        mid = registry.register(_DummyModel(), "M", "v1", _make_card())
        registry.update_status(mid, "production")
        entries = registry.list_models(status="production")
        assert any(e.model_id == mid for e in entries)

def test_update_status_invalid_raises():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        mid = registry.register(_DummyModel(), "M", "v1", _make_card())
        with pytest.raises(ValueError, match = "Invalid status"):
            registry.update_status(mid, "deleted")

def test_update_status_unknown_model_raises():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        with pytest.raises(KeyError):
            registry.update_status("nonexistent-id", "production")

def test_get_unknown_model_raises():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        with pytest.raises(KeyError):
            registry.get("does-not-exist")

def test_compare_two_models():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)

        card_1 = _make_card(name="ModelA")
        card_1.performance = _make_performance(roc_auc=0.80, gini=0.60)
        card_1.risk_score = 30.0

        card_2 = _make_card(name="ModelB")
        card_2.performance = _make_performance(roc_auc=0.90, gini=0.80)
        card_2.risk_score = 10.0

        id_a = registry.register(_DummyModel(), "ModelA", "v1", card_1)
        id_b = registry.register(_DummyModel(), "ModelB", "v1", card_2)

        result = registry.compare(id_a, id_b)

        assert "model_a" in result
        assert "model_b" in result
        assert "delta" in result

        assert result["model_a"]["gini"] == pytest.approx(0.60, abs=1e-4)
        assert result["model_b"]["gini"] == pytest.approx(0.80, abs=1e-4)
        assert result["delta"]["gini"] == pytest.approx(0.20, abs=1e-4)
        assert result["delta"]["risk_score"] == pytest.approx(-20.0, abs=1e-4)
