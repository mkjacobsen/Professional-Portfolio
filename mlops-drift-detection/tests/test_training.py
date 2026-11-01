"""
Unit tests for ModelTrainer and ValidationGate
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from mlops.training import ModelTrainer, TrainingResult, ValidationGate

# Fixtures

@pytest.fixture(scope = "module")
def simple_regression_data():
    X, y = make_regression(
        n_samples = 600, n_features = 8, noise = 5.0, random_state = 30
    )
    y = (y - y.min()) / (y.max() - y.min()) * 200
    col_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=col_names)
    y_ser = pd.Series(y, name="demand")
    return(
        X_df.iloc[:400],
        y_ser.iloc[:400],
        X_df.iloc[400:],
        y_ser.iloc[400:],
    )

# Training Tests

class TestModelTrainer:
    def test_returns_training_result(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        trainer = ModelTrainer()
        result = trainer.train(X_train, y_train, X_val, y_val)
        assert isinstance(result, TrainingResult)

    def test_result_fields_populated(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        trainer = ModelTrainer()
        result = trainer.train(X_train, y_train, X_val, y_val)

        assert result.model is not None
        assert result.mae >= 0.0
        assert result.rmse >= 0.0
        assert result.r2 <= 1.0
        assert result.trained_at != ""
        assert result.n_trained_samples == len(X_train)
        assert result.n_val_samples == len(X_val)
        assert len(result.feature_names) == X_train.shape[1]

    def test_gate_reason_non_empty(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        trainer = ModelTrainer()
        result = trainer.train(X_train, y_train, X_val, y_val)
        assert len(result.gate_reason) > 0

    def test_model_can_predict(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        trainer = ModelTrainer()
        result = trainer.train(X_train, y_train, X_val, y_val)
        preds = result.model.predict(X_val)
        assert len(preds) == len(X_val)

# Validation Gate Pass Scenarios

class TestValidationGatePass:
    def test_gate_passess_with_good_metrics(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae=500.0, max_rmse=10000.0, min_r2 = 0.0)
        trainer = ModelTrainer(gate = gate)
        result = trainer.train(X_train, y_train, X_val, y_val)
        assert result.passed_gate is True
        assert "PASSED" in result.gate_reason

    def test_gate_passes_no_champion(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 500.0, max_rmse = 1000.0, min_r2 = 0.0)
        trainer = ModelTrainer(gate=gate)
        result = trainer.train(X_train, y_train, X_val, y_val, champion_mae = None)
        assert result.passed_gate is True

# Validation Gate Fail Scenarios

class TestValidationGateFail:
    def test_gate_fails_on_mae_threshold(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 0.0, max_rmse = 1e9, min_r2 = -1e9)
        trainer = ModelTrainer(gate=gate)
        result = trainer.train(X_train, y_train, X_val, y_val)
        assert result.passed_gate is False
        assert "MAE" in result.gate_reason
        assert "FAILED" in result.gate_reason

    def test_gate_fails_on_rmse_threshold(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 1e9, max_rmse = 0.0, min_r2 = -1e9)
        trainer = ModelTrainer(gate=gate)
        result = trainer.train(X_train, y_train, X_val, y_val)
        assert result.passed_gate is False
        assert "RMSE" in result.gate_reason

    def test_gate_fails_on_r2_threshold(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 1e9, max_rmse = 1e9, min_r2 = 1.0)
        trainer = ModelTrainer(gate=gate)
        result = trainer.train(X_train, y_train, X_val, y_val)
        assert result.passed_gate is False
        assert "R2" in result.gate_reason

    def test_gate_fails_on_improvement_check(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 1e9, max_rmse = 1e9, min_r2 = -1e9, min_improvement=0.9999)
        trainer = ModelTrainer(gate=gate)
        result = trainer.train(X_train, y_train, X_val, y_val, champion_mae = 0.0001)
        assert result.passed_gate is False
        assert "champion" in result.gate_reason.lower() or "improve" in result.gate_reason.lower()

# Metrics Correctness

class TestMetricsValues:
    def test_mae_non_negative(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 1e9, max_rmse = 1e9, min_r2 = -1e9)
        result = ModelTrainer(gate=gate).train(X_train, y_train, X_val, y_val)
        assert result.mae >= 0.0

    def test_rmse_gte_mae(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 1e9, max_rmse = 1e9, min_r2 = -1e9)
        result = ModelTrainer(gate=gate).train(X_train, y_train, X_val, y_val)
        assert result.rmse >= result.mae - 1e-6

    def test_r2_bounded(self, simple_regression_data):
        X_train, y_train, X_val, y_val = simple_regression_data
        gate = ValidationGate(max_mae = 1e9, max_rmse = 1e9, min_r2 = -1e9)
        result = ModelTrainer(gate=gate).train(X_train, y_train, X_val, y_val)
        assert result.r2 <= 1.0


        