"""
Integration tests for MLOpsPipeline

These tests exercise the full train -> validate -> promote -> monitor workflow on
small synthetic datasets to keep runtime reasonable.
"""

import pandas as pd
import pytest

from mlops.pipeline import MLOpsPipeline
from mlops.training import TrainingResult, ValidationGate
from mlops.monitoring import MonitoringReport

# Fixtures
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Create a small MLOpsPipeline backed by temp directories."""
    base = tmp_path_factory.mktemp("pipeline")
    return MLOpsPipeline(
        data_path = base / "data",
        artifact_store_path = base / "artifacts",
        feature_store_path = base / "features",
        gate = ValidationGate(max_mae = 50.0, max_rmse = 1000.0, min_r2 = 0.0),
        n_stores = 3,
        n_products = 5,
        seed = 0,
    )

@pytest.fixture(scope="module")
def trained_result(pipeline):
    """Run the training pipeline once and cache the result."""
    return pipeline.run_training_pipeline(
        train_start = "2022-01-01",
        train_end = "2022-09-30",
        val_start = "2022-10-01",
        val_end = "2022-12-31",
    )

# Training pipeline tests

class TestTrainingPipeline:
    def test_returns_training_result(self, trained_result):
        assert isinstance(trained_result, TrainingResult)

    def test_passed_gate_with_lenient_thresholds(self, trained_result):
        assert trained_result.passed_gate is True

    def test_mae_is_finite_positive(self, trained_result):
        assert trained_result.mae > 0.0
        import math
        assert math.isfinite(trained_result.mae)

    def test_r2_bounded(self, trained_result):
        assert trained_result.r2 <= 1.0

    def test_sample_counts_reasonable(self, trained_result):
        assert trained_result.n_train_samples > 0
        assert trained_result.n_val_samples > 0

    def test_feature_names_populated(self, trained_result):
        assert len(trained_result.feature_names) > 0

    def test_champion_saved_in_artifact_store(self, pipeline, trained_result):
        champion = pipeline.artifact_store.load_champion()
        assert champion is not None
        model, metrics = champion
        assert model is not None
        assert "mae" in metrics

    def test_champion_mae_matches_result(self, pipeline, trained_result):
        stored_mae = pipeline.artifact_store.get_champion_mae()
        assert stored_mae == pytest.approx(trained_result.mae, rel = 1e-4)

    def test_feature_snapshot_saved(self, pipeline, trained_result):
        snapshots = pipeline.feature_store.list_snapshots()
        assert "training_features" in snapshots

    def test_feature_snapshot_has_correct_columns(self, pipeline):
        from mlops.features import FEATURE_COLUMNS
        snapshot = pipeline.feature_store.load_snapshot("training_features")
        for col in FEATURE_COLUMNS:
            assert col in snapshot.columns, f"Missing column: {col}"

# Monitoring Pipeline Tests

class TestMonitoringPipeline:
    def test_resturns_monitoring_reporT(self, pipeline, trained_result):
        report = pipeline.run_monitoring(
            monitoring_start="2023-01-01",
            monitoring_end="2023-03-31",
        )
        assert isinstance(report, MonitoringReport)

    def test_report_has_sample_counts(self, pipeline, trained_result):
        report = pipeline.run_monitoring(
            monitoring_start="2023-01-01",
            monitoring_end="2023-01-31",
        )
        assert report.n_reference_samples > 0
        assert report.n_current_samples > 0

    def test_report_summary_non_empty(self, pipeline, trained_result):
        report = pipeline.run_monitoring(
            monitoring_start="2023-01-01",
            monitoring_end="2023-01-31",
        )
        assert len(report.summary) > 0

    def test_drift_deteted_after_injection(self, pipeline, trained_result):
        report = pipeline.run_monitoring(
            monitoring_start="2023-01-01",
            monitoring_end="2023-03-31",
            drift_type="covariate",
        )
        assert report.overall_drift_detected is True

    def test_retraining_recommended_after_heavy_drift(self, pipeline, trained_result):
        report = pipeline.run_monitoring(
            monitoring_start="2023-01-01",
            monitoring_end="2023-06-30",
            drift_type="covariate",
        )
        assert report.retraining_recommended is True

    def test_monitoring_fails_without_snapshot(self, tmp_path):
        pipeline_no_snap = MLOpsPipeline(
            data_path = tmp_path / "data",
            artifact_store_path = tmp_path / "art",
            feature_store_path = tmp_path / "feat",
            n_stores = 2,
            n_products = 3,
        )
        with pytest.raises(RuntimeError, match = "snapshot"):
            pipeline_no_snap.run_monitoring("2023-01-01", " 2023-01-31")