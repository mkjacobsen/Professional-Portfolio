"""
Unit tests for DriftDetector and drift reporting
"""

import numpy as np
import pandas as pd
import pytest

from mlops.monitoring import DriftDetector, DriftResult, MonitoringReport

# Helpers

def _make_df(mean: float, std: float, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Return single-column DataFrame with normally distributed values"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"feature": rng.normal(loc=mean, scale=std, size=n)})

def _make_multi_df(
    means: dict[str, float],
    std: float = 5.0,
    n: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"col": rng.normal(loc=m, scale=std, size=n) for col, m in means.items()}
    )

# PSI Tests

class TestPSI:
    def test_psi_zero_for_identical_distributions(self):
        """PSI must be ~0 when reference and current are the same data"""
        rng = np.random.default_rng(1)
        arr = rng.normal(loc=10.0, scale=2.0, size=1000)
        reference = pd.DataFrame({"x": arr})
        current = pd.DataFrame({"x": arr})

        detector = DriftDetector(reference)
        results = detector.check_feature_drift(current, features=["x"])
        psi_results = [r for r in results if r.method == "PSI"]
        assert len(psi_results) == 1
        assert psi_results[0].statistic == pytest.approx(0.0, abs=1e-4)

    def test_psi_low_for_similar_distributions(self):
        """Small shift should produce PSI < PSI_WARNING"""
        rng = np.random.default_rng(2)
        reference = pd.DataFrame({"x": rng.normal(10.0, 2.0, 2000)})
        current = pd.DataFrame({"x": rng.normal(10.1, 2.0, 2000)})

        detector = DriftDetector(reference)
        results = detector.check_feature_drift(current, features=["x"])
        psi = next(r for r in results if r.method == "PSI")
        assert psi.statistic < DriftDetector.PSI_WARNING
        assert psi.is_drift is False
        assert psi.severity == "none"

    def test_psi_critical_for_heavily_shifted_distributions(self):
        rng = np.random.default_rng(44)
        reference = pd.DataFrame({"x": rng.normal(10.0, 2.0, 2000)})
        current = pd.DataFrame({"x": rng.normal(20.0, 2.0, 2000)})
about:blank#blocked
        detector = DriftDetector(reference)
        results = detector.check_feature_drift(current, features=["x"])
        psi = next(r for r in results if r.method == "PSI")
        assert psi.statistic >= DriftDetector.PSI_CRITICAL
        assert psi.is_drift is True
        assert psi.severity == "critical"

    def test_psi_warning_for_moderately_shifted_distributions(self):
        rng = np.random.default_rng(99)
        reference = pd.DataFrame({"x": rng.normal(0.0, 1.0, 4000)})
        current = pd.DataFrame({"x": rng.normal(1.5, 1.0, 4000)})

        detector = DriftDetector(reference)
        results = detector.check_feature_drift(current, features=["x"])
        psi = next(r for r in results if r.method == "PSI")
        assert psi.statistic >= DriftDetector.PSI_WARNING
        assert psi.is_drift is True

    def test_psi_result_fields(self):
        rng = np.random.default_rng(15)
        reference = pd.DataFrame({"x": rng.normal(0.0, 1.0, 400)})
        current = pd.DataFrame({"x": rng.normal(5.0, 1.0, 400)})

        detector = DriftDetector(reference)
        results = detector.check_feature_drift(current, features=["x"])
        psi = next(r for r in results if r.method == "PSI")
        
        assert psi.p_value is None
        assert psi.feature == "x"
        assert psi.method == "PSI"

# KS Test 

class TestKSTest:
    def test_ks_p_value_high_for_same_distribution(self):
        """Identical data should yield very high KS p-value"""
        rng = np.random.default_rng(5)
        arr = rng.normal(0.0, 1.0, 1000)
        ref = pd.DataFrame({"x": arr})
        cur = pd.DataFrame({"x": arr})

        detector = DriftDetector(ref)
        results = detector.check_feature_drift(cur, features=["x"])
        ks = next(r for r in results if r.method == "KS")
        assert ks.p_value == pytest.approx(1.0, abs=1e-6)
        assert ks.is_drift is False

    def test_ks_p_value_low_for_different_distributions(self):
        rng = np.random.default_rng(6)
        ref = pd.DataFrame({"x": rng.normal(0.0, 1.0, 1000)})
        cur = pd.DataFrame({"x": rng.normal(10.0, 1.0, 1000)})

        detector = DriftDetector(ref)
        results = detector.check_feature_drift(cur, features=["x"])
        ks = next(r for r in results if r.method == "KS")
        assert ks.p_value < DriftDetector.KS_ALPHA
        assert ks.is_drift is True

    def test_ks_result_has_p_value(self):
        rng = np.random.default_rng(7)
        ref = pd.DataFrame({"x": rng.normal(0.0, 1.0, 200)})
        cur = pd.DataFrame({"x": rng.normal(5.0, 1.0, 200)})
        detector = DriftDetector(ref)
        results = detector.check_feature_drift(cur, features=["x"])
        ks = next(r for r in results if r.method == "KS")
        assert ks.p_value is not None
        assert 0.0 <= ks.p_value <= 1.0

    def test_check_feature_drift_returns_both_methods(self):
        rng = np.random.default_rng(8)
        ref = pd.DataFrame({"a": rng.normal(0.0, 1.0, 300), "b": rng.normal(5.0, 1.0, 300)})
        cur = pd.DataFrame({"a": rng.normal(0.0, 1.0, 300), "b": rng.normal(6.0, 1.0, 300)})
        detector = DriftDetector(ref)
        results = detector.check_feature_drift(cur, features=["a", "b"])
        methods = [(r.feature, r.method) for r in results]
        assert ("a", "PSI") in methods
        assert ("a", "KS") in methods
        assert ("b", "PSI") in methods
        assert ("b", "KS") in methods

# Label Drift

class TestLabelDrift:
    def test_no_label_drift_for_same_data(self):
        rng = np.random.default_rng(9)
        arr = rng.normal(50.0, 10, 500)
        ref = pd.DataFrame({"demand": arr})
        detector = DriftDetector(ref)
        result = detector.check_label_drift(arr, arr)
        assert result.is_drift is False
        assert result.p_value == pytest.approx(1.0, abs=1e-6)

    def test_label_drift_detected_for_shifted_demand(self):
        rng = np.random.default_rng(10)
        ref_labels = rng.normal(50.0, 5.0 ,1000)
        cur_labels = rng.normal(20.0, 5.0, 1000)
        ref = pd.DataFrame({"demand": ref_labels})
        detector = DriftDetector(ref)
        result = detector.check_label_drift(ref_labels, cur_labels)        
        assert result.is_drift is True
        assert result.p_value < DriftDetector.KS_ALPHA
        assert result.feature == "demand"
        assert result.method == "KS"

# Generate Report

class TestGenerateReport:
    def test_report_type(self):
        rng = np.random.default_rng(11)
        ref = _make_multi_df({"price": 10.0, "lab_1_demand": 50.0})
        cur = _make_multi_df({"price": 10.0, "lag_1_demand": 50.0}, seed = 44)
        detector = DriftDetector(ref)
        report = detector.generate_report(cur)
        assert isinstance(report, MonitoringReport)


    def test_report_no_drift_for_stable_data(self):
        rng = np.random.default_rng(12)
        arr = rng.normal(10.0, 2.0, 1000)
        ref = pd.DataFrame({"x": arr})
        cur = pd.DataFrame({"x": arr})
        detector = DriftDetector(ref)
        report = detector.generate_report(cur)
        assert report.overall_drift_detected is False
        assert report.retraining_recommended is False

    def test_report_retraining_recommended_when_drift_injected(self):
        rng = np.random.default_rng(13)
        ref = pd.DataFrame({
            "price": rng.normal(10.0, 1.0, 2000),
            "lag_1_demand": rng.normal(50.0, 5.0, 2000),
        })

        cur = pd.DataFrame({
            "price": rng.normal(30.0, 1.0, 2000),
            "lag_1_demand": rng.normal(50.0, 5.0, 2000),
        })
        detector = DriftDetector(ref)
        report = detector.generate_report(cur)
        assert report.overall_drift_detected is True
        assert report.retraining_recommended is True

    def test_report_with_label_drift(self):
        rng = np.random.default_rng(14)
        ref_features = pd.DataFrame({"x": rng.normal(0.0, 1.0, 500)})
        ref_labels = rng.normal(50.0, 5.0, 500)
        cur_features = pd.DataFrame({"x": rng.normal(0.0, 1.0, 500)})
        cur_labels = rng.normal(10.0, 5.0, 500)

        ref = ref_features.copy()
        ref["demand"] = ref_labels
        detector = DriftDetector(ref)
        report = detector.generate_report(cur_features, cur_labels)
        assert report.label_drift is not None
        assert report.label_drift.is_drift is True
        assert report.retraining_recommended is True

    def test_report_sample_counts(self):
        rng = np.random.default_rng(15)
        ref = pd.DataFrame({"x": rng.normal(0.0, 1.0, 300)})
        cur = pd.DataFrame({"x": rng.normal(0.0, 1.0, 150)})
        detector = DriftDetector(ref)
        report = detector.generate_report(cur)
        assert report.n_reference_samples == 300
        assert report.n_current_samples == 150

    def test_report_summary_non_empty(self):
         rng = np.random.default_rng(16)
         ref = pd.DataFrame({"x": rng.normal(0.0, 1.0, 200)})
         cur = pd.DataFrame({"x": rng.normal(0.0, 1.0, 200)})
         detector = DriftDetector(ref)
         report =detector.generate_report(cur)
         assert len(report.summary) > 0

# mean_shift_pct

class TestMeanShiftPct:
     def test_positive_shift(self):
          rng = np.random.default_rng(17)
          ref = pd.DataFrame({"x": rng.normal(10.0, 0.1, 1000)})
          cur = pd.DataFrame({"x": rng.normal(12.0, 0.1, 1000)})
          detector = DriftDetector(ref)
          results = detector.check_feature_drift(cur, features=["x"])
          psi = next(r for r in results if r.method == "PSI")
          assert psi.mean_shift_pct == pytest.approx(20.0, abs=5.0)
          assert psi.current_mean < psi.reference_mean