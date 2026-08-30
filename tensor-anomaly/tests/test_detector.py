"""
Tests for AnomalyDetector

Critical Behavior Test: After fit on clean data, tensor with 
known-injected anomalies should yield above-median scores at the injection locations.
Verify location-level signal, not just global detection.
"""

import numpy as np
import pytest
import pandas as pd

from tensor_anomaly.data import TelemetryGenerator
from tensor_anomaly.detector import AnomalyDetector

# Fixtures

@pytest.fixture(scope="module")
def gen():
    return TelemetryGenerator(n_machines=10, n_sensors=6, n_timesteps=72, seed=222)

@pytest.fixture(scope="module")
def normal_tensor(gen):
    return gen.generate_normal()

@pytest.fixture(scope="module")
def fitted_detector(normal_tensor):
    det = AnomalyDetector(rank=5, threshold_percentile=95.0)
    det.fit(normal_tensor)
    return det

# Fit Checks

def test_fit_sets_threshold(fitted_detector):
    assert fitted_detector.threshold is not None
    assert fitted_detector.threshold > 0.0

def test_fit_Stores_resituals(fitted_detector, normal_tensor):
    n_machines, _, n_timesteps = normal_tensor.shape
    assert fitted_detector._residuals is not None
    assert fitted_detector._residuals.shape == (n_machines, n_timesteps)

def test_fit_returns_self(normal_tensor):
    det = AnomalyDetector(rank=3)
    result = det.fit(normal_tensor)
    assert result is det

# Score Shape and Value

def test_score_shape(fitted_detector, normal_tensor):
    scores = fitted_detector.score(normal_tensor)
    n_machines, _, n_timesteps = normal_tensor.shape
    assert scores.shape == (n_machines, n_timesteps)

def test_score_non_negative(fitted_detector, normal_tensor):
    scores = fitted_detector.score(normal_tensor)
    assert np.all(scores >= 0.0)

def test_predict_shape(fitted_detector, normal_tensor):
    flags = fitted_detector.predict(normal_tensor)
    n_machines, _, n_timesteps = normal_tensor.shape
    assert flags.shape == (n_machines, n_timesteps)
    assert flags.dtype == bool

def test_predict_before_fit_raises(normal_tensor):
    det = AnomalyDetector(rank=3)
    with pytest.raises(RuntimeError, match="fit"):
        det.predict(normal_tensor)

# Score Summary Dataframe

def test_score_summary_columns(fitted_detector, normal_tensor):
    df = fitted_detector.score_summary(normal_tensor)
    for col in ("machine", "timestep", "score", "is_anomaly"):
        assert col in df.columns

def test_score_summary_length(fitted_detector, normal_tensor):
    n_machines, _, n_timesteps = normal_tensor.shape
    df = fitted_detector.score_summary(normal_tensor)
    assert len(df) == n_machines * n_timesteps

def test_score_summary_sorted_descending(fitted_detector, normal_tensor):
    df = fitted_detector.score_summary(normal_tensor)
    assert (df["score"].diff().dropna() <= 0).all()

def test_score_summary_custom_machine_names(fitted_detector, normal_tensor):
    n_machines = normal_tensor.shape[0]
    names = [f"unit_{i}" for i in range(n_machines)]
    df = fitted_detector.score_summary(normal_tensor, machine_names=names)
    assert set(df["machine"].unique()) == set(names)

# Key Behavior Test - Locations Score Above Median

def test_injected_anomaly_locations_score_Above_median(gen):
    """
    Constructs a dedicates normal training set and test set with 5 spike
    anomalies at specific positions.  Then verifies that, for each injected
    anomaly, at least one timestep in the window receives a score above the 
    overall median from the test set. 
    """

    #Use a generator with a fresh seed
    train_gen = TelemetryGenerator(n_machines = 10, n_sensors = 6, n_timesteps = 72, seed=555)
    normal_train = train_gen.generate_normal()

    test_gen = TelemetryGenerator(n_machines = 10, n_sensors = 6, n_timesteps =72, seed = 888)
    normal_test = test_gen.generate_normal()
    anomalous_test, metadata = test_gen.inject_anomalies(
        normal_test, n_anomalies=5, anomaly_type="spike"
    )

    det = AnomalyDetector(rank=5, threshold_percentile=90.0)
    det.fit(normal_train)

    scores = det.score(anomalous_test)
    median_score = float(np.median(scores))

    above_median_count = 0
    for m in metadata:
        i = m["machine_idx"]
        start, end = m["timestep_range"]
        window_scores = scores[i, start:end]
        if np.any(window_scores > median_score):
            above_median_count += 1

    assert above_median_count >= 3, (
        f"Only {above_median_count}/5 injected anomaly windows scored above the median"
        f" (median = {median_score:.4f}). CP reconstruction is not localising faults."
    )

def test_normal_data_has_few_anomalies(fitted_detector, normal_tensor):
    """Training data should have few flagged anomalies"""
    flags = fitted_detector.predict(normal_tensor)
    false_positive_rate = flags.mean()

    assert false_positive_rate <= 0.06, (
        f"False positive rate on trainnig data is {false_positive_rate:.3f}, expected less or equal to 0.06"
    )

