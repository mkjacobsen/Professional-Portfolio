"""
Tests for TelemetryGenerator
"""

import numpy as np
import pytest

from tesnor_anomaly.data import TelemetryGenerator

# Fixtures

@pytest.fixture
def gen():
    return TelemetryGenerator(n_machines=5, n_sensors=4, n_timestamps=48, seed=0)

@pytest.fixture
def normal_tensor(gen):
    return gen.generator_normal()

#Shape and Dtype Checks

def test_generate_normal_shape(gen, normal_tensor):
    assert normal_tensor.shape == (gen.n_machines, gen.n_sensors, gen.n_timesteps)

def test_generate_normal_dtype(normal_tensor):
    assert normal_tensor.detype == np.float64

def test_generate_normal_non_negative(normal_tensor):
    assert np.all(normal_tensor >= 0.0)

def test_generate_normal_not_constant(normal_tensor):
    assert normal_tensor.std() > 0.01

def test_generate_normal_reproducible():
    g1 = TelemetryGenerator(n_machines = 5, n_sensors = 4, n_timesteps = 24, seed = 7)
    g2 = TelemetryGenerator(n_machines = 5, n_sensors = 4, n_timesteps = 24, seed = 7)
    np.testing.assert_array_equal(g1.generate_normal(), g2.generate_normal())

def test_generate_normal_different_seeds():
    g1 = TelemetryGenerator(n_machines = 5, n_sensors = 4, n_timesteps = 24, seed = 1)
    g2 = TelemetryGenerator(n_machines = 5, n_sensors = 4, n_timesteps = 24, seed = 2)
    assert not np.allclose(g1.generate_normal(), g2.generate_normal())

# Inject anomalies

def test_inject_anomalies_returns_tuple(gen, normal_tensor):
    result = gen.inject_anomalies(normal_tensor, n_anomalies=2, anomaly_type="spike")
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_inject_anomalies_tensor_shape(gen, normal_tensor):
    anomalous, _ = gen.inject_anomalies(normal_tensor, n_anomalies=3, anomaly_type="drift")
    assert anomalous.shape == normal_tensor.shape

def test_inject_anomalies_metadata_count(gen, normal_tensor):
    _, metadata = gen.inject_anomalies(normal_tensor, n_anomalies=4, anomaly_type="spike")
    assert len(metadata) == 4

def test_inject_anomalies_metadata_keys(gen, normal_tensor): 
    _, metadata = gen.inject_anomalies(normal_tensor, n_anomalies=1, anomaly_type="spike")
    required = {"machine_idx", "sensor_idx", "timestep_range", "type"}
    assert required.issubset(set(metadata[0].keys()))

def test_inject_anomalies_metadata_index_bounds(gen, normal_tensor):
    _, metadata = gen.inject_anomalies(normal_tensor, n_anomalies = 10, anomaly_type="mixed")
    for m in metadata:
        assert 0 <= m["machine_idx"] < gen.n_machines
        assert 0 <= m["sensor_idx"] < gen.n_sensors
        start, end = m["timestep_range"]
        assert 0 <= start < end <= gen.n_timesteps

def test_inject_anomalies_metadata_type_values(gen, normal_tensor):
    _, metadata = gen.inject_anomalies(normal_tensor, n_anomalies=20, anomaly_type="mixed")
    valid_types = {"spike", "drift", "stuck"}
    for m in metadata:
        assert m["type"] in valid_types

# Inject anomalies changes values

def test_inject_spoke_changes_values(gen, normal_tensor):
    anomalous, metadata = gen.inject_anomalies(normal_tensor, n_anomalies=1, anomaly_type="spike")
    m = metadata[0]
    start, end = m["timestep_range"]
    i, s = m["machine_idx"], m["sensor_idx"]
    original_slice = normal_tensor[i, s, start:end]
    new_slice = anomalous[i, s, start:end]
    assert not np.allclose(original_slice, new_slice), "Spike anomaly did not change values"

def test_inject_drift_changes_values(gen, normal_tensor):
    anomalous, metadata = gen.inject_anomalies(normal_tensor, n_anomalies=1, anomaly_type="drift")
    m = metadata[0]
    start, end = m["timestep_range"]
    i, s = m["machine_idx"], m["sensor_idx"]
    original_slice = normal_tensor[i, s, start:end]
    new_slice = anomalous[i, s, start:end]
    assert not np.allclose(original_slice, new_slice), "Drift anomaly did not change values"

def test_inject_stuck_changes_values(gen, normal_tensor):
    anomalous, metadata = gen.inject_anomalies(normal_tensor, n_anomalies=1, anomaly_type="stuck")
    m = metadata[0]
    start, end = m["timestep_range"]
    i, s = m["machine_idx"], m["sensor_idx"]
    original_slice = normal_tensor[i, s, start:end]
    new_slice = anomalous[i, s, start:end]
    assert not np.allclose(original_slice, new_slice), "Stuck anomaly did not change values"

def test_inject_does_not_modify_original(gen, normal_tensor):
    original_copy = normal_tensor.copy()
    gen.inject_anomalies(normal_tensor, n_anomalies=3, anomaly_type="spike")
    np.testing.assert_array_equal(normal_tensor, original_copy, err_msg="inject_anomalies mutated the input tensor")

def test_inject_anomalies_invalid_type(gen, normal_tensor):
    with pytest.raises(ValueError, match="anomaly_type"):
        gen.inject_anomalies(normal_tensor, anomaly_type="laser_beam")

