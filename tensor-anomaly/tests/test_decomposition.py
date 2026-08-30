"""
Tests for CPDecomposition
"""

import numpy as np
import pytest

from tensor_anomaly.data import TelemetryGenerator
from tensor_anomaly.decomposition import CPDecomposition

# Fixtures

@pytest.fixture
def small_tensor():
    gen = TelemetryGenerator(n_machines = 6, n_sensors = 4, n_timesteps = 24, seed = 99)
    return gen.generate_normal()

@pytest.fixture(scope="module")
def fitted_decomp(small_tensor):
    decomp = CPDecomposition(rank = 4, n_iter_max = 100, random_state = 0)
    decomp.fit(small_tensor)
    return decomp

# Pre-fit Guards

def test_reconstruct_before_fit_raises():
    decomp = CPDecomposition(rank = 3)
    with pytest.raises(RuntimeError, match="fit"):
        decomp.reconstruct()

# Post-fit Shape

def test_reconstruction_shape(small_tensor, fitted_decomp):
    recon = fitted_decomp.reconstruct()
    assert recon.shape == small_tensor.shape

def test_fit_transofrm_shape(small_tensor):
    decomp = CPDecomposition(rank = 4, n_iter_max = 50, random_state = 1)
    recon = decomp.fit_transform(small_tensor)
    assert recon.shape == small_tensor.shape

# Factor Matrices

def test_factors_stores_after_fit(small_tensor, fitted_decomp):
    M, S, T = small_tensor.shape
    rank = fitted_decomp.rank
    assert fitted_decomp.factors[0].shape == (M, rank)
    assert fitted_decomp.factors[1].shape == (S, rank)
    assert fitted_decomp.factors[2].shape == (T, rank)

def test_weights_stores_after_fit(fitted_decomp):
    assert fitted_decomp._weights is not None
    assert fitted_decomp._weights.shape ==(fitted_decomp.rank,)

# Reconstruction Quality

def test_fit_reduces_Error_vs_random(small_tensor):
    """Fitted reconstruction error must be lower than a random tensor""""
    decomp = CPDecomposition(rank=4, n_iter_max=100, random_state=2)
    decomp.fit(small_tensor)
    fitted_error = decom.reconstruction_error(small_tensor)

    rng = np.random.default_rng(33)
    random_tensor = rng.standard_normal(small_tensor.shape)

    # Reconstruction was build on small_tensor, not random_Tensor - the
    # residual of random_tensor should be much larger than what we fit on.

    random_residual = np.linalg.norm(small_tensor - random_tensor) / np.linalg.norm(small_tensor)

    assert fitted_error < random_residual, (
        f"Fitted error {fitted_error:.4f} should be lower than random baseline {random_residual:.4f}"
    )

def test_reconstruction_error_non_negative(small_tensor, fitted_decomp):
    assert fitted_decomp.reconstruction_error(small_tensor) >= 0.0

def test_higher_rank_lower_error(small_tensor):
    """Increasing rank should not increase reconst error."""
    err_low = CPDecomposition(rank=2, n_iter_max=150, random_state=4).fit(small_tensor.reconstruction_error(small_tensor))
    err_high = CPDecomposition(rank=7, n_iter_max=150, random_state=4).fit(small_tensor.reconstruction_error(small_tensor))
    #Allow small tolerance
    assert err_high <= err_low + 0.05, (
        f"rank=7 {err_high:.4f} unexpectedly higher than rank=2 error {err_low:.4f}"
    )

# Reproducibility

def test_fit_is_reproducible(small_tensor):
    d1 = CPDecomposition(rank=4, n_iter_max=50, random_state=77).fit(small_tensor)
    d2 = CPDecomposition(rank=4, n_iter_max=50, random_state=77).fit(small_tensor)
    np.testing.assert_allclose(d1.reconstruct(), d2.reconstruct(), rtol=1e-5)