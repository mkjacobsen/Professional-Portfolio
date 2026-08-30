"""
Thin wrapper around tensorly's CP/PARAFAC decomposition.

Uses CP (canonical Polyadic / PARAFAC) decomposition because it gives
a natural low-rank factorization of a three-way tensor:

    T \approx \sum_r a_r \cross b_r \cross c_r

where a_r, b_r, and c_r are vectors for the machine, sensor, and time modes
respectively.  Fitting on normal-operating data captures correlated structure
that healthy sensors exhibit; anomalies show up as points the learned factors
cannot reconstruct well.
"""

from __future__ import annotations
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

# Pin backend explicity for deterministic behavior
# regardless of whether PyTorch or JAX is installed

tl.set_backend("numpy")

class CPDecomposition:
    """CP (PARAFAC) decomposition of a three-way tensor.
    
    Parameters
    
    rank:
        Number of CP components (latent rank)
    n_iter_max: 
        Maximum ALS iterations.
    random_state:
        Seed passed for reproducible initiatlization
    """

    def __init__(
            self,
            rank: int = 10,
            n_iter_max: int = 200,
            random_state = 44,
    ) -> None:
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.factors: list[np.ndarray] | None = None
        self._weights: np.ndarray | None = None
        self._cp_tensor = None

    # Core API
    def fit(self, tensor: np.ndarray) -> "CPDecomposition":
        """
        Fit CP decomposition on *tensor*
        
        Parameters
        
        tensor: 
            Array of shape (n_machines, n_sensors, n_timesteps)
        
        Returns
        
        self
        """

        cp_tensor = parafac(
            tensor,
            rank = self.rank,
            n_iter_max = self.n_iter_max,
            random_state = self.random_state,
            normalize_factors = True,
        )
        self._cp_tensor = cp_tensor
        self._weights = np.array(cp_tensor.weights)
        self.factors = [np.array(f) for f in cp_tensor.factors]
        return self

    def reconstruct(self, tensor_shape: tuple | None = None) -> np.ndarray:
        """
        Reconstruct full tensor from stored factor matrices
        
        Parameters
        
        tensor_shape:
            Unused; kept for API symmetry. Shape inferred from factors
            
        Returns
        
        Reconstructed array with same shape as training tensor
        """

        if self.factors is None:
            raise RuntimeError("Call fit() before reconstruct().")
        return np.array(tl.cp_to_tensor(self._cp_tensor))

    def fit_transform(self, tensor: np.ndarray) -> np.ndarray:
        """Fit on *tensor* and return its reconstruction."""
        self.fit(tensor)
        return self.reconstruct()

    def reconstruction_error(self, tensor: np.ndarray) -> float:
        """Relative reconstruction error (Frobenius norm, normalized by ||tensor||)
        
        Returns

        float in [0, \inf) 0.0 mean perfect reconstruction
        """
        reconstructed = self.reconstruct()
        residual = tensor - reconstructed
        denom = np.linalg.norm(tensor)
        if denom == 0.0:
            return 0.0
        return float(np.linalg.norm(residual) / denom)
