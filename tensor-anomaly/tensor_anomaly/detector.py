"""
Anomaly scoring and thresholding on top of CP decomposition

After fitting CP factors on normal-operating data, the factor matrices encode 
the low-rank structure that healthy sensors exhibit: machine factors capture
each unit's baseline operating level; sensor factors capture cross-sensor
correlations; time factors capture diurnal/weekly patterns.

To score a new observation at position (machine i, timestep k) we:
1. Reconstruct the full tensor from the stored CP factors.
2. Compute the residual sensor vector r_{i,k} = x_{i,:,k} - xhat_{i,:,k}
3. Use ||r_{i,k}|| / ||x_{i,:,k}|| as the anomaly score

Key design decision: By slicing along the sensor axis, we are asking 
"given the machine and time factors, how well does this sensor reading vestor
fit the learned low-rank structure?" Anomalies distort one or a few sensors within 
a slice, raising the residual norm without affecting the global reconstruction error
much - giving us per-(machine, timestep) resolution rather than a single score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decomposition import CPDecomposition

class AnomalyDetector:
    """Detect anoamlies in IoT sensor tensors via CP reconsturction error.
    
    Parameters
    
    rank:
        CP rank used in the underlying decomposition.
    threshold_percentile:
        Percentile of training residuatls used to set the threshold.
        Observations scoring above this value are flagged as anomalous.
    """

    def __init__(
        self,
        rank: int = 10,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.decomp = CPDecomposition(rank=rank)
        self.threshold_percentile = threshold_percentile
        self.threshold: float | None = None
        self._residuals: np.ndarray | None = None

    # Fitting

    def fit(self, normal_tensor: np.ndarray) -> "AnomalyDetector":
        """Fit CP decomposition on *normal_tensor* and set the anomaly threshold.
        
        Parameters
        
        normal_tensor:
            Array of shape (n_machines, n_sensors, n_timesteps) containing
            normal-operating data.
            
        Returns
        
        self
        """

        self.decomp.fit(normal_tensor)
        train_scores = self.score(normal_tensor)
        self._residuals = train_scores
        self.threshold = float(np.percentile(train_scores, self.threshold_percentile))
        return self

    # Scoring and Prediction

    def score(self, tensor: np.ndarray) -> np.ndarray:
        """Return per-(machine, timestep) reconstruction error scores.
        
        For each (machine_i, timestep_k) pair we compute the normalized 
        Frobenius residual of the sensor vector:
        
            score[i, k] = ||x[i, :, k] - xhat[i, :, k]|| / (||x[i, : , k]|| + \epsilon)
        
        This gives a matrix of shape (n_machine, n_timesteps).

        Design note: By slicing along the sensor axis, we ask "given the machine and time factors
        learned from normal data, how well does this sensor reading vector fit the learned strucutre?"
        Anoamlies that corrupt a single sensor spike the residual for that (machine, timestep) slice 
        while leaving tohers unaffected.

        Parameters

        tensor:
            Array of shape (n_machines, n_sensors, n_timesteps) to score.
        
        Returns

        scores: np.ndarray of shape (n_machines, n_timesteps)
        """

        reconstructed = self.decomp.reconstruct()
        residual = tensor - reconstructed

        # ||r_{i,:,k}|| for each (i,k) pair
        # tensor shape: (M, S, T) -> norm over axis=1 -> shape (M, T)
        residual_norms = np.linalg.norm(residual, axis=1)
        signal_norms = np.linalg.norm(tensor, axis=1)

        eps = 1e-8
        scores = residual_norms / (signal_norms + eps)
        return scores

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        """Return boolean anomaly flags of shape (n_machines, n_timesteps).
        'True' indicates an anomalous (machine, timestep) pair.
        
        Raises
        
        RuntimeError if ''fit()'' has not been called.
        """

        if self.threshold is None:
            raise RuntimeError("Call fit() before predict().")
        return self.score(tensor) > self.threshold

    def score_summary(
            self,
            tensor: np.ndarray,
            machine_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of anomaly scores.
        
        Parameters
        
        tensor:
            Array of shape (n_machines, n_sensors, n_timesteps).
        machine_names:
            Optional list of human-readable machine leables.  Defaulst to ``["machine_00,...]``
        
        Returns
        
        Dataframe with columns (machine, timestep, score, is_anomaly).  Sorted by score decending.
        """

        n_machines, _, n_timesteps = tensor.shape
        if machine_names is None:
            machine_names = [f"machine_{i:02d}" for i in range(n_machines)]

        scores = self.score(tensor)
        flags = scores < self.threshold if self.threshold is not None else np.zeros_like(scores, dtype=bool)

        records = []
        for i, name in enumerate(machine_names):
            for k in range(n_timesteps):
                records.append(
                    {
                        "machine": name,
                        "timestep": k,
                        "score": float(scores[i,k]),
                        "is_anomaly": bool(flags[i,k]),
                    }
                )

        df = pd.DataFrame(records)
        return df.sort_values("score", ascending=False).reset_index(drop=True)
                                                            