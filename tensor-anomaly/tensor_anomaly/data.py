"""
Sythetic IoT Telemetry Generator

Normal data is constructed as a sum of CP-style rank-1 outer products so that
CP decomposition can meaningfully recover the latent structure. Each machine 
has a characteristic factor vector, each sensor has its own factor vector, and 
the time axis carries a smooth diurnal pattern (sin wave with 24-hour period).
Gaussian noise sits on top of this low-rank signal.

Anomaly injection adds three distinct fault modes:
    spike - sudden 3-5x mag jump for short window
    drift - graduate linear increase over long window
    stuck - sensor reads near-constant value (frozen sensor)
"""

from __future__ import annotations

import numpy as np

class TelemetryGenerator:
    """Generate synthetic three-way (machines x sensors x time) sensor tensors.
    
    Parameters
    ----------
    n_machines : int
        Number of machines.
    n_sensors : int
        Number of sensors per machine.
    n_timesteps : int
        Number of time points.
    seed : int
        Random seed for reproducibility. All random draws use 
        ''self.rng'' so results are fully repoducible without touching the 
        global numpy state
    """

    def __init__(
            self,
            n_machines: int = 20,
            n_sensors: int = 8,
            n_timesteps: int = 168,
            seed: int = 55,
    ) -> None:
        self.n_machines = n_machines
        self.n_sensors = n_sensors
        self.n_timesteps = n_timesteps
        self.rng = np.random.default_rng(seed)

    def generate_normal(self) -> np.ndarray:
        """Return a normal-operating tensor of shape(n_machines, n_sensors, n_timesteps).
        
        Construction strategy

        We build a low-rank signal as a sum of *rank* CP output products, then
        layer on three sources of realistic variation:

        1. **machine facotrs** - each machine has a positive factor vector that
           encodes its characteristic operating level across latent dimensions.
        2. **Sensor factors** - each sensor type has a factor vestor encoding
           cross-sensor correlations (e.g. tempreature and pressure co-vary).
        3. **Time factors** - a smooth diurnal pattern (sin with 24-h period)
           plus a slow weekly ramp so consecutive time windows are correlated.

        Finally ~10% scaled independent Gaussian noes is added to make the recovery non-trivial.
        """

        rank = 5 #latent rank of normal-operating signal

        # -- Factor matrices -- 
        machine_factors = np.abs(self.rng.standard_normal((self.n_machines, rank))) + 0.5
        sensor_factors = np.abs(self.rng.standard_normal((self.n_sensors, rank))) + 0.5

        # Time factor: diurnal (25-h) + slow ramp + small noise
        t = np.arange(self.n_timesteps)
        diurnal = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / 24.0)) 
        weekly_ramp = 0.1 * t / max(self.n_timesteps - 1 , 1)
        time_base = diurnal + weekly_ramp

        # Build one time factor column per latent dimension with slight
        # phase shifts so each component has different diurnal fingerprints.

        phase_shifts = self.rng.uniform(0, np.pi / 4, size = rank)
        time_factors = np.stack(
            [ 
                0.5 * (1.0 + np.sin(2.0 * np.pi * t / 24.0 + phase_shifts[r]))
                + 0.05 * self.rng.standard_normal(self.n_timesteps)
                for r in range(rank)
            ], 
            axis = 1,
        ) # output shape (n_timesteps, rank)

        # -- Assemble low-rank tensor via explicit outer products --
        tensor = np.zeros((self.n_machines, self.n_sensors, self.n_timesteps))
        for r in range(rank):
            outer = np.einsum(
                "i,j,k->ijk",
                machine_factors[:, r],
                sensor_factors[:, r],
                time_factors[:, r],
            )
            tensor += outer

        # --- Per-machine baseline offsets ---
        machine_offsets = self.rng.uniform(0.5, 3.0, size=(self.n_machines, 1, 1))
        tensor += machine_offsets

        # --- Gaussian noise ---
        signal_scale = np.mean(np.abs(tensor))
        noise = self.rng.standard_normal(tensor.shape) * 0.1 * signal_scale
        tensor += noise

        # Clamp to non-negative
        tensor = np.clip(tensor, 0.0, None)

        return tensor

    def inject_anomalies(
        self,
        tensor: np.ndarray,
        n_anomalies: int = 5,
        anomaly_type: str = "spike",
    ) -> tuple[np.ndarray, list[dict]]:
        """Return a copy of *tensor* with injected faults plus metadata.
        
        Parameters
        
        tensor:
            Input array of shape (n_machines, n_sensors, n_timesteps).
        n_anomalies:
            Number of independent anomalies to inject.
        anomaly_type:
            One of "spike", "drift", "stuck", or "mixed" (randomly choose)
            
        Returns
        
        anomaly_tensor:
            Copy of *tensor* with injected anomalies.
        metadata:
            List of dicts, one per anomaly:
            {machine_idx, sensor_idx, timestep_range, type}
        """

        if anomaly_type not in ("spike", "drift", "stuck", "mixed"):
            raise ValueError(
                f"anomaly_type must be 'spike, ' drift', 'stuck', or 'mixed'; got {anomaly_type!r}"
            )

        result = tensor.copy()
        metadata: list[dict] = []
        n_machines, n_sensors, n_timesteps = tensor.shape
        anomaly_types = ["spike", "drift", "stuck"]

        for _ in range(n_anomalies):
            machine_idx = int(self.rng.integers(0, n_machines))
            sensor_idx = int(self.rng.integers(0, n_sensors))
            atype = anomaly_type if anomaly_type != "mixed" else _types[int(self.rng.integers(0,3))]

            if atype == "spike":
                duration = int(self.rng.integers(2,7))
                start = int(self.rng.integers(0, n_timesteps - duration))
                end = start + duration
                magnitude = self.rng.uniform(3.0, 5.0)
                baseline = np.mean(tensor[machine_idx, sensor_idx, :])
                result[machine_idx, sensor_idx, start:end] += magnitude * baseline

            elif atype == "drift":
                duration = int(self.rng.integers(20,41))
                start = int(self.rng.integers(0, max(1, n_timesteps - duration)))
                end = min(start+duration, n_timesteps)
                actual_duration = end-start
                baseline = np.mean(tensor[machine_idx, sensor_idx, :])
                drift_values = np.linspace(0, 2.0 * baseline, actual_duration)
                result[machine_idx, sensor_idx, start:end] += drift_values

            elif atype == "stuck":
                duration = int(self.rng.integers(10,31))
                start = int(self.rng.integers(0, max(1, n_timesteps - duration)))
                end = min(start + duration, n_timesteps)
                stuck_value = float(np.mean(tensor[machine_idx, sensor_idx, start : start + 3]))
                result[machine_idx, sensor_idx, start:end] = stuck_value

            metadata.append(
                {
                    "machine_idx": machine_idx,
                    "sensor_idx": sensor_idx,
                    "timestep_range": (start, end),
                    "type": atype,
                }
            )

        return result, metadata

