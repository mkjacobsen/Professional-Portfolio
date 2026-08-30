# tensor-anomaly

Anomaly detection for IoT and manufacturing sensor telemetry using **CP (PARAFAC) tensor decomposition**.

## Project Overview

tensor-anomaly detects faults in multi-machine, multi-sensor industrial systems without requiring labelled failure data.  It targets the class of problem where you havea continuous telemetry from a fleet of machines -each reporting temperature, pressure, vibration, flow rate, and similar readings over time -and you want to know when and where things go wrong.  The intendedusers are ML engineers and data scientists working on predictive matinenance, process monitoring, or any domain where sensor readings from similar assets are collected at regular intervals. 

The library models normal operating data as a three-way tensor indexed by machiens, sensor types, and time windows.  This three-dimensional represnetation is not merely a ntional choice: it preserves the joint structure across all three axes simultaneously, something that is lost when you flatten the data into a matrix.  CP decomposition factorixes that tensor into a compact set of factor matrices that encode the low-rank correlation structure of healty operation.  A model trained on only normal data is sufficient; no fault labels are required.

The core diagnostic signal is reconstruction error.  After fitting on normal data, the factor matrices describe what healthy sensor readings look like acorss machiens and over time.  When a new obervation is presented - a bearing beginning to fail, a sensor freezing at a constant value, a pump slowly drifting out of range - the affected readins no longer conform to the leanred low-rank structure, and the normalized reconstruction error for that specific (machine, timestep) slice rises sharply.  The threshold for flagging anomalies is set automatically at a configurable perentile of the training-data error distribution.

## The Technique CP Decomposition for Anomaly Detection

**What is a tensor in this context** A tensor is a multi-dimensional array.  Here the data lives naturally in three dimensions: machiens (axis 0), sensor types (axis 1), and time windows (axis 2).  A single element T[i, j, k] is the reading of sendor j on machine i at time k.  Stacking this as a matrix by unfolding any one axis would discard the joint structure along the others; keeping it as a tensor allows the model to capture how machine identity, sensor type, and time all interact simultaneously. 

**What CP decomposition does** CP (Canonical Polyadic, also called PARAFAC) decomposition factorized the tensor into a sum of rank-1 outer products:

T \approx sum_r (a_r \cross b_r \cross c_r)

Each term is the outer product of three vectors: a_r (length n_machines) captures machine-level variation for latent component r; b_r (length n_sensors) captures the sensor correlation pattern for that component; c_r (length n_timesteps) captures the time signature.  The result is three factor matrices - A (machines x rank), B (sensors x rank), C (time x rank) - that together describe the dominant correlated patterns in the training data.  Fitting is done via Alternating Least Squares (ALS) as implemented in tensorly.

**How reconstruction error signals anomalies.**. Once the factor matrices are fixes, any new observation can be projected back through them to produce a reconstruction.  For a healthy reading, the residual 'x - x_hat' will be small because the observation conforms to the same low-rank structure the model learned.  An amoalous reading breaks that structure: a spike, a drift, or a frozen sensor value is not explained by the learned factors, so the residual is large.  The normalized residual norm '||x - x_hat|| / ||x||' serves as the anomaly score.

**Why per-slice scoring gives spatial localization.** Rather than computing a single reconstruction error for the entire tensor, the detector computes a separate score for each (machine, timestep) pair by slicing along the sensor axis.  The slice 'T[i, :, k]' is the vector of all sensor readings for machine i at time k.  The question we are interested in is: given the machine and time factors the model has learned, how well does this particular sensor vector fit?  A fault on one machine at one point in time corrupts only a handful of these slices.  Scoring them independently means the output is a matrix of shape (n_machines, n_timesteps) rather than a single number, giving the operator immediate spatial and temporal localization of the fault. 

**Why is this better than per-sensor thresholds or PCA?** A simple threshold on each sensor individually will miss faults that are only detectable from the relationship between sensors: a temperature reading that is normal on its own but is anomalous given the pressure and flow readings.  PCA applied to a reshaped matrix captures pairwise correlations within a single snapshot but treats teach time window independently, discarding the temporal structure.  CP decomposition captures three-way interactions - across machines, sensors, and time - simultaneously, which is precisely the structure that healthy multi-machine industiral systems exhibit.

## Architecture

```
TelemetryGenerator -> CPDecomposition -> AnomalyDetector -> viz
    data.py             decomposition.py   detector.py      vis.py
```

**data.py - TelemetryGenerator** Constructs synthetic three-way tensors that have genuine low-rank structure, making the CP recovery problem meaningful.  Normal data is built as a sum of five rank-1 outer products with machine factors, sensor factors, and diurnal time factors, plus Gaussian noise at roughtly 10% of base signal magnitude.  The `inject_naomalies` method adds three distinct fault modes - spike, drift, and frozen/stuck - into a copy of a normal tensor and returns metadata recording exactly where each fault was placed.  This metadata is used in tests to verify spatial localization.

**decomposition.py - CPDecomposition** A thin wrapper around tensorly's parafac function.  It handles fitting the ALS algorithm, storing the resulting factor matrices, reconstructing the full tensor from those factors, and computing the global relative frobenius reconstruction error.  The `rank` parameter is the primary knob; see the Rank Selection section for guidance.  The tensorly backend is pinned to numpy at import time so behavior is deterministic regardless of whether PyTorch of JAX is installed.

**detector.py - AnomalyDetector** The main entrypoint for most users.  This owns a `CPDecomposition` instance, fits it on normal training data, sets a percentile-based anomaly threshold from the training residuals, and exposes `score`, `predict`, and `score_summary` for inference.  The score computation slices the tensor along the sensor axis described above.  `score_summary` returns a tidy pandas DataFrame sorted by score descending, reading for inpsection or downstream alerting logic. 

**viz.py - Visualization helpers** Three matplotlib functsion, each returning `(fig, ax)` so the caller decides whether to display or save.  `plot_sensor_timeseries` shows all sensor readings for a single machine over time; `plot_anomaly_heatmap` shows the full score matrix as a color heatmap; `plot_reconstruction_error` overlays the training and test error distributions as histograms with the threshold drawn as a vertical line.  None of the functions have side effects - they do not call plt.show() or write files. 

## Quickstart

```bash
pip intall -e .
python exmples/detect_anomalies.py
```

The script runs through seven steps and prints a summary to stdout.  Expected terminal output includes:
- Training tensor shape `(20, 8, 168)` and value statistics
- A table of five injected anomaly locations with machine index, sensor index, and timestep range
- The fitted anomaly threshold (a float around `0.08` to `0.15` depending on rank)
- The fraction of test slices flagged as anomalous
- A table of the top flagged (machine, timestep) pairs with their scores
- A cross-reference showing the max score and number of flagged timesteps within each injected fault window
- A reconstruction error comparison table (mean and 95th percentile, train vs test)

Three plots are saved to `./output/`:

| File | Content |
|---|---|
| `sensor_timeseries.png` | All sensor readings for the machine with the highest anomaly score |
| `anomaly_heatmap.png` | Score heatmap across all 20 machiens and 168 timesteps |
| `reconstruction_error_dist.png` | Overlaid histograms of training vs test per-slice error with threshold line |

## API Reference: TelemetryGenerator
```python
from tensor_anomaly.data import TelemetryGenerator
```

### '__init__'

```python
TelemetryGenerator(
    n_machines: int = 20,
    n_sensors: int = 8,
    n_timesteps: int = 168,
    seed: int = 42,
)
```

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `n_machines` | `int` | `20` | Number of distinct machines (first tensor mode). |
| `n_sensors` | `int` | `8` | Number of sensor types per machine (second tensor mode). |
| `n_timesteps` | `int` | `168` | Length of the time axis.  Default of 168 is one week of hourly readings. |
| `seed` | `int` | `42` | Seed for the internal `numpy.random.Generator`. All random draws go through `self.rng`, so results are fully reproducible without touching global numpy state. |

### `generate_normal() -> np.ndarray`

Returns a tensor of shape `(n_machines, n_sensors, n_timesteps)` representing normal operating data. The values are non-negative floats.  The tensor is constructed as a sum of five rank-1 outer products (machine factors x sensor factors x time factors) plus machine-level baseline offsets and Gaussian noise at approximately 10% of the signal magnitude.  The time factors include a diurnal 24-hour sine pattern with per-component phase shifts and a slow weekly ramp, so consecutive timestamps are correlated.

### `inject_anomalies(tensor, n_anomalies, anomaly_type) -> tuple[np.ndarray, list[dict]]`

```python
inject_anomalies(
    tensor: np.ndarray,
    n_anomalies: int = 5,
    anomaly_type: str = "spike",
) -> tuple[np.ndarray, list[dict]]
```

Returns a copy of `tensor` with faults injected and a metadata list.  The original tensor is not modified.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- | 
| `tensor` | `np.ndarray` | - | Input array of shape `(n_machiens, n_sensors, n_timesteps)`. |
| `n_anomalies` | `int` | 5 | Number of independent anomalies to inject.  Each is placed on a randomly selected (machine, sensor) pair |
| `anomaly_type` | `str` | `"spike"` | One of `"spike"`, `"drift"`, `"stuck"`, or `"mixed"`.  With `"mixed"`, each anomaly is assigned a type uniformly at random from the three base types. |

The returned metadata is a list of dicts, one per injected anomaly, with keys:

| Key | Type | Description |
| --- | ---- | ----------- |
| `machine_idx` | `int` | Index of affected machine |
| `sensor_idx` | `int` | Index of affected sensor |
| `timestep_range` | `tuple[int, int]` | `(start, end)` half-open interval of affected timesteps |
| `type` | `str` | The anomaly type that was applied |

**Anomaly Types**

- **spike** - A sudden magnitude jump applied to a short contiguous window of 2 to 6 timesteps.  The added value is 3 to 5 times the sensor's mean baseline.  This simulated impulsive events, such as electrical transients or mechanical impacts.
- **drift** - A gradual linear increase applied over a longer window of 20 to 40 timesteps.  The drift ramps from zero to 2 times the sensor's mean baseline over the window duration.  This simulates slow degradation such as bearing wear, fouling, or calibration drift.
- **stuck** - The sensor is set to a near-constant value for a window of 10 to 30 timesteps.  The stuck value is the mean of the first three readings in the window, simulating a frozen or saturated sensor.

Raises `ValueError` if `anomaly_type` is not one of the four valid values.

---

## API Reference: CPDecomposition

```python
from tensor_anomaly.decomposition import CPDecomposition
```

### `__init__`

```python
CPDecomposition (
    rank: int = 10,
    n_iter_max: int - 200,
    random_state: int = 42,
)
```

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `rank` | `int` | `10` | Number of CP components. This is the primary hyperparameter. See the Rank Selection section. |
| `n_iter_max` | `int` | `200` | Maximum number of ALS iterations passed to tensorly's parafac. | 
| `random_state` | `int` | `42` | Random seed for tensorly's factor initialization. |

After fitting, the decomposition stores:
- `self.factors`: list of three numpy arrays - `[A (n_ machines × rank), B (n_sensors × rank), C (n_timesteps × rank)]`
- `self._weights` : 1-D array of component weights (length `rank`), since tensorly normalizes factors by default.

### Methods

**`fit(tensor: np.ndarray) -> CPDecomposition`**

Fits the CP decomposition on `tensor` of shape `(n_machines, n_sensors, n_timesteps)` using ALS. Store the resulting factor matrices and weights.  Returns `self` for method chaining.

**`reconstruct(tensor_shape=None) -> np.ndarray`**

Reconstructs the full tensor from the stored factor matrices.  Returns an array with the same shape as the training tensor.  The `tensor_shape` parameter is accepted for API symmetry, but is unused;  shape is inferred from the stored factors.  Raises `RuntimeError` if called before `fit`.

**`fit_transform(tensor: np.ndarray) -> float`**

Computes the relative Frobenius reconstruction error: `||tensor - reconstructed|| / ||tensor||`. Returns a float in `[0, \inf)`. A value of `0.0` means perfect reconstruction. This is a global scalar; for per-slice scoring use `AnomalyDetector.score()`. 

`rank` is the key hyperparameter and directly controls the fidelity of the normal-data model;  see the Rank Selection section for guidance on choosing it.

----

## API Reference : Anomaly Detector

```python
from tensor_anomaly.detector import AnomalyDetector
```

### `__init__`

```python
AnomalyDetector(
    rank: int = 10,
    threshold_percentile: float = 95.0,
)
```

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `rank` | `int` | `10` | CP rank passed to the internal `CPDecomposition` |
| `threshold_percentile` | `float` | `95.0` | Percentile of traning-set per-slice scores used to set the anomaly threshold.  Observations scoring above this threshold are flagged as anomalous.  At the default of 95.0, approximately 5% of training slices will be flagged, setting the expected false-positive rate on in-distribution data. |

### Methods

**`fit(normal_tensor): np.ndarray) -> AnomalyDetector`***

Fits the CP decomposition on `normal_tensor` of shape `(n_machines, n_sensors, n_timesteps)`. Computes per-slice scores on the training data and sets `self.threshold` at the configured percentile.  Returns `self`. This method must be called before `score`, `predict`, or `score_summary`.

**score (tensor: np.ndarray) -> np.ndarray'**

```python
score (tensor: np,ndarray)-> np.ndarrayshape (nmachines, ntimesteps)
```

Returns a float array of shape `(n_machines, n_timesteps)`. Each entry `scores[i,k]` is the normalized reconstruction error for machine `i` at timestep `k`:

```
score[i, k] = ||x[i, :, k] - x_hat[i, : ,k]|| / (||x[i, :, k]|| + \epsilon)
```

where the norm is taken over the sensor axis and `epsilon = 1e-8` prevents division by zero.  All values are non-negative.  Can be called on any tensor with the same first and third dimension as the training tensor. 

**`predict(tensor: np.ndarray) -> np.ndarray`**

```python
predict(tensor: np.ndarray) -> np.ndarray # shape (n_machines, n_timesteps), dtype bool
```

Returns a boolean array of shape `(n_machines, n_timesteps)`. `True` indicates an anomalous (machine, timestep) pair - specifically, one whose score exceeds `self.threshold`. Raises `RuntimeError` if called before `fit`. 

**`score_summary(tensor, machine_names=None) -> pd.DataFrame`**

```python
score_summary(
    tensor: np.ndarray,
    machine_names: list[str] | None = None,
) -> pd.DataFrame
```

Returns a tidy DataFrame with one row per (machine,timestep) pair, sorted descending by score. Column:

| Column | Type | Description |
| ------ | ---- | ----------- | 
| `machine` | `str` | Machine label.  Defaults to "machine_00".... |
| `timestep` | `int` | Time index |
| `score` | `float` | Normalized reconstruction error for this (machine, timestep) slice |
| `is_anomaly` | `bool` | Whether the score exceeds the fitted threshold |

The `machine_names` parameter accepts an optional list of human-readable labels whose length must equal `n_machines`. If omitted, labels default to zero-padded integers.

---

## API Reference: viz

```python
from tensor_anomaly.viz import (
    plot_sensor_timeseries,
    plot_anomaly_heatmap,
    plot_reconstruction_error,
)
```

All three functions return `(fig, ax)` - a matplotlib `Figure` and `Axes` - and have no side effects.  The caller is responsible for calling `fig.savefig()` or `plt.show()`.

### `plot_sensor_timeseries`

```python
plot_sensor_timeseries(
    tensor: np.ndarray,
    machine_idx: int = 0,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]
```

Draws a line plot of all sensor readings for a single machine over the full time axis.  Each sensor is shown as a separate colored line using the tab10 colormap.  The x-axis is labelled in hours.  Useful for visually identifying which sensor is behaving anomalously and approximately when the fault occurs.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- | 
| `tensor` | `np.ndarray` | - | Array of shape `(n_machines, n_sensors, n_timesteps)` |
| `machine_idx` | `int` | `0` | Index of the machine to plot |
| `title` | `str \| None` | `None` | Figure title.  Defaults to "Machine (machine_idx) - All Sensors" |

### `plot_anomaly_heatmap`

```python
plot_anomaly_heatmap(
    scores: np.ndarray,
    machine_names: list[str] | None = None,
    title: str = "Anomaly Scorew Heatmap (machines x time)",
) 0> tuple[plt.Figure, plt.Axes]
```

Renders the `(n_machines, n_timesteps)` score matrix as a heatmap using the `YlOrRd` colormap. Machiens are on the y-axis; time is on the x-axis with major tick marks every 24 hours.  A colorbar labels the normalized reconstruction error scale.  High-score cells appear in red, making anomalous (machine, time) regions immediately visible across the entire fleet.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `scores` | `np.ndarray` | - | Array of shape `(n_machines, n_sensors, n_timesteps)` as returned by `AnomalyDetector.score()` |
| `machine_names` | `list[str] \| None` | `None` | Optional y-axis labels. Defaults to "machine_{machine_idx}" |
| `title` | `str` | `"Anomaly Score Heatmap (machines x time)"` | Figure title |

