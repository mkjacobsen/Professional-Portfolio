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

### `plot_reconstruction_error`

```python
plot_reconstruction_error(
    normal_errors: np.ndarray,
    test_errors: np.ndarray,
    threshold: float,
    title: str = "Reconstruction Error Distribution: Normal vs Test",
) -> tuple[plt. Figure, plt.Axes]
```

Overlays two normalized histrograms - training-set scores in blue and test-set scores in red - with a vertical dashed line at the anomaly threshold.  If the detector is working well, the test distribution will have a heavier right tail than the training distribution, and anmalous slices will cluster beyond the threshold line.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- | 
| `normal_errors` | `np.ndarray` | - | 1-D array of per-slice scores from the training set (e.g., `detector.score(normal_tensor).ravel()`) |
| `test_errors` | `np.ndarray` | - | 1-D array of per-slice scores from the test set. |
| `threshold` | `float` | - | The fitted anomaly threshold drawn as a vertical line.  Use `detector.threshold`. |
| `title` | `str| `"Reconstruction Error Distribution: Normal vs Test"` | Figure title |

---

## Rank Selection

The CP rank controls the number of latent components and is the most consequential hyperparameter.  A rank that is too low underfits the normal data: the reconstruction is poor even on healthy observations, which inflates the baseline residual and raises the false-positive rate.  A rank that is too high overfits to noise in the training data, meaning the model can reconstruct almost anything, an tru anomalies no longer produce a distinguisable spike in the residual.  In practice, ranks in the 5-15 range work well for IoT data with clear diurnal and weekly periodicity, because such data genuinely has only a handful of dominant latent patterns.

To select rank systematically, fit `CPDecomposition` at several candidate ranks on a held-out normal validation set and plot the global `reconstruction_error` as a function of rank.  Look for the elbow: the point at which adding another component yields diminishing returns in reconstruction quality.  This is the scree-plot method adapted from PCA.  Alternatively, use k-fold cross-validation on the normal data: hold out a fraction of timesteps, fit on the remainder, evaluate reconstruction error on the held-out portion, and average across folds.  Select the trank that minimizes held-out error.  Both approaches require only normal data and give an objective criterion rather than guessing.

---

## Running the Example

`examples/detect_anomalies.py` is a self-contained end-to-end demonstration.  Each step maps to a printed banner.

**Step 1- Generate normal training data.*** Creates a `TelemetryGenerator` with 20 machines, 8 sensors, and 168 timesteps (one week at hourly resolution, seed 42) and calls `generate_normal()`.  Prints the tensor shape and value statistics to confirm the data is non-negative and in a reasonable range.

**Step 2 - Generate test data with 5 injected anomalies.** Creates a second generator with a different seed (99) to produce independent test data.  Calls `inject_anomalies` with `anomaly_type="mixed"` to inject five faults drawn uniformly from spike, drift, and stuck types.  Prints the exact location of each fault (machine index, sensor index, timestep range, type).  Keep this output: it tells you where to look in the subsequent plots.

**Step 3 - Fit AnomalyDetector** Instantiates `AnomalyDetector(rank=10, threshold_percentile=95.0)` and calls `fit(normal_tensor)`. Prints the resulting threshold. The threshold is the 95th percentrile of the training-set per-slice scores; approximately 5% of training slices will exceed it.

**Step 4 - Score test data** Calls `detector.score` on both the anomalous test tensor and the clean training tensor.  Prints the score ranges and the fraction of test slices flagged as anomalous.  The test score maximum should be noticeably higher than the training score maximum; the fraction flagged will typically be in the 5-15% range with five injected faults in a 20 x 168 score matrix.

**Step 5 - Print detected anomalies table** Calls `score_summary` to get the full ranked DataFrame, then prints the top 20 flagged rows.  Below the table, a cross-reference block prints, for each known injection site, the maximum score in the affected window and the count of timesteps within that window that exceeded the threshold.  This is the key diagnostic output: it shows whether the detector is finding the faults where they were actually placed.  If a fault is missed here, either the rank is too high, the threhold too agressive, or the injected anomaly happens to land at a timestep already well-explained by the learned factors.

**Step 6 - Save plots** Saves the three plots described in the Quickstart section to `./output/`. The sensor time-series plot is drawn for the machine wit hthe globally highest anomaly score, which is usually one of the machines that received an injected fault. 

**Step 7 - Reconstruction error comparison** PRints a two-column table of mean score and 95th-percentile score for training and test data, along with the ratio of test mean to train mean.  A ratio noticably above 1.0 confirms that the injected faults are elevating the score distribution.

**Interpreting the anomaly score table** The `score` column is a dimensionless ratio in the range `[0,inf)`. A score of `0.0` is perfect reconstruction; a score near `1.0` means the residual is approximately as large as the original signal, which is severely anomalous.  In a healthy system trained at rank 10, typical scores are in the `0.05` to `0.15` range; injected spikes will commonly reach `0.3` to `0.8` depending on maginutde.  The `is_anomaly` column is `True` for any score above the fitted threshold. Sort by `score` decending and look at the top rows: if the machine indices match the injection metadata from Step 2, the detector is spatially localizing the faults correctly.

##Extending the Detector

**Plugging in a different decomposition method** AnomalyDetector couples to CPDecomposition through two methods: fit (tensor)and reconstruct () . To use Tucker decomposition, NMF, or any other method, implement a class with those two method signatures - fit accepts a 3-D array and returns self; reconstruct takes no arguments and returns an array of the same shape as the training tensor - and substitute it for the CPDecomposition instance in AnomalyDetector._init_ No other changes are required.

**Adding a new anomaly injection type** In data.py, inject_anomalies selects a branch based on the string value of atype. To add a new type, for example burst (repeated short spikes), add it to the _types list, add a corresponding branch in the if/elif chain that modifies result [machine_idx, sensor_idx, start:end], and add "burst" to the valid values in the ValueError check at the top of the method. The metadata dict at the end of each loop iteration will automatically record the new type.

**Adjusting the scoring function for different tensor shapes** The current scoring function slices along axis 1 (sensor axis), which assumes the tensor is organized as (machines x sensors x time). If your tensor has a different axis ordering, change the 'axis-argument in the two np.linalg.norm" calls inside "AnomalyDetector.score'. If you want to score along a different granularity - for example, per-machine averages rather than per- (machine, timestep) pairs - replace the norm-over-axis-1 computation with a norm over the combined sensor and time axes. The threshold-setting logic in fit and the flag logic in predict operate on whatever array call score returns, so they adapt automatically

---
## Running Tests
```bash
pip install -e ". (dev]"
pytest tests/ -v
```

The test suite covers shape and type contracts, threshold-setting behavior, Dataframe output format, and the key behavioral assertion.

The most meaningful test is test_injected_anomaly_locations_score_above median in tests/test_detector.py. It:

1. Generates a clean training tensor and a separate test tensor with five spike anomalies at known (machine, sensor, timestep) locations.
2. Fits an "Anomalydetector" on the training tensor.
3. Scores the anomalous test tensor.
4. For each of the five injected windows, checks whether at least one timestep in that window scores above the overall test-set median.
5. Asserts that at least 3 of the 5 windows pass this check.

This is a stronger claim than merely "some anomalies are detected." It asserts that the reconstruction error is **spatially localized** to the actual fault locations rather than being uniformly elevated across the score matrix. A detector that raised all scores indiscriminately would not pass this test. Passing it means the CP factors Learned from normal data are specifically failing to reconstruct the corrupted sensor vectors, which is the behavior the technique is designed to produce.

## Known Limitations
 Portfolio Code
- CP decomposition assumes the normal operating data has low-rank structure - that machines, sensors, and time interact in a small number of dominant patterns. If all machines are truly independent of one another, the tensor will not be low-rank and the method will produce poor reconstructions even on healthy data, inflating the false-positive rate.

- Rank selection is manual. There is no automatic rank determination built into this library. Users must select rank by inspecting reconstruction error curves as described in the Rank Selection section.

- The anomaly threshold is set from the training data distribution. If the production environment shifts - new machines added to the fleet, sensors recalibrated, operating regime changed - the threshold will no longer reflect the in-distribution error level, and false positives will increase. Periodic retraining or threshold recalibration is necessary to handle distribution shift.

- Very large tensors wil1 be slow. ALS complexity scales roughly as 0(zank × n_machines x n_sensors × n_timesteps) per iteration. Tensors with thousands of machines or sensors, or with very long time horizons, may require significant compute time. tensorly does not parallelize ALS across cores by default.

## Data Note
All data in this repository is fully synthetic. TelemetryGenerator constructs tensors as a sum of CP-stylerank-1 outer products (machine factors,sensor factors, diurnal time factors) plus Gaussian noise, so the CP recovery problem is meaningful but non-trivial. Noreal sensor data is required to run any part of this project.