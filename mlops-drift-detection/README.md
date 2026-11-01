# MLOps Drift Detection - Retail Demand Forecasting Pipeline

An end-to-end MLOps portfolio project demonstrating production-grade machine learning operations for retail demand forecasting. The pipeline covers the complete lifecycle: synthetic data generation, feature engineering, model training, automated validation gating, model serving via FastAPI, and continuous production monitoring with statistical drift detection.

## Overview

In production, model performance degrades silently when the data distribution shifts. This project implements two complementary statistical tests to catch that drift early:

- **Population Stability Index (PSI)** - quantifies covariate drift (feature distribution shift) between training and production populations.
- **Kolmogorov-Smirnov (KS) test** - a non-parametric test for both feature and label (concept) drift.

When drift crosses configurable thresholds a retraining workflow is automatically triggered, preventing the model from silently making bad predictions.

## Architecture
```
              DemandDataGenerator  <- Synthetic Retail Data
              | raw DataFrame
              Feature Pipeline
              LagFeatureBuilder (point in time lags)
              CyclicalEncoder (sin/cos time features)
              |
    ----------------------
    |         |          |
    Training   Validation FeatureStore
    set         set         (snapshot)
    |         |
    Model Trainer
    GradientBoostingRegressor
          |
        ValidationGate (MAE/RMSE/R2/improvement)
    pass  |    fail (discard)
    ArtifactStore
    (champion)
          |
    FastAPI Server (POST /predict, GET /health, POST /monitor)
          |
    DriftDetector
    PSI (covariate drift)
    KS test (label drift)
          |  MonitoringReport
    retraining_recommended? -> yes -> re-run training pipeline
```

### Components

| Module | Responsibility |
| ------ | -------------- |
| `data.py` | Synthetic retail demand generator with configurable drift injection |
| `features.py` | Point-in-time lag features, cyclical time encoding, feature snapshots |
| `training.py` | GradientBoostingRegressor training, validation gate |
| `monitoring.py` | PSI and KS-test drift detection, monitoring repots, drift plots |
| `store.py` | File-based artifact store for models, metrics, and champion pointer |
| `serving.py` | FastAPI prediction service with `/predict`, `/health`, `/monitor` |
| `pipeline.py` | Orchestrator end-to-end train -> validate -> promote -> monitor workflow |


## PSI Methodology

**Population Stability Index** measures how much a variable's distribution has shifted between a *reference* (training-time) population and a *current* (production) population.

### Formula

```
PSI = \sum((A_{i}% - E_{i}%) * ln(A_{i}% / E_{i}%))
```

where:
- **E_{i}%** = proportion of reference observations in bin *i*
- **A_{i}%** = proportion of current observations in bin *i*
- Bins are derived from the reference distribution using equal-frequency quantiles (Deciles by default)

### Thresholds (industry standard)

| PSI Value | Interpretation | Action |
| --------- | -------------- | ------ |
| < 0.10 | No meaningful shift | Monitor as usual |
| 0.10 - 0.20 | Moderate Shift | Investigatel consider retraining |
| >= 0.20 | Significant shift | Retrain immediately |

### Why PSI?

PSI is preferred over simple mean comparisons because it captures the full shape of the distribution shift, not just the first moment.  It is widely used in financial ML (credit risk scoring) where regulatory scrutiny of distrubtion stability is high. It is also interpretable: a PSI value can be traced back to the specific bin(s) driving the drift.

## KS Test Methodology

The **Klmogorov-Smirnov two-sample test** (`scipy.stats.ks_2samp`) measures the maximum absolute difference between the empirical CDFs of two samples: 

```
D = sup|F_{1}(x) - F_{2}(s)|
```

The test returns a **p-value**: the probability of observing a statistic as extreme as *D* if both samples came from the same distribution.

- **p-value less than 0.05** -> reject the null hypothesis -> drift detected
- **p-value greater or equal 0.05** -> insufficient evidence of drift

KS is used for both feature drife and label (concept) drift.  It is distribution-free and sensitive to shifts in mean, variance, and shape.

## Validation Gate

The validation gate prevents a newly trained challenger model from being promoted to production unless all of the following criteria are met: 

1. **MAE** <= `max_mae` (Default 50 demand units)
2. **RMSE** <= `max_rmse` (Default 75 demand units)
3. **R-squared** >= `min_r2` (default 0.65)
4. **Improvement**: MAE at least `min-improvement` (default 2%) better than the current champion model

If any check fails, the model is discarded with a human-readable reason string explaining exactly which threshold was violated.  This prevents silent performance regressions from reaching production.

## Quickstart

### Install

```bash
cd mlops-drift-detection
pip install -e ".[dev]"
```

### Run end-to-end example

```bash
python examples/run_pipeline.py
```

This will:
1. Generate synthetic data and train a model. 
2. Inject covariate drift and run monitoring.
3. Save a drift visualization to `examples/output/drift_report.png`.

### Run tests

```bash
pytest tests/ -v
```

### Start the FastAPI server

```bash
ARTIFACT_STORE_PATH=./artifacts uvicorn mlops.serving:app --reload
```

Then:
```bash
# Health check
curl https://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
    "store_id": 1,
    "product_id": 3,
    "date": "2024-01-15",
    "price": 9.99,
    "is_weekend": false,
    "is_holiday": false,
    "lag_1_demand": 52.0,
    "lag_7_demand": 48.5,
    "rolling_7_mean": 50.2
}'
...
```

## API Reference

### `DemandDataGenerator`

```python
DemandDataGenerator(n_stores=10, n_products=20, seed=44).generate(start_date, end_date) -> pd.DataFrame.inject_drift(df, drift_start, drift_type="covariate") -> pd.Dataframe
```

### `LagFeatureBuilder`

```python
lagFeatureBuilder(lag_days = [1,7,14], rolling_windows=[7,14]).fit(X) -> self.transform(X) -> pd.DataFrame #adds lag_N_demand, rolling_N_mean columns
```

### `CyclicalEncoder`

```python
CyclicalEncoder().transform(X) -> pd.DataFrame #adds day_of_week_sin/cox, month_sin/cos
```

### `FeatureStore`

```python
FeatureStore(store_path)
.save_snapshot(name, df) -> None 
.load_snapshot(name) -> pd.DataFrame
.list_snapshots() -> list[str]
```

### `ModelTrainer`

```python
ModelTrainer(gate=ValidationGate())
.train(X_train, y_train, X_val, y_val, champion_mae = None) -> TrainingResult
```

### `DriftDetector`

```python
DriftDetector(reference_data)
    .check_feature_drift(current_Data, features=None) -> list[DriftResult]
    .check_label_Drift(reference_labels, current_labels) -> Drift Result
    .plot_drift(current_data, features) -> matplotlib.Figure
```

### `ArtifactStore`

```python
ArtifactStore(store_path)
    .save_model(model, metrics, version) -> artifact_id
    .load_champion() -> (model, metrics) | None
    .promote_to_champion(artifact_id) -> None
    .get_champion_mae() -> float | None
```

### `MLOpsPipeline`

```python
MLOpsPipeline(data_path, artifact_store_path, feature_store_path, gate=None)
    .run_training_pipeline(train_start, train_end, val_start, val_end) -> TrainingResult
    .run_monitoring(monitoring_start, monitoring_end, drift_type=None) -> MonitoringReport
```

### FastAPI Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/health` | Service health, model version, champion MAE |
| `POST` | `/predict` | Single demand forecast |
| `POST` | `/monitor` | Batch drift detection on recent observations |

## Running Tests

```bash
#All Tests with verbose output
pytest tests/ -v

#Specific test file
pytest /tests/test_monitoring.py -v

#With coverage (requires pytest-cov)
pytest tests/ --cov=mlops --cov-report=term-missing
```

Test files:
- `test_data.py` - data generation shape, column presence, drift injection
- `test_features.py` - lag correctness, cyclical encoding, FeatureStore round-trip
- `test_training.py` - TrainingResult fields, gate pass/fail scenarios
- `test_monitoring.py` - PSI=0 for identifical distributions, PSI critical for heavy drift, KS p-value interpretation, full report generation
- `test_pipeline.py` - full pipeline integration: champion saved, monitoring report generated, retraining recommended after drift

## Project Layout

```
mlops-drift-detection/
|---- pyproject.toml
|---- README.md
|---- src/
|     |---__init__.py
|     |---data.py #DemandDataGenerator
|     |---features.py #LagFeatureBuilder, CyclicalEncoder, FeatureStore
|     |---training.py #ModelTrainer, ValidationGate, TrainingResult
|     |---serving.py #FastAPI app
|     |---monitoring.py #DriftDetector, DriftResult, MonitoringReport
|     |---pipeline.py #MLOpsPipeline
|     |---store.py #ArtifactStore
|---- tests/
|     |--- test_data.py
|     |--- test_features.py
|     |--- test_training.py
|     |--- test_monitoring.py
|     |--- test_pipeline.py
|---- examples/
      |--- run_pipeline.py #complete demo script
```






