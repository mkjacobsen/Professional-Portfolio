"""
run_pipeline.py - End-to-end MLOps demo for the retail deman forecasting pipeline.

Steps
1. Generate 1 year of synthetic training data + three months of validation data.
2. Run the full training pipeline; print the TrainingResult.
3. Generate 3 months of "production" data with injected covariance drift.
4. Run monitoring; print the MonitoringReport.
5. Confirm drift is detected and retraining is recommended.
6. Generate a drift visualization and save it to ./output/drive_report.png.
7. (Optional, commented out) Start the FastAPI prediction server.

Usage::
    cd mlops-drift-detection
    python examples/run_pipeline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is importable when running
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mlops.data import DemandDataGenerator
from mlops.features import FEATURE_COLUMNS, build_feature_pipeline
from mlops.monitoring import DriftDetector
from mlops.pipeline import MLOpsPipeline
from mlops.training import ValidationGate

# Configuration

ARTIFACT_STORE = _PROJECT_ROOT / "artifacts"
FEATURE_STORE = _PROJECT_ROOT / "feature_store"
DATA_PATH = _PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok = True)

# Helpers

def _section(title: str) -> None:
    width = 70
    print("\n" + "="*width)
    print(f".  {title}")
    print("="*width)

def _print_training_result(result) -> None:
    print(f"  Passed gate   : {result.passed_gate}")
    print(f"  Gate reason   : {result.gate_reason}")
    print(f"  MAE           : {result.mae:.4f}")
    print(f"  RMSE          : {result.rmse: .4f}")
    print(f"  R2            : {result.r2:.4f}")
    print(f"  Train samples : {result.n_train_samples:,}")
    print(f"  Val samples   : {result.n_val_samples:,}")
    print(f"  Trained at    : {result.trained_at}")

def _print_monitoring_report(report) -> None:
    print(f"  Evaluated at     : {report.evaluated_at}")
    print(f"  Reference samples: {report.n_reference_samples:,}")
    print(f"  Current samples  : {report.n_current_samples:,}")
    print(f"  Overall drift    : {report.overall_drift_detected}")
    print(f"  Retraining needed: {report.retraining_recommended}")
    print(f"  Summary          : {report.summary}")

    print("\n Feature Drift Details (PSI):")
    psi_results = [r for r in report.feature_drift if r.method == "PSI"]
    for r in sorted(psi_results, key = lambda x: x.statistic, reverse=True)[:8]:
        flag = " *** DRIFT *** " if r.is_drift else ""
        print(
            f"    {r.feature:<24} PSI = {r.statistic:.4f}  "
            f"severity={r.severity:<8} shift={r.mean_shift_pct:+.1f}%{flag}"
        )

# Main

def main() -> None:

    # Step 1 & 2: Training Pipeline
    _section("STEP 1-2: Training Pipeline")

    pipeline = MLOpsPipeline(
        data_path=DATA_PATH,
        artifact_store_path=ARTIFACT_STORE,
        feature_store_path=FEATURE_STORE,
        gate=ValidationGate(max_mae=500.0, max_rmse=800.0, min_r2=0.0),
        n_stores=5,
        n_products=10,
        seed=55,
    )

    print("\nGenerating 12 months of training data + 3 months of validation data..")
    result = pipeline.run_training_pipeline(
        train_start="2021-10-01",
        train_end="2022-09-30",
        val_start="2022-10-01",
        val_end="2022-12-31",
    )

    print("\nTraining Result:")
    _print_training_result(result)

    # Step 3 & 4: Monitoring on drifted production data
    _section("STEP 3-4: Production Monitoring (With injected drift)")
    print("\nGenerating 3 months of production data with covariate drift (price +30)...")
    report = pipeline.run_monitoring(
        monitoring_start="2023-01-01",
        monitoring_end="2023-03-31",
        drift_type="covariate",
    )

    print("\nMonitoring Report:")
    _print_monitoring_report(report)

    # Step 5: Confirm drift detected
    _section("STEP 5: Drift Detection Confirmation")

    if report.overall_drift_detected:
        print("\n Drift DETECTED in production data.")
    else:
        print("\n No significant drift detected.")

    if report.retraining_recommended:
        print("   Retraining IS recommended.")
        print("   Action: trigger retraining workflow (run_training_pipeline)")
    else:
        print("   Retraining not required at this time.")

    # Step 6: Drift Visualization
    _section("STEP 6: Generating Drift Report Plot")

    import matplotlib
    matplotlib.use("Agg")

    # Load reference and produch fresh "drifted" production DataFrame
    generator = DemandDataGenerator(n_stores=5, n_products=10, seed=30)
    prod_df = generator.generate("2023-01-01", "2023-03-31")
    drifted_df = generator.inject_drift(prod_df, drift_start="2023-01-01", drift_type="covariate")

    # Feature Engineer
    from mlops.pipeline import _prepare_features, _encode_categoricals
    prod_engineered = _encode_categoricals(_prepare_features(drifted_df))

    reference_df = pipeline.feature_store.load_snapshot("training_features")

    # Plot representative set
    features_to_plot = [
        f for f in ["price", "lag_1_demand", "lag_7_demand", "rolling_7_mean",
                    "is_weekend", "is_holiday"]
        if f in prod_engineered.columns and f in reference_df.columns
    ]

    detector = DriftDetector(reference_df)
    fig = detector.plot_drift(prod_engineered, features = features_to_plot)

    out_path = OUTPUT_DIR / "drift_report.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n Drift Plot saved to: {out_path}")

    # Step 7: FastAPI Server
    _section("STEP 7: FastAPI Server (Optional)")

    print(
        "\n To start the prediction server, uncomment the block below"
        " and set ARTIFACT_STORE_PATH:\n"
    )

    # Uncomment to start the server:
    # import uvicorn
    # os.environ["ARTIFACT_STORE_PATH"] = str(ARTIFACT_STORE)
    # uvicorn.run("nlpos.server:app", host="0.0.0.0", port = 8000, reload = False)

    print("\nDone.")

if __name__ == "__main__":
    main()

    
