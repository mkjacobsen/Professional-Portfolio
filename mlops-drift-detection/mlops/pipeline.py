"""
Orchestrates the full MLOps workflow

The :class:`MLOpsPipeline` class ties together data generation, feature
engineering, model training, validation gating, artifact storage, feature
snapshotting, and production monitoring into a single coherent workflow.

Typical usage::

    pipeline = MLOpsPIpeline(
        data_path="./data",
        artifact_store_path="./artifacts",
        feature_store_path="./feature_store",
    )
    result = pipeline.run_training_pipeline(
        train_start="2023-01-01",
        train_end="2023-09-30",
        val_start="2023-10-01",
        val_end="2023-12-31",
    )
    report = pipeline.run_monitoring(
        monitoring_start="2024-01-01",
        monitoring_end="2024-03-31",
    )
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DemandDataGenerator
from .features import FEATURE_COLUMNS, FeatureStore, build_feature_pipeline
from .monitoring import DriftDetector, MonitoringReport
from .store import ArtifactStore
from .training import ModelTrainer, TrainingResult, ValidationGate

# Cols that need to be ordinal-encoded
_STORE_SIZE_MAP = {"small": 0, "medium": 1, "large": 2}
_PRODUCT_CAT_MAP = {"A": 0, "B": 1, "C": 2}


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode store_size and product_category in place
    """

    out = df.copy()
    if "store_size" in out.columns:
        out["store_size_encoded"] = out["store_size"].map(_STORE_SIZE_MAP).fillna(0).astype(int)
    if "product_category" in out.columns:
        out["product_category_encoded"] = (
            out["product_category"].map(_PRODUCT_CAT_MAP).fillna(0).astype(int)
        )
    return out

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature pipeline + categorical encoding, return model-ready df"""
    pipeline = build_feature_pipeline()
    engineered = pipeline.fit_transform(df)
    engineered = _encode_categoricals(engineered)

    # Drop rows with NaN lag features
    engineered = engineered.dropna(subset=[c for c in FEATURE_COLUMNS if c in engineered.columns])
    return engineered

class MLOpsPipeline:
    """
    End-to-end MLOps pipeline: train to validate to deploy to monitor
    
    Parameters:
    data_path: Path to a directory where raw data CSVs are written.  Directory created if not there already.
    artifact_store_path: Root directory for the :class:`store.ArtifactStore`
    feature_store_path: Root directory for the :class:`features.FeatureStore`
    gate: Optional custom :class:`training.ValidationGate`. Defaults to standard gate thresholds.
    n_stores: Number of stores in the synthetic dataset
    n_products: Number of products in the synthetic dataset
    seed: Random seed passed to :class:`data.DemandDataGenerator`
    """

    _TRAINING_SNAPSHOT = "training_features"

    def __init__(
        self,
        data_path: str | Path,
        artifact_store_path: str | Path,
        feature_store_path: str | Path,
        gate: ValidationGate | None = None,
        n_stores: int = 5,
        n_products: int = 10,
        seed: int = 55,
    ) -> None:
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.artifact_store = ArtifactStore(artifact_store_path)
        self.feature_store = FeatureStore(feature_store_path)
        self.gate = gate or ValidationGate()
        self.generator = DemandDataGenerator(
            n_stores = n_stores,
            n_products = n_products,
            seed = seed,
        )
        self.trainer = ModelTrainer(gate=self.gate)


    # Training Pipeline
    def run_training_pipeline(
        self,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
    ) -> TrainingResult:
        """
        Ingest, engineer, train, validate, and optionally promote a model

        Steps:
        1. Generate synthetic data covering ``train_start`` to ``val_end``
        2. Apply feature engineering pipeline
        3. Split into train / validation sets by date
        4. Train model and apply validation gate
        5. If gate passed: save artifact, promote to champion, snapshot features

        Parameters:
        train_start, train_end: Inclusive date range for training split
        val_start, val_end: Inclusive date range for validation split

        Returns:
        Trainingresult: Full training outcome including gate verdict
        """

        # Generate Data
        full_df = self.generator.generate(
            start_date=train_start, end_date=val_end
        )

        # Feature Engineering
        engineered = _prepare_features(full_df)

        # Split
        train_mask = (engineered["date"] >= pd.to_datetime(train_start)) & (
            engineered["date"] <= pd.to_datetime(train_end)
        )

        val_mask = (engineered["date"] >= pd.to_datetime(val_start)) & (
            engineered["date"] <= pd.to_datetime(val_end)
        )

        train_df = engineered[train_mask]
        val_df = engineered[val_mask]

        feature_cols = [c for c in FEATURE_COLUMNS if c in engineered.columns]
        X_train = train_df[feature_cols]
        y_train = train_df["demand"]
        X_val = val_df[feature_cols]
        y_val = val_df["demand"]

        # Train + gate
        champion_mae = self.artifact_store.get_champion_mae()
        result = self.trainer.train(X_train, y_train, X_val, y_val, champion_mae)

        # Persist if passed
        if result.passed_gate:
            version = self._next_version()
            artifact_id = self.artifact_store. save_model(
                result.model,
                metrics = {
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "r2": result.r2,
                    "version": version,
                    "trained_at": result.trained_at,
                },
                version = version,
            )
            self.artifact_store.promote_to_champion(artifact_id)

            # Safe Feature snapshots for drift monitoring
            snapshot_cols = feature_cols + ["demand"]
            snapshot_df = train_df[[c for c in snapshot_cols if c in train_df.columns]]
            self.feature_store.save_snapshot(self._TRAINING_SNAPSHOT, snapshot_df)

            print(f"[Pipeline] Model promoted to champion: {artifact_id}")
        else:
            print(f"[Pipeline] Model did NOT pass gate: {result.gate_reason}")

        return result

    # Monitoring Pipeline
    
    def run_monitoring(
        self,
        monitoring_start: str,
        monitoring_end: str,
        drift_type: str | None = None,
    ) -> MonitoringReport:
        """
        Generate production data and run drift detection

        Steps:
        1. Generate synthetic "production" data for the monitoring window
        2. Optionally inject drift (when *drift_type* is not ``None``)
        3. Apply feature engineering
        4. Load the training feature snapshot from the FeatureStore
        5. Run :class:`monitoring.DriftDetector` and return report

        Parameters:
        monitoring_start, monitoring_end: Inclusive date range for production window
        drift_type: Optional drift type to inject: covariate, label, or both.  Pass None for clean production data.

        Returns:
        MonitoringReport
        """

        # 1 and 2
        prod_df = self.generator.generate(
            start_date=monitoring_start, end_date=monitoring_end
        )

        if drift_type is not None:
            prod_df = self.generator.inject_drift(
                prod_df, drift_start=monitoring_start, drift_type=drift_type,
            )

        # 3.
        prod_engineered = _prepare_features(prod_df)
        prod_engineered = _encode_categoricals(prod_engineered)

        # 4.
        try:
            reference_df = self.feature_store.load_snapshot(self._TRAINING_SNAPSHOT)
        except KeyError:
            raise RuntimeError(
                "Training feature snapshot not found."
                "Run run_training_pipeline first."
            )

        feature_cols = [c for c in FEATURE_COLUMNS if c in prod_engineered.columns]
        current_features = prod_engineered[
            [c for c in feature_cols if c in prod_engineered.columns]
        ]
        current_labels = prod_engineered["demand"].to_numpy(dtype=float) if "demand" in prod_engineered.columns else None

        # 5.
        detector = DriftDetector(reference_df)
        report = detector.generate_report(current_features, current_labels)

        if report.retraining_recommended:
            print(
                "[Monitoring] WARNING: Drift dtected - retraining is recommended! \n"
                f"  Summary: {report.summary}"
            )
        else:
            print(F"[Moniotring] No significant drift. {report.summary}")

        return report

    # Private helpers

    def _next_version(self) -> int:
        """Return next artifact version number (1-indexed)"""
        existing = [
            d for d in self.artifact_store.path.iterdir() if d.is_dir() and d.name.startswith("model_v")
        ]

        return len(existing) + 1

    