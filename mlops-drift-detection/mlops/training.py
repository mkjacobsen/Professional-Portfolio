"""
Model training and validation gate for retail demand forecasting

Validation gate prevents a newly trained model from being promoted to
production unless it meets minimum quality thresholds *and* improved upon the currently
deployed champion model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass
class ValidationGate:
    """
    Thresholds that a trained model must satisfy before promotion
    
    Parameters:
    - max_mae: Maximum acceptable Mean Absolute Error on the validation set
    - max_rmse: Maximum acceptable Root Mean Squared Error on the validation set
    - min_r2: Minimum acceptable R2 (coefficient of determination) on the validation set
    - min_improvement: Minimum fractional improvement in MAE over current champion.  (e.g. 0.02 means challenger must be 2% better)
    """

    max_mae: float = 50.0
    max_rmse: float = 75.0
    min_r2: float = 0.65
    min_improvement: float = 0.02


@dataclass
class TrainingResult:
    """
    Outcome of a single training run, including gate verdict

    Attributes:
    - model: Trained sklearn estimator
    - mae: Mean Absolute Error on validation set
    - rmse: Root Mean Squared Error on the validation set
    - r2: R2 score on validation set
    - passed_gate: ``True`` if model satisfied all :class:`ValidationGate` checks
    - gate_reason: Human-readable explanation of why the gate passed or failed
    - trained_at: UTC timestamp of when training was completed
    - n_train_samples: Number of rows used for training
    - n_val_samples: Number of rows used for validation
    - feature_names: Ordered list of feature column names used during training
    """

    model: object
    mae: float
    rmse: float
    r2: float
    passed_gate: bool
    gate_reason: str
    trained_at: str
    n_trained_samples: int
    n_val_samples: int
    feature_names: list[str] = field(default_factory=list)

class ModelTrainer:
    """
    Train a GradientBoostingRegressor and evaluate it through a validation gate

    Parameters:
    - gate: Validation thresholds to apply.  Defaults to :class:`ValidationGate` with its default parameters
    """

    def __init__(self, gate: ValidationGate | None = None) -> None:
        self.gate = gate or ValidationGate()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        champion_mae: float | None = None,
    ) -> TrainingResult:
        """
        Train a model and evaluate it against the validation gate.

        Parameters:
        - X_train: Feature matrix for training
        - y_train: Target series for training
        - X_val: Feature matrix for validation
        - y_val: Target series for validation
        - champion_mae: MAE of currently deployed champion model.  If ``None`` the improvement check is skipped (e.g. no champion exists)

        Returns:
        - TrainingResult: Contains trained model, metrics, gate verdict, and metadata
        """

        model = GradientBoostingRegressor(
            n_estimators = 200,
            learning_rate = 0.05,
            max_depth = 4,
            min_samples_leaf = 10,
            subsample = 0.8,
            random_state = 55,
        )

        model.fit(X_train, y_train)

        mae, rmse, r2 = self._evaluate(model, X_val, y_val)
        passed, reasons = self._apply_gate(mae, rmse, r2, champion_mae)

        return TrainingResult(
            model = model,
            mae = round(mae, 4),
            rmse = round(rmse, 4),
            r2 = round(r2, 4),
            passed_gate = passed,
            gate_reason = reasons,
            trained_at = datetime.now(timezone.utc).isoformat(),
            n_trained_samples = len(X_train),
            n_val_samples = len(X_val),
            feature_names = list(X_train.columns),
        )

    # Private Helpers

    def _evaluate(
        self,
        model: GradientBoostingRegressor,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[float, float, float]:
        """
        Return MAE, RMSE, R2 for *model* on the validation set.
        """

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        r2 = r2_score(y_val, preds)

        return mae, rmse, r2

    def _apply_gate(
        self,
        mae: float,
        rmse: float,
        r2: float,
        champion_mae: float | None,
    ) -> tuple[bool, str]:
        """
        Return ``(passed, reason)`` after evaluating gate checks
        
        Checks are applied in order; the first failure short-circuits and
        returns a descriptive reason string so engineers know exactly which
        threshold was violated.
        """

        if mae > self.gate.max_mae:
            return False, (
                f"FAILED: MAE {mae:.4f} exceeds threshold {self.gate.max_mae}"
            )
        if rmse > self.gate.max_rmse:
            return False, (
                f"FAILED: RMSE {rmse:.4f} exceeds threshold {self.gate.max_rmse}"
            )
        if r2 < self.gate.min_r2:
            return False, (
                f"FAILED: R2 {r2:.4f} below minimum {self.gate.min_r2}."
            )
        if champion_mae is not None:
            required = champion_mae * (1 - self.gate.min_improvement)
            if mae >= required:
                return False, (
                    f"FAILED: MAE {mae:.4f} does not improve champion "
                    f"{champion_mae:.4f} by {self.gate.min_improvement*100:.0f}% "
                    f"(required < {required:.4f})"
                )
        return True, (
            f"PASSED: MAE={mae:.4f}, RMSE={rmse:.f4}, R2={r2:.4f}"
            + (
                f", improved champion by "
                f"{(champion_mae - mae) / champion_mae * 100:.1f}%"
                if champion_mae is not None
                else ""
            )
        )