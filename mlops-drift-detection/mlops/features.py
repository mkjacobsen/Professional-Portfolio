"""
Feature engineering pipeline for retail demand forecasting.

Provides:
- LagFeatureBuilder: adds point in time correct lag and rolling mean features
- CyclicalEncoder: encodes day of week and month as sin/cos pairs
- FeatureStore: saves/loads feature snapshots (parquet) for drift monitoring
- build_feature_pipeline(): convenience factory for full sklearn pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Public constant - canonical feature columns consumed by model
FEATURE_COLUMNS: list[str] = [
    "price",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "lag_1_demand",
    "lag_7_demand",
    "lag_14_demand",
    "rolling_7_mean",
    "rolling_14_mean",
    "store_size_encoded",
    "product_category_encoded",
]

# Transformers

class LagFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Add lag and rolling-mean demand features to a retail demand DataFrame
    
    Features are computed *point-in-time correctly*: for each
    (store_id, product_id) pair the time series is sorted by date and values 
    are shifted forward so that the prediction for day *t* only sees demand
    from day *t-k* or earlier (``pd.Series.shift(k)``)
    
    Parameters
    lag_days: list of lag periods in days to compute
    rolling_windows: Lis of rolling-window sizes (in days) to compute means over
        The rolling mean is applied to the 1-day shifted series so it
        never leaks the current day's demand.
    """

    def __init__(
        self,
        lag_days: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        self.lag_days = lag_days if lag_days is not None else [1, 7, 14]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [7, 14]

    def fit(self, X: pd.DataFrame, y=None) -> "LagFeatureBuilder": 
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        return *X* with lag and rolling-mean columns appended.

        The input DataFrame must constin ``date``, ``store_id``,
        ``product_id``, and ``demand`` columns.  The ``demand`` columns
        is used as the source for lagged values.
        """

        df = X.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["store_id", "product_id", "date"])

        group = df.groupby(["store_id", "product_id"])["demand"]

        for k in self.lag_days:
            col = f"lag_{k}_demand"
            if col not in df.columns: 
                df[col] = group.shift(k).values
        
        # Rolling means - shift (1) first to avoid look-ahead, then rolling
        shifted = group.shift(1)
        for w in self.rolling_windows:
            col = f"rolling_{w}_mean"
            if col not in df.columns:
                df[col] = shifted.transform(lambda s: s.rolling(w).mean()).values

        return df

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode ``day_of_week`` and ``month`` as sin/cos pairs

    Cyclical encoding preserves the natural periodicity of time features
    (e.g., Sunday and Monday are close in angle) without implying a linear
    ordering.  For a feature with period *P*, the encoding is:

    ..code-block::text
        sin_val = sin(2pi * value / P)
        cos_val = cos(2pi * value / P)
    
    ``day_of_week`` has period 7 (values 0 - 6)
    ``month`` has period 12 (values 1 - 12)
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CyclicalEncoder":
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "day_of_week" in df.columns:
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        if "month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
            df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
        
        return df

# Feature Store

class FeatureStore:
    """
    Simple file-backed store for feature snapshots

    Snapshots are saved as parquet files under *store_path*.  An index JSON
    file tracks available snapshots.  The store is used to persist the training-time feature
    distribution so the rift detector can compare it against production observations
    
    Parameters
    store_path: Directory where snapshots are written. Created if it does not exist.
    """

    _INDEX_FILE = "feature_store_index.json"

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok = True)
        self._index_path = self.store_path / self._INDEX_FILE
        self._index: dict[str, str] = self._load_index()

    # Public methods
    def save_snapshot(self, name: str, df: pd.DataFrame) -> None:
        """
        Persist a feature DataFrame as a Parquet snapshot

        Parameters
        name: logical name of snapshot
        df: DataFrame to persist
        """

        filename = f"{name}.parquet"
        path = self.store_path / filename
        df.to_parquet(path, index=False)
        self._index[name] = filename
        self._save_index()

    def load_snapshot(self, name: str) -> pd.DataFrame:
        """
        Load a saved snapshot
        
        Parameters
        name: Logical name of snapshot to load
        
        Raises
        KeyError - if *name* does not exist in the store
        """

        if name not in self._index:
            raise KeyError(f"Snapshot '{name}' not found.")

        path = self.store_path / self._index[name]
        return pd.read_parquet(path)

    def list_snapshots(self) -> list[str]:
        """Return names of all stored snapshots"""
        return list(self._index.keys())


    # Private Helpers
    def _load_index(self) -> dict[str, str]:
        if self._index_path.exists():
            with open(self._index_path) as fh:
                return json.load(fh)

        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as fh:
            json.dump(self._index, fh, indent=2)


# Pipeline Factory
def build_feature_pipeline() -> Pipeline:
    """Return an sklearn Pipeline that applies lag features then cyclical encoding
    
    The pipeline is stateless (both transformers implement ``fit`` as a no-op)
    so it can be applied to both training and serving data without risk of 
    training-time leakage between the two splits.
    """

    return Pipeline(
        [
            ("lag_features", LagFeatureBuilder()),
            ("cyclical_encoder", CyclicalEncoder()),
        ]
    )