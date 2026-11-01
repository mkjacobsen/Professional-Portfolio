"""
Unit tests for feature engineering components
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops.features import (
    CyclicalEncoder,
    FeatureStore,
    LagFeatureBuilder,
    build_feature_pipeline,
)

# Shared Fixtures

@pytest.fixture()
def tiny_df():
    """Minimal dataframe with 20 days of one store/product pair."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": 1,
            "product_id": 1,
            "demand": np.arange(1.0, 21.0),
            "day_of_week": dates.dayofweek,
            "month": dates.month,
        }
    )

# LagFeatureBuilder

class TestLagFeatureBuilder:
    def test_lag_1_correct(self, tiny_df):
        """lag_1_demand on row i should equal demand on row i-1"""
        builder = LagFeatureBuilder(lag_days=[1], rolling_windows=[])
        result = builder.transform(tiny_df)
        # Row 1: demand=2, lag_1_demand should be 1
        assert result.iloc[1]["lag_1_demand"] == pytest.approx(1.0)

    def test_lag_7_correct(self, tiny_df):
        """lag_7_demand on row 8 (0-indexed) should be equal demand on row 1"""
        builder = LagFeatureBuilder(lag_days=[7], rolling_windows=[])
        result = builder.transform(tiny_df)
        assert result.iloc[7]["lag_7_demand"] == pytest.approx(1.0)

    def test_lag_14_correct(self, tiny_df):
        """lag_14_demand on row 14 should equal demand on row 0"""
        builder = LagFeatureBuilder(lag_days=[14], rolling_windows=[])
        result = builder.transform(tiny_df)
        assert result.iloc[14]["lag_14_demand"] == pytest.approx(1.0)

    def test_first_row_lag_is_nan(self, tiny_df):
        """The first row must have NaN for all lag features (no prior data)"""
        builder = LagFeatureBuilder(lag_days = [1, 7], rolling_windows=[])
        result = builder.transform(tiny_df)
        assert pd.isna(result.iloc[0]["lag_1_demand"])
        assert pd.isna(result.iloc[0]["lag_7_demand"])

    def test_rolling_7_mean_value(self, tiny_df):
        """rolling_7_mean on row 7 should be mean of rows 0-6"""
        builder = LagFeatureBuilder(lag_days=[], rolling_windows=[7])
        result = builder.transform(tiny_df)
        expected = np.mean([1, 2, 3, 4, 5, 6, 7])
        assert result.iloc[7]["rolling_7_mean"] == pytest.approx(expected, rel=1e-4)

    def test_no_look_ahead(self, tiny_df):
        """lag_1_demand at index i must never exceed demand at index i"""
        builder = LagFeatureBuilder(lag_days=[1], rolling_windows=[])
        result = builder.transform(tiny_df).dropna(subset=["lag_1_demand"])
        assert (result["lag_1_demand"] < result["demand"]).all()

    def test_fit_returns_self(self, tiny_df):
        builder = LagFeatureBuilder()
        assert builder.fit(tiny_df) is builder

    def test_output_is_dataframe(self, tiny_df):
        builder = LagFeatureBuilder()
        result = builder.transform(tiny_df)
        assert isinstance(result, pd.DataFrame)

    def test_multi_store_product_independent(self):
        """Lag features must not cross (store_id, product_id) boundaries"""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5).tolist() * 2,
                "store_id": [1] * 5 + [2] * 5,
                "product_id": [1] * 5 + [1] * 5,
                "demand": [10.0] * 5 + [100.0] * 5,
                "day_of_week": [0] * 10,
                "month": [1] * 10,
            }
        )
        builder = LagFeatureBuilder(lag_days=[1], rolling_windows=[0])
        result = builder.transform(df)
        store2_first = result[(result["store_id"] == 2)].sort_values("date").iloc[0]
        assert pd.isna(store2_first["lag_1_demand"])

# Cyclical Encoder

class TestCyclicalEncoder:
    def test_day_of_week_sin_cos_period(self):
        df = pd.DataFrame({"day_of_week": np.arange(7)})
        enc = CyclicalEncoder()
        result = enc.transform(df)
        assert result.iloc[0]["day_of_week_sin"] == pytest.approx(
            result.iloc[6]["day_of_week_sin"] * 0 + np.sin(2 * np.pi * 0/7), abs=1e-6
        )

        s2 = result["day_of_week_sin"] ** 2
        c2 = result["day_of_week_cos"] ** 2
        np.testing.assert_allclose((s2 + c2).values, 1.0, atol=1e-10)

    def test_month_sin_cos_identity(self):
        """sin + cos = 1 for month encoding."""
        df = pd.DataFrame({"month": np.arange(1, 13)})
        enc = CyclicalEncoder()
        result = enc.transform(df)
        s2 = result["month_sin"] ** 2
        c2 = result["month cos"] ** 2
        np.testing.assert_allclose((s2 + c2).values, 1.0, atol=1e-10)

    def test_monday_and_sunday_close_in_angle(self):
        df = pd.DataFrame({"day_of_week": [0, 6]})
        enc = CyclicalEncoder()
        result = enc.transform(df)
        sin_mon, cos_mon = result.iloc[0][["day_of_week_sin", "day_of_week_cos"]]
        sin_sun, cos_sun = result.iloc[1][["day_of_week_sin", "day_of_week_cos"]]
        dot = sin_mon * sin_sun + cos_mon * cos_sun
        assert dot == pytest.approx(np.cos(2 * np.pi / 7), abs=1e-6)

    def test_columns_added(self):
        df = pd.DataFrame({"day_of_week": [0, 1], "month": [1, 6]})
        enc = CyclicalEncoder()
        result = enc.transform(df)
        for col in ("day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"):
            assert col in result.columns()

    def test_fit_returns_self(self):
        enc = CyclicalEncoder()
        assert enc.fit(pd.DataFrame()) is enc


# FeatureStore
class TestFeatureStore:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = FeatureStore(tmp_path / "fs")
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        store.save_snapshot("test_snap", df)
        loaded = store.load_snapshot("test_snap")
        pd.testing.assert_frame_equal(df, loaded, check_like=True)

    def test_list_snapshots(self, tmp_path):
        store = FeatureStore(tmp_path / "fs2")
        store.save_snapshot("snap_a", pd.DataFrame({"x":[1]}))
        store.save_snapshot("snap_b", pd.DataFrame({"x":[2]}))
        assert set(store.list_snapshots()) == ("snap_a", "snap_b")

    def test_missing_snapshot_raises(self, tmp_path):
        store = FeatureStore(tmp_path / "fs3")
        with pytest.raises(KeyError, match="not_here"):
            store.load_snapshot("not_here")

    def test_overwrite_snapshot(self, tmp_path):
        store = FeatureStore(tmp_path / "fs4")
        df1 = pd.DataFrame({"v": [1.0]})
        df2 = pd.DataFrame({"v": [99.0]})
        store.save_snapshot("snap", df1)
        store.save_snapshot("snap", df2)
        loaded = store.load_snapshot("snap")
        assert loaded["v"].iloc[0] == 99.0

    def test_store_creates_directory(self, tmp_path):
        path = tmp_path / "nested" / "dir"
        FeatureStore(path)
        assert path.is_dir()

# Build Feature Pipeline

class TestBuildFeaturePipeline:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        assert isinstance(build_feature_pipeline(), Pipeline)

    def test_pipeline_transforms(self, tiny_df):
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(tiny_df)
        assert "day_of_week_sin" in result.columns
        assert "month_cos" in result.columns
        