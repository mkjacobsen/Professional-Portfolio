"""
Unit tests for DemandDataGenerator
"""

import pandas as pd
import pytest

from mlops.data import DemandDataGenerator

# Fixtures

@pytest.fixture(scope="module")
def generator():
    return DemandDataGenerator(n_stores=3, n_products=4, seed=0)

@pytest.fixture(scope="module")
def small_df(generator):
    return generator.generate("2023-01-01", "2023-01-31")

# Shape and Column Tests

class TestGenerate:
    def test_returns_dataframe(self, small_df):
        assert isinstance(small_df, pd.DataFrame)

    def test_expected_columns_present(self, small_df):
        required = {
            "date", "store_id", "product_id", "demand",
            "price", "day_of_week", "month", "is_weekend", "is_holiday",
            "store_size", "product_category", "lag_1_demand",
            "lag_7_demand", "rolling_7_mean",
        }
        assert required.issubset(set(small_df.columns))

    def test_row_count(self, generator, small_df):
        """31 days x 3 stores x 4 products = 372 rows"""
        expected = 31 * generator.n_stores * generator.n_products
        assert len(small_df) == expected

    def test_store_ids(self, generator, small_df):
        assert set(small_df["product_id"].unique()) == set(range(1, generator.n_products + 1))

    def test_demand_non_negative(self, small_df):
        assert (small_df["demand"] >= 0).all()

    def test_price_positive(self, small_df):
        assert (small_df["price"] > 0).all()

    def test_is_weekend_binary(self, small_df):
        assert set(small_df["is_weekend"].unique()).issubset((0,1))

    def test_is_holiday_binary(self, small_df):
        assert set(small_df["is_holiday"].unique()).issubset((0,1))

    def test_store_size_values(self, small_df):
        assert set(small_df["store_size"].unique()).issubset({"small", "medium", "large"})

    def test_product_category_values(self, small_df):
        assert set(small_df["product_category"].unique()).issubset({"A","B","C"})

    def test_date_column_is_datetime(self, small_df):
        assert pd.api.types.is_datetime64_any_dtype(small_df["date"])

    def test_sorted_by_date(self, small_df):
        dates = small_df["date"].values
        assert (dates[:-1] <= dates[1:]).all()

# Drift injection tests

class TestInjectDrift:
    def test_covariate_drift_shifts_price_mean(self, generator):
        """Price mean after drift_start should be higher than before"""
        df = generator.generate("2023-01-01", "2023-06-30")
        drifted = generator.inject_drift(df, drift_start="2023-04-01", drift_type="covariate")

        pre_drift = drifted[drifted["date"] < "2023-04-01"]["price"].mean()
        post_drift = drifted[drifted["date"] >= "2023-04-01"]["price"].mean()
        assert post_drift > pre_drift, "Price should increase after covariate drift"

    def test_label_drift_shifts_demand_mean(self, generator):
        """Demand mean after drift_start should be lower than before (concept drift)"""
        df = generator.generate("2023-01-01", "2023-06-30")
        drifted = generator.inject_drift(df, drift_start="2023-04-01", drift_type="label")

        pre_drift = drifted[drifted["date"] < "2023-04-01"]["demand"].mean()
        post_drift = drifted[drifted["date"] >= "2023-04-01"]["demand"].mean()
        assert post_drift < pre_drift, "Demand should decrease after label drift"

    def test_both_drift_affects_price_and_demand(self, generator):
        """Both price and demand should shift when drift_type='both'"""
        df = generator.generate("2023-01-01", "2023-06-30")
        drifted = generator.inject_drift(df, drift_start="2023-04-01", drift_type="both")

        pre_price = drifted[drifted["date"] < "2023-04-01"]["price"].mean()
        post_price = drifted[drifted["date"] >= "2023-04-01"]["price"].mean()
        pre_demand = drifted[drifted["date"] < "2023-04-01"]["demand"].mean()
        post_demand = drifted[drifted["date"] >= "2023-04-01"]["demand"].mean()

        assert post_price > pre_price
        assert post_demand < pre_demand

    def test_drift_preserves_shape(self, generator, small_df):
        """Injecting drift must not change the number of rows or columns"""
        drifted = generator.inject_drift(small_df, drift_start="2023-01-15")
        assert drifted.shape == small_df.shape

    def test_original_not_mutated(self, generator, small_df):
        """inject_drift must return a copy, not modify the original"""
        original_price_mean = small_df["price"].mean()
        generator.inject_drift(small_df, drift_start="2023-01-01", drift_type="covariate")
        assert small_df["price"].mean() == pytest.approx(original_price_mean, rel=1e-6)