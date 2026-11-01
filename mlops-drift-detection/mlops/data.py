"""
Synthetic retail demand data generator

Generates time-series demand data for a configurable number of stores and 
products. Supports injecting covariate drift (price shift) and lable/concept
drift (demand shift) for use in drift-detection experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

class DemandDataGenerator:
    """
    Generate synthetic retail deman time series

    Parameters:
    n_stores: number of distinct store IDs to include.
    n_products: number of distinct product IDs to include.
    seed: random seed for reproducibility.
    """

    STORE_SIZES = ["small", "medium", "large"]
    PRODUCT_CATEGORIES = ["A", "B", "C"]

    # US Federal Holidays
    _HOLIDAYS = (
        (1, 1), # New Years
        (1, 18), # MLK Day (approximate)
        (7, 4), # Independence Day
        (11, 11), # Veterans Day
        (11, 27), # Thanksgiving (approximate)
        (12, 24), # Christmas Eve
        (12, 25), # Christmas Day
    )

    def __init__(
        self,
        n_stores: int = 10,
        n_products: int = 20,
        seed: int = 33,
    ) -> None:
        self.n_stores = n_stores
        self.n_products = n_products
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Stable per-store and per-product attributes
        self._store_sizes: dict[int, str] = {
            i: self.STORE_SIZES[i % 3] for i in range(1, n_stores + 1)
        }
        self._product_categories: dict[int, str] = {
            i: self.PRODUCT_CATEGORIES[i % 3] for i in range(1, n_products + 1)
        }

        # Base demand varies by store size and product category
        self._store_base: dict[int, float] = {
            i: {"small": 50.0, "medium": 100.0, "large": 200.0}[
                self._store_sizes[i]
            ] for i in range(1, n_stores + 1)
        }

        self._product_base: dict[int, float] = {
            i: {"A": 1.2, "B": 1.0, "C": 0.8}[self._product_categories[i]]
            for i in range(1, n_products + 1)
        }

    # Public API

    def generate(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate a synthetic demand dataset.

        Parameters
        start_date: inclusive start date string, e.g. "2023-01-01"
        end_date: inclusive end date string, e.g. "2023-12-31"

        Returns
        pd.Dataframe
            One row per (date, store_id, product_id) combination with columns:
            date, store_id, product_id, demand_price, day_of_week, month,
            is_weekend, is_holiday, store_Size, product_category,
            lag_1_demand, lag_7_demand, rolling_7_mean
        """

        dates = pd.date_range(start_date, end_date, freq="D")
        records: list[dict] = []

        rng = np.random.default_rng(self.seed)

        for store_id in range(1, self.n_stores + 1):
            for product_id in range(1, self.n_products + 1):
                base = self._store_base[store_id] * self._product_base[product_id]

                # Price centered around 10 with per-product offset
                price_base = 10.0 + (product_id % 5) * 0.5
                prices = rng.normal(loc=price_base, scale=0.5, size=len(dates))
                prices = np.clip(prices, 1.0, 50.0)

                demands = []
                for idx, date in enumerate(dates):
                    price = prices[idx]
                    dow = date.dayofweek # Mon = 0, Sun = 6
                    month = date.month

                    # Seasonal factor (higher in Nov/Dec)
                    seasonal = 1.0 + 0.4 * np.sin((month - 3) * np.pi / 6)

                    # Weekend bump
                    weekend_factor = 1.25 if dow >= 5 else 1.0

                    # Day of Week factor (midweek slower)
                    dow_factor = 1.0 - 0.05 * abs(dow - 2)/2

                    # Price Elasticity (demand falls as price rises)
                    price_elasticity = (price_base / price) ** 0.5

                    # Random promo spike (5% of days)
                    promo = 1.4 if rng.random() < 0.05 else 1.0

                    # Noise
                    noise = rng.lognormal(mean=0.0, sigma=0.08)

                    demand = (
                        base
                        * seasonal
                        * weekend_factor
                        * dow_factor
                        * price_elasticity
                        * promo
                        * noise
                    )
                    demand = max(0.0, round(demand, 2))
                    demands.append(demand)

                # Build lag features
                demand_series = pd.Series(demands, index=dates)
                lag_1 = demand_series.shift(1)
                lag_7 = demand_series.shift(7)
                rolling_7 = demand_series.shift(1).rolling(7).mean()

                for idx, date in enumerate(dates):
                    is_holiday = (date.month, date.day) in self._HOLIDAYS
                    records.append(
                        {
                            "date": date.date(),
                            "store_id": store_id,
                            "product_id": product_id,
                            "demand": demands[idx],
                            "price": round(prices[idx], 2),
                            "day_of_week": date.dayofweek,
                            "month": date.month,
                            "is_weekend": int(date.dayofweek >= 5),
                            "is_holiday": int(is_holiday),
                            "store_size": self._store_sizes[store_id],
                            "product_category": self._product_categories[product_id],
                            "lag_1_demand": lag_1.iloc[idx],
                            "lag_7_demand": lag_7.iloc[idx],
                            "rolling_7_mean": rolling_7.iloc[idx],
                        }
                    )
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)
        return df

    def inject_drift(
        self,
        df: pd.DataFrame,
        drift_start: str,
        drift_type: Literal["covariate", "label", "both"] = "covariate",
    ) -> pd.DataFrame:
        """
        Inject synthetic distribution drift into a dataset
        
        Parameters
        df: DataFrame produced by :meth: `generate`.
        drift_start: Date from which drift is applied (inclusive)
        drift_type: drift type to inject
            ``"covariate"`` shifts the price distribution upwards
            ``"label"`` shifts demand downward (concept drift)
            ``"both"`` applies both types

        Returns
        pd.DataFrame: a modified copy of *df* with drift applied after *drift_start*
        """

        out = df.copy()
        mask = pd.to_datetime(out["date"]) >= pd.to_datetime(drift_start)

        if drift_type in ("covariate", "both"):
            # Shift price distribution up by 30%
            out.loc[mask, "price"] = out.loc[mask, "price"] * 1.30 + 2.0

        if drift_type in ("label", "both"):
            # Reduce demand by 40%
            out.loc[mask, "demand"] = out.loc[mask, "demand"] * 0.60

        return out
    