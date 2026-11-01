"""
Statistical drift detection for production ML models

To complementary methods are implements:

Population Stability Index - PSI measures how much a variable's distribution has shifted between a
reference (training) population and a current (production) population.

    PSI = /sum((actual_i% - expected_i%) * ln(actual_i%/expected_i%))

where the sum is over *N BINS* equal-frequency bins derived from the reference distribution.

Interpretation (industry standard thresholds):
PSI < 0.10 - no meaningful shift ("none")
0.1 <= PSI < 0.2 - moderate shift, monitor closely ("warning")
PSI >= 0.2 - significant shift, investigate/retrain ("critical")

Kolmogorov-Smirnov (KS) test:
The two-sample KS test from ``scipy.stats.ks_2samp`` measures the maximum
absolute difference between the empirical CDFs of the reference and current
samples.  A small p-value (<0.05) indicates the two samples are unlikely to
have come from the same underlying distribution.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Data Classes

@dataclass
class DriftResult:
    """
    Result of a single feature- or label-drift check

    Attributes
    feature: Name of the feature (or ``"demand"`` for label drift)
    method: PSI value or KS test statistic
    p_value: KS p-value (``None`` for PSI, which has no p-value)
    is_drift: ``True`` if drift is considered significant
    severity: ``"none"``, ``"warning"``, ``"critical"``
    reference_mean: Mean of the feature in the reference dataset
    current_mean: Mean of the feature in the current dataset
    mean_shift_pct: Percentage change: ``(current - reference)/|reference| * 100``
    """

    feature: str
    method: str
    statistic: float
    p_value: Optional[float]
    is_drift: bool
    severity: str
    reference_mean: float
    current_mean: float
    mean_shift_pct: float


@dataclass
class MonitoringReport:
    """
    Aggregated drift monitoring report for a production window

    Attributes
    evaluated_at: ISO-8601 UTC timestamp of when the report was generated
    n_reference_samples: Number of rows in the reference (training) dataset
    n_current_samples: Number of rows in the current (production) dataset
    feature_drift: One :class:`DriftResult` per (feature, method) combination
    label_drift: Drift result for the targer variable, or ``None`` if labels were not provided
    overall_drift_detected: ``True`` if *any* drift result has ``is_drift == True``
    retraining_recommended: ``True`` if any PSI result is ``"critical"`` *or* label drift is detected
    summary: Human-readable one-paragraph summary of findings
    """

    evaluated_at: str
    n_reference_samples: int
    n_current_samples: int
    feature_drift: list[DriftResult]
    label_drift: Optional[DriftResult]
    overall_drift_detected: bool
    retraining_recommended: bool
    summary: str

# DriftDetector

class DriftDetector:
    """
    Detect distribution shift between reference (training) and current data

    Parameters
    reference_data: DataFrame of feature values collected at training time.  Only numeric cols are analysed
    """

    PSI_WARNING: float = 0.1
    PSI_CRITICAL: float = 0.2
    KS_ALPHA: float = 0.05
    N_BINS: int = 10

    def __init__(self, reference_data: pd.DataFrame) -> None:
        self.reference = reference_data.copy()
        self._numeric_cols: list[str] = list(
            reference_data.select_dtypes(include=[np.number]).columns
        )
        self._reference_stats: dict[str, dict] = self._compute_stats(reference_data)

    # Core Drift Computation
    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Compute the population stability index between two 1-D arrays

        Algorithm
        1. Derive *N_BINS* equal-frequency bin edges from the reference array using ``np.percentile``.
        2. Compute *expected* proportions from the reference array
        3. Clip the current array to ``[min_edge, max_edge]`` and compute *actual* proportions using the same bin edges
        4. Use PSI equation (sum (actual-expected) * ln(actual/expected))

        A small epsilon (1e-6) is added to every proportion before taking the log to avoid ``log(0)`` or division-by-zero
        """

        eps = 1e-6
        percentiles = np.linspace(0, 100, self.N_BINS + 1)
        bin_edges = np.unique(np.percentile(reference, percentiles))

        # Need at least 2 distinct edges
        if len(bin_edges) < 2:
            return 0.0

        # Expected proportions from reference
        expected_counts, _ = np.histogram(reference, bins=bin_edges)
        expected = expected_counts / len(reference)

        #Clip to reference range, then compute actual proprtions
        current_clipped = np.clip(current, bin_edges[0], bin_edges[-1])
        actual_counts, _ = np.histogram(current_clipped, bins=bin_edges)
        actual = actual_counts/len(current_clipped)

        # Add epsilon to avoid log(0)
        expected = np.where(expected == 0, eps, expected)
        actual = np.where(actual == 0, eps, actual)

        psi = float(np.sum((actual-expected) * np.log(actual / expected)))

        return psi

    def _compute_stats(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Compute summary statistics (mean, std, quantiles) per column.
        """

        stats: dict[str, dict] = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "q25": float(s.quantile(0.25)),
                "q50": float(s.quantile(0.50)),
                "q75": float(s.quantile(0.75)),
            }

        return stats

    # Public API

    def check_feature_drift(
        self,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> list[DriftResult]:
        """
        Check drift for each numeric feature using PSI and KS test

        Parameters
        current_data: Recent production observations
        features: Subset of column names to analyze. Defaults to all numeric cols shared by reference and current data

        Returns
        list[DriftResult]: Two results per feature (one PSI, one KS)
        """

        if features is None:
            shared = set(self._numeric_cols) & set(
                current_data.select_dtypes(include=[np.number]).columns
            )
            features = sorted(shared)

        results: list[DriftResult] = []
        for feat in features: 
            if feat not in self.reference.columns or feat not in current_data.columns:
                continue

            ref_arr = self.reference[feat].dropna().to_numpy(dtype=float)
            cur_arr = current_data[feat].dropna().to_numpy(dtype=float)

            if len(ref_arr) == 0 or len(cur_arr) == 0:
                continue

            ref_mean = float(np.mean(ref_arr))
            cur_mean = float(np.mean(cur_arr))

            denom = abs(ref_mean) if abs(ref_mean) > 1e-9 else 1.0
            mean_shift_pct = (cur_mean - ref_mean) / denom * 100

            # PSI
            psi_val = self._compute_psi(ref_arr, cur_arr)
            if psi_val >= self.PSI_CRITICAL:
                psi_severity = "critical"
                psi_drift = True
            elif psi_val >= self.PSI_WARNING:
                psi_severity = "warning"
                psi_drift = True
            else:
                psi_severity = "none"
                psi_drift = False

            results.append(
                DriftResult(
                    feature = feat,
                    method = "PSI",
                    statistic = round(psi_val, 6),
                    p_value = None,
                    is_drift = bool(psi_drift),
                    severity = psi_severity,
                    reference_mean = round(ref_mean, 4),
                    current_mean = round(cur_mean, 4),
                    mean_shift_pct = round(mean_shift_pct, 2),
                )
            )

            # KS Test
            ks_stat, ks_p = ks_2samp(ref_arr, cur_arr)
            ks_drift = bool(ks_p < self.KS_ALPHA)
            ks_severity = "critical" if ks_drift else "none"

            results.append(
                DriftResult(
                    feature = feat,
                    method = "KS",
                    statistic = round(float(ks_stat), 6),
                    p_value = round(float(ks_p), 6),
                    is_drift = ks_drift,
                    severity = ks_severity,
                    reference_mean = round(ref_mean, 4),
                    current_mean = round(cur_mean, 4),
                    mean_shift_pct = round(mean_shift_pct, 2),
                )
            )

        return results

    def check_label_drift(
        self,
        reference_labels: np.ndarray,
        current_labels: np.ndarray,
    ) -> DriftResult:
        """
        Detect label (concept) drift using the KS test.

        Parameters
        reference_labels: Training-time target values
        current_labels: Production target values observed during monitoring window

        Returns
        DriftResult: A single result for the label distribution
        """

        ref_arr = np.asarray(reference_labels, dtype=float)
        cur_arr = np.asarray(current_labels, dtype=float)

        ks_stat, ks_p = ks_2samp(ref_arr, cur_arr)
        is_drift = bool(ks_p < self.KS_ALPHA)

        ref_mean = float(np.mean(ref_arr))
        cur_mean = float(np.mean(cur_arr))
        denom = abs(ref_mean) if abs(ref_mean) < 1e-9 else 1.0
        mean_shift_pct = (cur_mean - ref_mean) / denom * 100

        return DriftResult(
            feature = "demand",
            method = "KS",
            statistic = round(float(ks_stat), 6),
            p_value = round(float(ks_p), 6),
            is_drift = is_drift,
            severity = "critical" if is_drift else "none",
            reference_mean = round(ref_mean, 4),
            current_mean = round(cur_mean, 4),
            mean_shift_pct = round(mean_shift_pct, 2),
        )

    def generate_report(
        self,
        current_data: pd.DataFrame,
        current_labels: np.ndarray | None = None,
    ) -> MonitoringReport:
        """
        Run all drift checks and return an aggregated monitoring report

        Parameters:
        current_data: Production feature observations for monitoring window
        current_labels: Observed target values (demand) for the monitoring window. Pass ``None`` to skip label drift detection

        Returns:
        MonitoringReport: Aggregated findings with a human-readable summary
        """

        feature_results = self.check_feature_drift(current_data)

        label_result: DriftResult | None = None

        if current_labels is not None:
            ref_labels = self.reference["demand"].dropna().to_numpy(dtype=float)
            label_result = self.check_label_drift(ref_labels, current_labels)

        overall_drift = bool(any(r.is_drift for r in feature_results))

        if label_result is not None:
            overall_drift = overall_drift or bool(label_result.is_drift)

        retraining_recommended = bool(
            any(
                r.method == "PSI" and r.severity =="critical" for r in feature_results
            )
        )

        if label_result is not None:
            retraining_recommended = retraining_recommended or bool(label_result.is_drift)

        # Human Readable Summary
        critical_features = [
            r.feature
            for r in feature_results
            if r.method == "PSI" and r.severity == "critical"
        ]
        warning_features = [
            r.feature
            for r in feature_results
            if r.method == "PSI" and r.severity == "warning"
        ]

        summary_parts: list[str] = []
        if not overall_drift:
            summary_parts.append(
                "No significant drift detected.  Model distirbutions appear stable."
            )
        else:
            if critical_features:
                summary_parts.append(
                    f"CRITICAL drift (PSI >= 0.20) in : {', '.join(critical_features)}."
                )
            if warning_features:
                summary_parts.append(
                    f"WARNING drift (0.10 <= PSI < 0.20) in : {', '.join(warning_features)}."
                )

            if label_result and label_result.is_drift:
                summary_parts.append(
                    f"Label drift detected (KS p={label_result.p_value:.4f});"
                    f"demand mean shifted {label_result.mean_shift_pct:+.1f}%."
                )
            if retraining_recommended:
                summary_parts.append("Retraining is recommended.")

        return MonitoringReport(
            evaluated_at = datetime.now(timezone.utc).isoformat(),
            n_reference_samples = len(self.reference),
            n_current_samples = len(current_data),
            feature_drift = feature_results,
            label_drift = label_result,
            overall_drift_detected = overall_drift,
            retraining_recommended = retraining_recommended,
            summary = " ".join(summary_parts),
        )

    def plot_drift(
        self,
        current_data: pd.DataFrame,
        features: list[str],
    ) -> "matplotlib.figure.Figure": 
        """
        Generate side-by-side histograms for reference vs current distributions.
        
        Each subplot compares the reference and current distributions for one feature.
        Subplots with detected PSI drift are highlighted with a red border; stable
        features have a green border.

        Parameters:
        current_data: Production observations
        features: features to plot

        Returns:
        matplotlib.figure.Figure: Completed figure (caller is responsible for saving/showing)
        """

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n = len(features)

        ncols = min(3, n)
        nrows = (n + ncols -1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
        )
        fig.suptitle("Feature Distribution: Reference vs Current", fontsize = 14, y=1.02)

        # Pre-compute PSI drift status for each feature
        psi_results = {
            r.feature: r
            for r in self.check_feature_drift(current_data, features)
            if r.method == "PSI"
        }

        for i, feat in enumerate(features):
            row, col = divmod(i, ncols)
            ax = axes[row][col]

            ref_arr = self.reference[feat].dropna().to_numpy(dtype=float)
            cur_arr = current_data[feat].dropna().to_numpy(dtype=float)

            bin_min = min(ref_arr.min(), cur_arr.min())
            bin_max = max(ref_arr.max(), cur_arr.max())
            bins = np.linspace(bin_min, bin_max, 31)

            ax.hist(ref_arr, bins=bins, alpha=0.6, label="Reference", color="steelblue", density=True)
            ax.hist(cur_arr, bins=bins, alpha=0.6, label="Current", color="orange", density=True)

            drift_result = psi_results.get(feat)
            border_color = "green"
            title_suffix = ""
            if drift_result:
                psi_val = drift_result.statistic
                title_suffix = f"  PSI={psi_val:.3f}"
                if drift_result.severity =="critical":
                    border_color = "red"
                elif drift_result.severity == "warning":
                    border_color = "orange"

            ax.set_title(f"{feat}{title_suffix}, fontsize=10")
            ax.legend(fontsize=8)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

        # Hide unused axes
        for i in range(n, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].set_visible(False)

        # Legend patch
        ref_patch = mpatches.Patch(color="steelblue", alpha=0.6, label="Reference")
        cur_patch = mpatches.Patch(color="orange", alpha=0.6, label="Current")
        fig.legend(
            handles = [ref_patch, cur_patch],
            loc = "lower_center", 
            ncol = 2,
            bbox_to_anchor = (0.5, -0.02),
            fontsize = 10,
        )

        fig.tight_layout()
        return fig