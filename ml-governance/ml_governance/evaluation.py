"""
evaluation.py - Performance metrice and fairness slice computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model_card import FairnessSlice, PerformanceMetrics

_MIN_SLICE_SAMPLES = 30

class ModelEvaluator:
    """Compute performance metrics and fairness slices for a binary clasaifier."""

    # Performance
    def compute_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> PerformanceMetrics:
        """
        Compute the full set of :class:`PerformanceMetrics`.

        Parameters

        y_true:
            Ground-truth binary labels (0/1)
        y_pred: 
            Hard binary predictions (0/1)
        y_prob:
            Predicted probability of the positive class
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        gini = 2.0 * roc_auc - 1.0

        # Credit KS: max separation between the score distirbutions of defaulters and non-defaulters.
        pos_scores = y_prob[y_true == 1]
        neg_scores = y_prob[y_true == 0]

        ks_result = ks_2samp(pos_scores, neg_scores)
        ks_statistic = float(ks_result.statistic)

        return PerformanceMetrics(
            accuracy = accuracy,
            precision = precision,
            recall = recall,
            f1 = f1,
            roc_auc = roc_auc,
            gini = gini,
            ks_statistic = ks_statistic,
        )

    # Fairness Slices

    def compute_fairness_slices(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        slice_features: list[str],
    ) -> list(FairnessSlice):
        """
        Compute per-value fairness metrics for each feature in "slice_features".

        A slice is included only when it has >= 30 samples.
        In credit risk, "approval" = model predicts non-default (y_pred = 0)
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        overall_approval_rate = float((y_pred == 0).mean())

        slices: list[FairnessSlice] = []

        for feature in slice_features:
            if feature not in X.columns:
                continue
            for value in sorted(X[feature].unique()):
                mask = X[feature] == value
                n = int(mask.sum())
                if n < _MIN_SLICE_SAMPLES:
                    continue

                yt = y_true[mask]
                yp = y_pred[mask]

                approval_rate = float((yp == 0).mean())

                # FPR: of actual negatives (non-defaults), fraction predicted positive
                actual_neg = yt == 0

                fpr = (
                    float((yp[actual_neg] == 1).mean()) if actual_neg.sum() > 0 else 0.0
                )

                # FNR: of actual positives (defaults), fraction predicted negative (non-default)
                actual_pos = yt == 1
                fnr = (
                    float((yp[actual_pos] == 0).mean()) if actual_pos.sum() > 0 else 0.0
                )

                disparate_impact = (
                    approval_rate / overall_approval_rate
                    if overall_approval_rate > 0
                    else 0.0
                )

                slices.append(
                    FairnessSlice(
                        slice_name = f"{feature}={value}",
                        feature = feature,
                        value = str(value),
                        n_samples = n,
                        approval_rate = approval_rate,
                        fpr = fpr,
                        fnr = fnr,
                        disparate_impact = disparate_impact,
                    )
                )

        return slices

    # KS Curve (Decile table)

    def compute_ks_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict:
        """
        Compute cumulative good/bad rates at decile thresholds.

        Returns a dict with keys ``thresholds``, ``cumulative_good``, ``cumulative_bad``, and ``ks_values``.
        """

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        order = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[order]

        n_total = len(y_true_sorted)
        n_pos = y_true_sorted.sum()
        n_neg = n_total - n_pos

        thresholds = []
        cumulative_good = []
        cumulative_bad = []
        ks_values = []

        for decile in range(1, 11):
            cutoff = int(n_total * decile / 10)
            subset = y_true_sorted[:cutoff]
            cumulative_bad_rate = float(subset.sum() / n_pos) if n_pos > 0 else 0.0
            cumulative_good_rate = float((cutoff - subset.sum()) / n_neg) if n_neg > 0 else 0.0
            ks = abs(cumulative_bad_rate - cumulative_good_rate)

            thresholds.append(decile * 10)
            cumulative_good.append(round(cumulative_good_rate, 4))
            cumulative_bad.append(round(cumulative_bad_rate, 4))
            ks_values.append(round(ks, 4))

        return {
            "thresholds": thresholds,
            "cumulative_good": cumulative_good,
            "cumulative_bad": cumulative_bad,
            "ks_values": ks_values
        }
        

