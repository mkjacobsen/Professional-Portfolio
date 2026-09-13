"""
risk.py - Structured model risk socrer (0-100 scale)
"""

from __future__ import annotations

from .model_card import ModelCard

class RiskScorer:
    """
    Compute structure model risk score across four dimensions from 0-100

    Each dimension contributes up to 25 points; higher indicates greater risk

    Dimensions

    Performance (0-25):
        Penalizes low Gini coefficient and low F1 score
    Fairness (0-25):
        Penalizes disparate-impact violations (ratio < 0.8 threshold)
    Data (0-25):
        Penalizes insufficient training dataset size
    Stability (0-25):
        Penalizes low KS statistic
    """

    GINI_THRESHOLD_HIGH: float = 0.3 #below this = maximum performance risk
    GINI_THRESHOLD_LOW_RISK: float = 0.5 #above this = zero performance risk
    DISPARATE_IMPACT_THRESHOLD: float = 0.8 #4/5ths rule
    MIN_TRAINING_SIZE: int = 1_000
    TARGET_TRAINING_SIZE: int = 10_000
    KS_THRESHOLD_HIGH: float = 0.2 #below this = maximum stability risk
    KS_THRESHOLD_LOW_RISK: float = 0.4 #above this = zero stability risk

    # Public API

    def score(self, card: ModelCard) -> tuple[float, dict]:
        """
        Compute the overall risk score and a breakdown dict

        Returns

        tuple[float, dict]
            ``(total_score, risk_factors)`` where *total_score* is 0-100 and *risk_factors* is ::
            {
                "performance": float,
                "fairness": float,
                "data": float,
                "stability": float,
                "details": {
                    "performance": str,
                    "fairness": str,
                    "data": str,
                    "stability": str,                    
                }
            }
        """

        perf_score, perf_detail = self._performance_risk(card)
        fair_score, fair_detail = self._fairness_risk(card)
        data_score, data_detail = self._data_risk(card)
        stab_score, stab_detail = self._stability_risk(card)

        total = perf_score + fair_score + data_score + stab_score

        risk_factors = {
            "performance": round(perf_score, 2),
            "fairness": round(fair_score, 2),
            "data": round(data_score, 2),
            "stability": round(stab_score, 2),
            "details": {
                "performance": perf_detail,
                "fairness": fair_detail,
                "data": data_detail,
                "stability": stab_detail,
            },
        }

        return round(total, 2), risk_factors

    def risk_label(self, score: float) -> str:
        """Map numeric risk score to categorical label"""

        if score <= 25:
            return "Low"
        elif score <= 50:
            return "Medium"
        elif score <= 75:
            return "High"
        else:
            return "Critical"

    # Dimension Scorers

    def _performance_risk(self, card: ModelCard) -> tuple[float, str]:
        """
        0-25 points

        Gini < GINI_THRESHOLD_HIGH -> Full 20 points
        Gini between thresholds -> linearly scaled
        F1 < 0.5 = additional 5 points
        """

        gini = card.performance.gini
        f1 = card.performance.f1

        if gini <= self.GINI_THRESHOLD_HIGH:
            gini_score = 20.0
        elif gini >= self.GINI_THRESHOLD_LOW_RISK:
            gini_score = 0.0
        else:
            #Linear Interpolation
            span = self.GINI_THRESHOLD_LOW_RISK - self.GINI_THRESHOLD_HIGH
            gini_score = 20.0 * (1.0 - (gini - self.GINI_THRESHOLD_HIGH) / span)

        f1_score = 5.0 if f1 < 0.5 else 0.0
        total = min(25.0, gini_score + f1_score)

        detail = (
            f"Gini = {gini:.3f} (threshold {self.GINI_THRESHOLD_HIGH});"
            f"F1 = {f1:.3f}; score = {total:.1f}/25"
        )

        return total, detail

    def _fairness_risk(self, card: ModelCard) -> tuple[float, str]:
        """
        0-25 points

        Fraction of fairness slices with disparate_impact < 0.8,
        scaled to 0-25
        """

        slices = card.fairness_slices
        if not slices:
            return 0.0, "No fairness slices available; risk assumed 0."

        violations = [
            s for s in slices
            if s.disparate_impact < self.DISPARATE_IMPACT_THRESHOLD
        ]

        violation_rate = len(violations) / len(slices)
        score = round(25.0 * violation_rate, 2)

        violated_names = [v.slice_name for v in violations]
        detail = (
            f"{len(violations)}/{len(slices)} slices violate disparate impact"
            f"< {self.DISPARATE_IMPACT_THRESHOLD};"
            + (f"violations: {violated_names}" if violated_names else "none")
        )

        return score, detail

    def _data_risk(self, card: ModelCard) -> tuple[float, str]:
        """
        0-25 points

        Training slice < MIN_TRAINING_SIZE = 25 pts
        Linear decay to 0.0 at TARGET_TRAINING_SIZE
        """

        n = card.training_dataset_size

        if n <= 0:
            score = 25.0
        elif n <= self.MIN_TRAINING_SIZE:
            score = 25.0
        elif n >= self.TARGET_TRAINING_SIZE:
            score = 0.0
        else:
            span = self.TARGET_TRAINING_SIZE - self.MIN_TRAINING_SIZE
            score = 25.0 * (1.0 - (n - self.MIN_TRAINING_SIZE) / span)

        detail = (
            f"Training size = {n:,} (min={self.MIN_TRAINING_SIZE:,}, "
            f"target={self.TARGET_TRAINING_SIZE:,}); score={score:.1f}/25"
        )
        return round(score, 2), detail

    def _stability_risk(self, card: ModelCard) -> tuple[float, str]:
        """
        0-25 points

        KS < KS_THRESHOLD_HIGH = 25 pts
        Linear decay to 0 at KS_THRESHOLD_LOW_RISK
        """

        ks = card.performance.ks_statistic

        if ks <= self.KS_THRESHOLD_HIGH:
            score = 25.0
        elif ks >= self.KS_THRESHOLD_LOW_RISK:
            score = 0.0
        else:
            span = self.KS_THRESHOLD_LOW_RISK - self.KS_THRESHOLD_HIGH
            score = 25.0 * (1.0 - (ks - self.KS_THRESHOLD_HIGH) / span)

        detail = (
            f"KS={ks:.3f} (threshold {self.KS_THRESHOLD_HIGH}); "
            f"score = {score:.1f}/25"
        )

        return round(score, 2), detail
