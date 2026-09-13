"""
model_card.py - ModelCard dataclass and supporting types.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import json

@dataclass
class PerformanceMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    gini: float # 2 * AUC - 1; standard credit rísk discriminator
    ks_statistic: float # KS score: max separation between good/bad score distribution

@dataclass
class FairnessSlice:
    slice_name: str
    feature: str
    value: str
    n_samples:int
    approval_rate:float
    fpr: float # false positive rate in this slice
    fnr: float # false negative rate in thís slice
    disparate_impact: float # slice_approval_rate / overall_approval_rate

@dataclass
class FeatureImportance:
    feature_name: str
    shap_mean_abs: float
    rank: int

@dataclass
class ModelCard:
    model_id: str
    model_name: str
    version: str
    description: str
    model_type: str #e.g. "GradientBoostingClassifier"
    training_date: str
    training_dataset_size: int
    feature_names: list[str]
    performance: PerformanceMetrics
    fairness_slices: list[FairnessSlice] = field(default_factory=list)
    feature_importances: list[FeatureImportance] = field(default_factory=list)
    risk_score: Optional[float] = None #populated by RiskScorer
    risk_factors: dict = field(default_factory=dict) #populated by RiskScorer
    intended_use: str = ""
    limitations: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Serialization Helpers
    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCard":
        """Reconstruct a ModelCard from a plain dict (e.g. after JSON parse)"""
        d = dict(d) #shallow copy to pop/replace keys

        performance_raw = d.pop("performance")
        performance = PerformanceMetrics(**performance_raw)

        fairness_raw = d.pop("fairness_slices", [])
        fairness_slices = [FairnessSlice(**s) for s in fairness_raw]

        importances_raw = d.pop("feature_importances", [])
        feature_importances = [FeatureImportance(**fi) for fi in importances_raw]

        return cls(
            performance = performance,
            fairness_slices = fairness_slices,
            feature_importances = feature_importances,
            **d,
        )

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelCard":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(s))