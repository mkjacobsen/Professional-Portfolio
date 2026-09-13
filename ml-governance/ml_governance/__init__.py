"""
ml_governance - ML model governance framework for regulated industries
"""

from .model_card import (
    ModelCard,
    PerformanceMetrics,
    FairnessSlice,
    FeatureImportance,
)

from .registry import ModelRegistry, RegistryEntry
from .evaluation import ModelEvaluator
from .explainability import SHAPExplainer
from .risk import RiskScorer
from .reporting import ReportGenerator

__all__ = [
    "ModelCard",
    "PerformanceMetrics",
    "FairnessSlice",
    "FeatureImportance",
    "ModelRegistry",
    "RegistryEntry",
    "ModelEvaluator",
    "SHAPExplainer",
    "RiskScorer",
    "ReportGenerator",
]