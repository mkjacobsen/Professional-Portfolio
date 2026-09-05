"""
tensor_anomaly - IoT sensor anomaly detection via CP tensor decomposition.
"""

from .data import TelemetryGenerator
from .decomposition import CPDecomposition
from .detector import AnomalyDetector

__all__ = ["TelemetryGenerator", "CPDecomposition", "AnomalyDetector"]