"""pii_identifier - hybrid regex + NER PII detection for HR/legal documents."""

from .types import PIIType, DetectionMethod, PIIMatch
from .patterns import RegexMatcher
from .ner import NERMatcher
from .pipeline import PIIIdentifier
from .injector import HRDocumentInjector

__all__ = [
    "PIIType",
    "DetectionMethod",
    "PIIMatch",
    "RegexMatcher",
    "NERMatcher",
    "PIIIdentifier",
    "HRDocumentInjector",
]

