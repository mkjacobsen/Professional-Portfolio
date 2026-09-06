"""Tests for ner.py - spaCy NER wrapper.

These tests requore ``en_core_web_sm`` to be installed:
    python -m spacy download en_core_web_sm
"""

import pytest

from pii_identifier.ner import NERMatcher
from pii_identifier.types import PIIType, DetectionMethod

@pytest.fixture(scope="module")
def ner():
    """Load the NER model once for the entire module to keep tests fast."""
    return NERMatcher()

@pytest.mark.requires_spacy_model()
class TestEntityMapping:
    def test_person_detected(self, ner):
        """A clear human name should be tagged as PERSON_NAME."""
        text = "The contract was signed by Jennifer Walters on behalf of the employee."
        matches = ner.match(text)
        person_matches = [m for m in matches if m.pii_type ==PIIType.PERSON_NAME]
        assert len(person_matches) >= 1

        # The detected name should overlap with the injected name.
        texts = [m.text for m in person_matches]
        assert any("Jennifer" in t or "Walters" in t for t in texts)

    def test_org_detected(self, ner):
        """A corporate entity should be tagged as ORGANIZATION."""
        text = "The employee was previously employed at Acme Corporation."
        matches = ner.match(text)
        org_matches = [m for m in matches if m.pii_type == PIIType.ORGANIZATION]
        assert len (org_matches) >= 1

    def test_person_and_org_in_same_sentence(self, ner):
        """Both PERSON_NAME and ORGANIZATION can appear in one sentence."""
        text = "Robert Chen signed an NDA with GlobalTech Industries last quarter."
        matches = ner.match(text)
        types_found = [m.pii_type for m in matches]
        # At minimum we expect person or org; both is ideal.
        assert types_found & (PIIType.PERSON_NAME, PIIType.ORGANIZATION)

    def test_method_is_ner(self, ner):
        text = "Please return the form to Sarah Johnson in HR."
        matches = ner.match(text)
        for m in matches:
            assert m.method == DetectionMethod.NER

    def test_confidence_in_range(self, ner):
        text = "Michael Scott submitted the paperwork."
        matches = ner.match(text)
        for m in matches:
            assert 0.0 < m.confidence <= 1.0

    def test_noisy_labels_suppressed (self, ner):
        """DATE and MONEY entities should NOT appear in the output."""
        text = "The salary of $120,000 was effective January 2024."
        matches = ner.match(text)
        # None of the matches should have DATE or MONEY mapped types.
        # (These labels are dropped by the NERMatcher.)
        for m in matches:
            assert m.pii_type in (
                PIIType.PERSON_NAME,
                PIIType.ADDRESS,
                PIIType. ORGANIZATION,
            )