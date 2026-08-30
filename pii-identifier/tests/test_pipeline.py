"""
Tests for hybrid PIIIdentifier
"""

import pytest
from pii_identifier.pipeline import PIIIdentifier, _spans_overlap
from pii_identifier.types import PIIType, DetectionMethod, PIIMatch

# Regex-only identifier - no spaCy model needed
@pytest.fixture(scope="module")
def regex_only():
    return PIIIdentifier(use_ner=False)

# Full Identifier - requires en-core_web_sm
@pytest.fixture(scope="module")
def identifier():
    return PIIIdentifier(use_ner=True)

# Identify tests - requires spaCy model

@pytest.mark.requires_spacy_model
class TestIdentify:
    def test_snn_detected(self, identifier):
        text = "Please submit your SSN (123-45-6789) to HR before Friday."
        matches = identifier.identify(text)
        assert any(m.pii_type == PIIType.SSN for m in matches)

    def test_email_detected(self, identifier):
        text = "Send your documents to onboarding@company.com by end of day."
        matches = identifier.identify(text)
        assert any(m.pii_type == PIIType.EMAIL for m in matches)

    def test_snn_and_email_both(self, identifier):
        text = (
            "Employee SSN: 987-65-4321"
            "For questions, contact hr@example.org"
        )
        matches = identifier.identify(text)
        types = {m.pii_type for m in matches}
        assert PIIType.SSN in types
        assert PIIType.EMAIL in types

    def test_results_sorted_by_start(self, identifier):
        text = "email: a@b.com ssn: 111-22-3333 phone: 999-888-7777"
        matches = identifier.identify(text)
        offsets = [m.start for m in matches]
        assert offsets == sorted(offsets)

    def test_min_confidence_filter(self):
        """Matches below min_confidence should be excluded."""
        ident = PIIIdentifier(use_ner=True, min_confidence=0.99)
        text = "Call Alice Smith at 415-555-0100."
        matches = ident.identify(text)
        #phone number should have confidence 1 and pass
        assert any(m.pii_type == PIIType.Phone for m in matches)
        # NER matches should be filtered out
        ner_matches = [m for m in matches if m.method == DetectionMethod.NER]
        assert len(ner_matches) == 0

# Regex Identify Tests

class TestIdentifyRegexOnly:
    def test_ssn_detected(self, regex_only):
        text = "Please submit your SSN (123-45-6789) to HR before Friday."
        matches = regex_only.identify(text)
        assert any(m.pii_type == PIIType.SSN for m in matches)

    def test_email_detected(self, regex_only):
        text = "Send your documents to onboarding@company.com by end of day."
        matches = regex_only.identify(text)
        assert any(m.pii_type == PIIType.EMAIL for m in matches)

    def test_ssn_and_email_both(self, regex_only):
        text = (
            "Employee SSN: 987-65-4321"
            "For questions contact hr@example.org."
        )
        matches = regex_only.identify(text)
        types = {m.pii_type for m in matches}
        assert PIIType.SSN in types
        assert PIIType.EMAIL in types

    def test_results_sorted_by_start(self, regex_only):
        text = "email: a@b.com ssn: 111-22-3333 phone: 999-888-77777"
        matches = regex_only.identify(text)
        offsets = [m.start for m in matches]
        assert offsets == sorted(offsets)

# Regex Only Redact Function

