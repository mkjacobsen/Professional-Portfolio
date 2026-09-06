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
class TestRedact:
    def test_ssn_redacted(self, regex_only):
        text = "SSN: 123-45-6789"
        redacted = regex_only.redact(text)
        assert "123-45-6789" not in redacted
        assert " (REDACTED) " in redacted

    def test_email_redacted(self, regex_only):
        text = "Email: user@domain.com"
        redacted = regex_only.redact(text)
        assert "user@domain.com" not in redacted

    def test_custom_replacement(self, regex_only):
        text = "Phone: 800-555-1234"
        redacted = regex_only.redact(text, replacement="****")
        assert "800-555-1234" not in redacted
        assert "****" in redacted

    def test_multiple_pii_all_redacted(self,regex_only):
        text = "Name: John Doe, SSN: 111-22-3333, email: jftest.com"
        redacted = regex_only.redact(text)
        assert "111-22-3333" not in redacted
        assert "jftest.com" not in redacted

    def test_non_pii_preserved(self, regex_only):
        text = "The salary is $90,000 per year."
        redacted = regex_only.redact(text)
        assert "90,000" in redacted
        assert "per year" in redacted

# Overlap Resolution: Regex Wins
@pytest.mark.requires_spacy_model
class TestOverlapResolutionWithNER:
    def test_regex_beats_ner_on_overlap(self):
        """When regex and NER both fire on the same span, only the 
        regex match should survive in the merged output."""
        ident = PIIIdentifier(use_ner=True)

        #Craft a text where SSN might accidentally be detected by NER too.
        text = "Social Security Number: 456-78-9012"
        matches = ident.identify(text)

        ssn_matches = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) >= 1

        #None of the matches covering the SSN span should be NER-sourced
        ssn_match = ssn_matches[0]
        overlapping = [
            m for m in matches
            if m.method == DetectionMethod.NER
            and max(m.start, ssn_match.start) < min(m.end, ssn_match.end)
        ]
        assert overlapping == [], (
            f"NER match(es) overlap the regex SSN match: {overlapping}"
        )

class TestSpansOverlapHelper:
    """Pure unit tests for the _spans_overlap utility - no spacy needed"""
    def _make(self, start: int, end: int) -> PIIMatch:
        return PIIMatch(
            pii_type = PIIType.SSN, text="x",
            start = start, 
            end = end,
            method = DetectionMethod.REGEX, 
            confidence = 1.0
        )

    def test_overlapping(self):
        assert _spans_overlap(self._make(0, 10), self._make(5, 15))
        assert _spans_overlap(self._make(5, 15), self._make(0, 10))

    def test_adjacent_does_not_overlap(self):
        assert not _spans_overlap(self._make(0, 5), self._make(5, 10))

    def test_completely_separate(self):
        assert not _spans_overlap(self._make(0, 5), self._make(10, 20))

    def test_containment_overlaps (self):
        assert _spans_overlap(self._make(0, 20), self._make(5, 10))

# report () - uses regex_only fixture (no spacy needed)
class TestReport:
    def test_report_structure(self, regex_only):
        text = "SSN: 123-45-6789"
        report = regex_only.report(text)
        assert "matches" in report
        assert "pii_types_found" in report
        assert "total_count" in report
        assert "redacted_text" in report

    def test_report_counts_match (self, regex_only):
        text = "Phone: 800-555-1234 email: a@b.com"
        report = regex_only.report(text)
        assert report["total_count"] == len(report["matches"])

    def test_report_types_are_strings(self, regex_only):
        text = "SSN: 111-22-3333"
        report = regex_only.report(text)
        for t in report["pii_types_found"]:
            assert isinstance(t, str)
