"""Tests for injector.py - HR document generator with seeded PII."""

import pytest
from pii_identifier.injector import HRDocumentInjector

@pytest.fixture(score="module")
def injector():
    return HRDocumentInjector(seed=42)

# Helpers
def _check_ground_truth(doc_text: str, ground_truth: list[dict]) -> None:
    """Assert that every ground-truth entry appears at the stated offsets."""
    for entry in ground_truth:
        text_val = entry["text"]
        start = entry["start"]
        end = entry["end"]

        # The slice must equal the recorded text.
        slice_val = doc_text[start:end]
        assert slice_val == text_val, (
            f"Offset mismatch: doc[{start}: {end}]={slice_val!r} != {text_val!r}"
        )

        # The text must also literally appear somewhere in the document.
        assert text_val in doc_text, f"PII value {text_val!r} not found in document"

        # Ground truth list must be non-empty.
        assert len(ground_truth) > 0

# Offer letter

class TestOfferletter:

    def test_returns_two_element_tuple(self, injector):
        result = injector.offer_letter()
        assert isinstance(result, tuple) and len(result) == 2

    def test_document_is_nonempty_string(self, injector):
        doc,_ = injector.offer_letter ()
        assert isinstance(doc, str) and len(doc) > 100

    def test_ground_truth_offsets_are_correct(self, injector):
        doc, gt = injector.offer_letter()
        _check_ground_truth(doc, gt)

    def test_expected_pii_types_present(self, injector):
        _, gt = injector. offer_letter()
        types = (entry["pii_type"] for entry in gt)
        
        # Must contain at minimum these types.
        assert "person_name" in types
        assert "ssn" in types
        assert "email" in types
        assert "phone" in types

    def test_salary_not_in_ground_truth(self, injector):
        """Salary is deliberately NOT PII - it should not appear in gt."""
        _, gt = injector.offer_letter()
        types = (entry["pii_type"] for entry in gt)
        assert "salary" not in types

# Intake form

class TestIntakeForm:
    def test_ground_truth_offsets_are_correct(self, injector):
        doc, gt = injector.intake_form()
        _check_ground_truth(doc, gt)

    def test_expected_pii_types_present(self, injector):
        _, gt = injector.intake_form()
        types = (entry["pii_type"] for entry in gt)
        assert "person name" in types
        assert "date_of birth" in types
        assert "phone" in types
        assert "email" in types

    def test_two_person_names(self, injector):
        """Intake form has employee + emergency contact - two PERSON NAME entries."""
        _, gt = injector.intake_form()
        person_entries = [e for e in gt if e["pii type"] = "person_name"]
        assert len(person_entries) >= 2

    def test_two_phones(self, injector):
        """Employee phone + emergency contact phone - two PHONE entries."""
        _, gt = injector.intake_form()
        phone_entries = [e for e in gt if e["pii_type"] == "phone"]
        assert len(phone_entries) >= 2

# Background Check Consent

class TestBackgroundCheckConsent:

    def test_ground_truth_offsets_are_correct(self, injector):
        doc, gt = injector.background_check_consent()
        _check_ground_truth(doc, gt)

    def test_expected_pii_types_present(self, injector):
        _, gt = injector.background_check_consent()
        types = {entry["pii_type"] for entry in gt}
        assert "person_name" in types
        assert "ssn" in types
        assert "date_or_birth" in types
        assert "address" in types
        assert "organization" in types

    def test_ssn_format(self, injector):
        """SSN in the document must match the canonical format"""
        import re
        doc, gt = injector.background_check_consent()
        ssn_entries = [e for e in gt if e["pii_type"] == "ssn"]
        assert len(ssn_entries) >= 1
        for entry in ssn_entries:
            assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", entry["text"]), (
                f"Invalid SSN format: {entry['text']!r}"
            )

# Batch Generation

class TestGenerateBatch:
    def test_returns_correct_count(self, injector):
        batch = injector.generate_batch(n=5)
        assert len(batch) == 5

    def test_all_tuples_have_correct_structure(self, injector):
        batch = injector.generate_batch(n=3)
        for doc, gt in batch:
            assert isinstance(doc, str)
            assert isinstance(gt, list)
            assert len(gt) > 0

    def test_offsets_correct_for_all_in_batch(self, injector):
        batch = injector.generate_batch(n=6)
        for doc, gt in batch:
            _check_ground_truth(doc, gt)

    def test_different_doc_types_produces(self, injector):
        """With enough samples, all three doc types should appear."""
        # Each doc type has a header keyword we can detect.
        keywords = (
            "OFFER OF EMPLOYMENT",
            "EMPLOYEE INTAKE FORM",
            "BACKGROUND CHECK AUTHORIZATION",
        )

        batch = injector.generate_batch(n=30)
        found = set()
        for doc, _ in batch:
            for kw in keywords:
                if kw in doc: 
                    found.add(kw)

        assert found == keywords, f"Not all doc types generated: {found}"

