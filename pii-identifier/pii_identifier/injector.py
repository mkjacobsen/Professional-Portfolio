"""HR Document Generator with Injected PII

Each generator method returns a ``(document_text, ground_truth)`` tuple where
``ground_truth`` is a list of dicts describing every PII value that was
deliberately inserted:

    {"text": "Jane Dow", "pii_type": "person_name", "start": 42, "end": 50}

Offsets are computed precisely by building documents as a sequence of
``(segment_text, pii_type_or_None)`` tuples, then joining them while
accumulating the running character position.  This is more reliable than
``str.find()`` because it handles the case where the same value happens to
appear more than once in the template prose.
"""

from __future__ import annotations

import random
from faker import Faker

from .types import PIIType

# Type alias for ground-truth records
GroundTruth = list[dict]

# Document Builder Helper

class _DocBuilder:
    """Accumulate text segments and track PII offsets precisely."""

    def __init__(self) -> None:
        self._parts: list[tuple[str, str | None]] = [] # (text, pii_type_value | None)

    def add(self, text: str, pii_type: PIIType | None = None) -> "_DocBuilder":
        """Append a text segment, optionally tagged as PII."""
        self._parts.append((text, pii_type.value if pii_type is not None else None))
        return self

    def build(self) -> tuple[str, GroundTruth]:
        """Join all segments and return ``(full_text, ground_truth_list)``"""
        full_text = ""
        ground_truth: GroundTruth = []
        for segment, pii_type_value in self._parts:
            if pii_type_value is not None:
                start = len(full_text)
                end = start + len(segment)
                ground_truth.append(
                    {"text": segment, "pii_type": pii_type_value, "start": start, "end": end}
                )

            full_text += segment
        return full_text, ground_truth


# Main Injector Class

class HRDocumentInjector:
    """Generate realistic HR documents with deterministic, seeded PII

    Parameters

    seed: 
        Random seed passed to both :class: `~faker.Faker` and Python's 
        ``random`` module, ensuring reproducible output across runs.
    """

    def __init__(self, seed: int = 42) -> None:
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)

    # Document Generators

    def offer_letter(self) -> tuple[str, GroundTruth]:
        """~300-word employment offer letter with seeded PII."""
        name = self.fake.name()
        street = self.fake.street_address()
        city = self.fake.city()
        state = self.fake.state_abbr()
        zipcode = self.fake.zipcode()
        address = f"{street}, {city}, {state} {zipcode}"
        ssn = self.fake.ssn()
        start_date = self.fake.date_between(start_date="+14d", end_date="+60d").strftime("%B %d, %Y")
        salary = f"${random.randint(70,200) * 1_000:,}"

        