"""
schema.py - Core dataclass definitions for synthetic health records.

All identifiers are UUIDs generated at construction time if not provided.
Validation is enforced via __post_init__ where meaningful constraints exist.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

def _new_uuid() -> str:
    return str(uuid.uuid4())

@dataclass
class ICD10Code:
    """An ICD-10-CM diagnosis code with its human-readable description."""

    code: str
    description: str

    def __post_init__(self) -> None:
        if not self.code: 
            raise ValueError("ICD10Code.code must not be empty")
        if not self.description:
            raise ValueError("ICD10Code.description must not be empty")

    def __repr__(self) -> str:
        return f"ICD10Code({self.code[r]}, {self.description!r})"

@dataclass
class Patient:
    """Demographic record for a synthetic patient."""

    first_name: str
    last_name: str
    dob: date
    sex: Literal["M","F"]
    race: str
    zip_code: str
    patient_id: str = field(default_factory=_new_uuid)

    def __post_init__(self) -> None:
        if self.dob >= date.today():
            raise ValueError(f"Patient dob must be in the past")
        if self.sex not in ("M", "F"):
            raise ValueError(f"Sex must be 'M' or 'F', got {self.sex!r}")
        if not self.zip_code:
            raise ValueError("zip_code must not be empty")

    @property
    def age(self) -> int:
        today = date.today()
        return(
            today.year-self.dob.year
            - ((today.month, today.day) < (self.dob.month, self.dob.day))
        )

@dataclass
class Encounter:
    """A single clinical encounter linked to a patient."""

    patient_id: str
    date: date
    encounter_type: Literal["outpatient","inpatient","emergency"]
    diagnoses: list[ICD10Code]
    encounter_id: str = field(default_factory=_new_uuid)

    def __post_init__(self) -> None:
        if self.encounter_type not in ("outpatient", "inpatient", "emergency"):
            raise ValueError(f"Encounter type must be one of outpatient/inpatient/emergency")
        if not isinstance(self.diagnoses, list):
            raise TypeError("diagnoses must be a list of ICD10Code")


@dataclass
class LabResult:
    """A single laboratory test result linked to an encounter."""

    patient_id: str
    encounter_id: str
    test_name: str
    value: float
    unit: str
    reference_low: float
    reference_high: float
    collection_date: date
    result_id: str = field(default_factory=_new_uuid)

    def __post_init__(self) -> None:
        if self.reference_low >= self.reference_high:
            raise ValueError(
                "reference_low must be less than reference_high"
            )

    @property
    def is_abnormal(self) -> bool:
        return self.value < self.reference_low or self.value > self.reference_high

@dataclass
class ClinicalNote:
    """A free-text clinical note linked to an encounter."""

    patient_id: str
    encounter_id: str
    note_type: Literal["soap","discharge_summary","referral"]
    text: str
    created_at: datetime
    note_id: str = field(default_factory=_new_uuid)

    def __post_init__(self) -> None:
        if self.note_type not in ("soap", "dischage_summary", "referral"):
            raise ValueError(
                "note_type must be soap/discharge_summary/referral"
            )

        if not self.text.strip():
            raise ValueError("ClinicalNote.text must not be blank")

        