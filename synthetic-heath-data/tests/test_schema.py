"""
test_schema.py - Unit tests for src.schema dataclasses.

Verifies field type, validation behavior, and convenience properties.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta

import pytest

from src.schema import (
    ClinicalNote,
    Encounter,
    ICD10Code,
    LabResult,
    Patient,
)

# ----------------------
# ICD10Code
# ----------------------

class TestICD10Code:

    def test_basic_construction(self):
        dx = ICD10Code("I10", "Essential hypertension")
        assert dx.code == "I10"
        assert dx.description == "Essential hypertension"

    def test_empty_code_raises(self):
        with pytest.raises(ValueError, match="code"):
            ICD10Code("", "Random description")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            ICD10Code("E11.9", "")

# ---------------------
# Patient
# ---------------------

class TestPatient:
    _valid_kwargs = dict(
        first_name="Jane",
        last_name="Doe",
        dob=date(1985, 6, 15),
        sex="F",
        race="White",
        zip_code="20001",
    )

    def test_basic_construction(self):
        p = Patient(**self._valid_kwargs)
        assert p.first_name == "Jane"
        assert p.sex == "F"
        # patient_id should be a valid UUID
        uuid.UUID(p.patient_id)

    def test_patient_id_is_unique(self):
        p1 = Patient(**self._valid_kwargs)
        p2 = Patient(**self._valid_kwargs)
        assert p1.patient_id != p2.patient_id

    def test_age_property(self):
        today = date.today()
        dob = date(today.year - 40, today.month, today.day) - timedelta(days=1)
        p = Patient(
            first_name="A", last_name="B", dob=dob, sex="M", race="X", zip_code="00000"
        )
        assert p.age==40

    def test_future_dob_raises(self):
        future = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="past"):
            Patient(
                first_name="A",
                last_name="B",
                dob=future,
                sex="M",
                race="X",
                zip_code="00000",
            )

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            Patient(
                first_name = "A",
                last_name = "B",
                dob = date.today() - timedelta(days=366),
                sex = "B",
                race = "X",
                zip_code = "00000",
            )

    def test_explicit_patient_id(self):
        custom_id = str(uuid.uuid4())
        p = Patient(patient_id=custom_id, **self._valid_kwargs)
        assert p.patient_id == custom_id

# ----------------------
# Encounter
# ----------------------

class TestEncounter:

    _dx = [ICD10Code("I10", "Hypertension")]

    def test_basic_construction(self):
        enc = Encounter(
            patient_id="pid-1",
            date=date(2024,3,1),
            encounter_type="outpatient",
            diagnoses=self._dx,
        )
        uuid.UUID(enc.encounter_id)
        assert enc.encounter_type =="outpatient"
        assert len(enc.diagnoses) == 1

    def test_invalid_encounter_type_raises(self):
        with pytest.raises(ValueError, match="encounter_type"):
            Encounter(
                patient_id="pid-1",
                date=date(2024,3,1),
                encounter_type="telehealth",
                diagnoses=self._dx,
            )

    def test_diagnoses_must_be_list(self):
        with pytest.raises(TypeError, match="list"):
            Encounter(
                patient_id = "pid-1",
                date = date(2024, 3, 1),
                encounter_type = "outpatient",
                diagnoses = self._dx[0], 
            )

# ------------------------
# LabResult
# ------------------------

class TestLabResult:

    def test_basic_construction(self):
        lab = LabResult(
            patient_id = "pid-1",
            encounter_id = "eid-1",
            test_name = "Glucose",
            value = 95.0,
            unit = "mg/dL",
            reference_low = 70.0,
            reference_high = 100.0,
            collection_date = date(2024,3,1),
        )

        uuid.UUID(lab.result_id)
        assert lab.is_abnormal is False

    def test_abnormal_flag_high(self):
        lab = LabResult(
            patient_id = "pid-1",
            encounter_id = "eid-1",
            test_name = "Glucose",
            value = 115.0,
            unit = "mg/dL",
            reference_low = 70.0,
            reference_high = 100.0,
            collection_date = date(2024,3,1),
        )

        assert lab.is_abnormal is True

    def test_abnormal_flag_low(self):
        lab = LabResult(
            patient_id = "pid-1",
            encounter_id = "eid-1",
            test_name = "Glucose",
            value = 35.0,
            unit = "mg/dL",
            reference_low = 70.0,
            reference_high = 100.0,
            collection_date = date(2024,3,1),
        )

        assert lab.is_abnormal is True

    def test_invalid_reference_range_raises(self):
        with pytest.raises(ValueError, match="reference_low"):
            lab = LabResult(
                patient_id = "pid-1",
                encounter_id = "eid-1",
                test_name = "Glucose",
                value = 95.0,
                unit = "mg/dL",
                reference_low = 200.0,
                reference_high = 100.0,
                collection_date = date(2024,3,1),
            )

# -----------------------------
# ClinicalNote
# -----------------------------

class TestClinicalNote:

    def test_basic_construction(self):
        note = ClinicalNote(
            patient_id = "pid-1",
            encounter_id = "eid-1",
            note_type = "soap",
            text = "S: Patient presents with headache. O: Vitals stable.",
            created_at = datetime(2024,3,1,10,0),
        )

        uuid.UUID(note.note_id)
        assert note.note_type =="soap"

    def test_invalid_note_type_raises(self):
        with pytest.raises(ValueError, match="note_type"):
            note = ClinicalNote(
                patient_id = "pid-1",
                encounter_id = "eid-1",
                note_type = "progress_note",
                text = "Some text",
                created_at = datetime(2024,3,1,10,0),
            )

    def test_blank_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            note = ClinicalNote(
                patient_id = "pid-1",
                encounter_id = "eid-1",
                note_type = "soap",
                text = "  ",
                created_at = datetime(2024,3,1,10,0),
            )

    