"""
test_generators. Tests for each individual generator.
Validates output counts, value ranges, and distributional properties.
"""
from __future__ import annotations
from datetime import date

import pytest

from src.synth_health.generators.encounter import EncounterGenerator
from src.synth_health.generators.lab_values import LabGenerator, _LAB_SPECS
from src.synth_health.generators.notes import NoteGenerator
from src.synth_health.generators.patient import PatientGenerator
from src.schema import ICD10Code, Patient

# PatientGenerator
class TestPatientGenerator:
    def test_generates_correct_count(self):
        gen = PatientGenerator(seed=0)
        patients = gen.generate(50)
        assert len(patients) == 50

    def test_zero_raises(self):
        gen = PatientGenerator(seed=0)
        with pytest.raises(ValueError):
            gen.generate(0)

    def test_sex_distribution_approximately_correct(self):
        gen = PatientGenerator(seed=1, male_fraction=0.5)
        patients = gen.generate(1000)
        males = sum(1 for p in patients if p.sex == "M")
        # With 1000 samples at 50/50, expect 450-550 males
        assert 400 <= males <= 600, f"Unexpected male count: (males) "

    def test_age_within_range(self):
        gen = PatientGenerator(seed=2, age_min=30, age_max=60)
        patients = gen.generate (100)
        for p in patients:
            assert 30 <= p.age <= 61, f"Patient age {p.age} outside range"

    def test_sex_only_m_or_f(self):
        gen = PatientGenerator(seed=3)
        patients = gen. generate (100)
        for p in patients:
            assert p.sex in ("'M", "F")

    def test_all_have_zip_codes(self):
        gen = PatientGenerator(seed=4)
        patients = gen.generate(20)
        for p in patients:
            assert p.zip_code, "zip_code should not be empty"

    def test_invalid_male_fraction_raises(self):
        with pytest.raises(ValueError):
            gen = PatientGenerator (male_fraction=1.5)

    def test_reproducibility(self):
        g1 = PatientGenerator(seed=99)
        g2 = PatientGenerator(seed=99)
        p1 = g1.generate(5)
        p2 = g2.generate(5)

        for a, b in zip(p1, p2):
            assert a.first_name == b.first_name
            assert a.last_name == b.last_name
            assert a.dob == b.dob

#--------------------------
# EncounterGenerator
#--------------------------

_SAMPLE_PATIENTS = [
    Patient(
        first_name="Alice",
        last_name= "Test",
        dob=date (1975, 4, 20),
        sex="F",
        race="White",
        zip_code="10001",
    ),
    Patient(
        first_name= "Bob",
        last_name= "Test",
        dob=date (1960, 11, 5),
        sex="M",
        race="Black or African American",
        zip_code="90210",
    ),
]

_START = date (2022, 1, 1)
_END = date (2024, 12, 31)

class TestEncounterGenerator:
    def test_generates_encounters_for_all_patients(self):
        gen = EncounterGenerator(seed=0)
        encs = gen.generate(_SAMPLE_PATIENTS, _START, _END)
        patient_ids = (e.patient_id for e in encs)
        assert patient_ids == (p.patient_id for p in _SAMPLE_PATIENTS)

    def test_encounter_count_within_range(self):
        gen = EncounterGenerator(
            seed=0, encounters_per_patient_range= (1, 8)
        )
        encs = gen.generate(_SAMPLE_PATIENTS, _START, _END)
        for pid in (p.patient_id for p in _SAMPLE_PATIENTS):
            count = sum(1 for e in encs if e.patient_id == pid)
            assert 1 <= count <= 8, f"Encounter count (count) outside [1,8]"

    def test_encounter_dates_within_range(self):
        gen = EncounterGenerator(seed=0)
        encs = gen.generate(_SAMPLE_PATIENTS, _START, _END)
        for enc in encs:
            assert _START <= enc.date <= _END

    def test_encounter_types_are_valid(self):
        gen = EncounterGenerator(seed=0)
        valid_types = ("outpatient", "inpatient", "emergency")
        encs = gen. generate(_SAMPLE_PATIENTS, _START, _END)
        for enc in encs:
            assert enc.encounter_type in valid_types

    def test_each_encounter_has_at_least_one_diagnosis(self):
        gen = EncounterGenerator(seed=0)
        encs = gen.generate(_SAMPLE_PATIENTS, _START, _END)
        for enc in encs:
            assert len(enc.diagnoses) >= 1

    def test_start_date_after_end_raises(self):
        gen = EncounterGenerator(seed=0)
        with pytest.raises(ValueError):
            gen.generate(_SAMPLE_PATIENTS, _END, _START)

#---------------------------
# LabGenerator
#---------------------------

class TestLabGenerator:

    def _make_encounters(self):
        gen = EncounterGenerator(seed=5)
        return gen.generate(_SAMPLE_PATIENTS, _START, _END)

    def test_lab_fraction_respected(self):
        # Use a larger patient set so the fraction estimate is stable
        gen = PatientGenerator(seed=5)
        many_patients = gen.generate(50)
        enc_gen = EncounterGenerator(seed=5)
        encounters = enc_gen.generate(many_patients, _START, _END)
        n = len(encounters)
        lab_gen = LabGenerator(seed=5, encounter_lab_fraction=0.6)
        labs = lab_gen.generate(encounters)

        # Each encounter with labs produces len (_LAB_SPECS) results 
        unique_encounters_with_labs = len({lab. encounter_id for lab in labs})

        # Allow 20% slack around the 60% target
        assert 0.40 <= unique_encounters_with_labs / n <= 0.80, (
            f"Lab fraction {unique_encounters_with_labs/n: 2f} outside [0.40, 0.80]"
        )

    def test_lab_values_plausible(self):
        encounters = self._make_encounters()
        lab_gen = LabGenerator(seed=6, encounter_lab_fraction=1.0)
        labs = lab_gen.generate(encounters)

        # Build a spec lookup
        spec_map = {s.test_name: s for s in _LAB_SPECS}
        for lab in labs:
            spec = spec_map[lab.test_name]

            # Values should be non-negative
            assert lab.value >= 0, f"{lab.test_name} value {lab.value} < 0"

            # Values should not be wildly out of range (> 10x reference high)
            assert lab.value <= spec.reference_high * 10, (
                f"{lab.test_name} value {lab.value} implausibly large"
            )

    def test_all_lab_tests_covered(self):
        encounters = self._make_encounters()
        lab_gen = LabGenerator(seed=7, encounter_lab_fraction=1.0)
        labs = lab_gen.generate(encounters)
        test_names = (lab.test_name for lab in labs)
        expected = (s.test_name for s in _LAB_SPECS)
        assert test_names == expected

    def test_collection_dates_near_encounter_date (self):
        encounters = self._make_encounters()
        lab_gen = LabGenerator (seed=8, encounter_lab_fraction=1.0)
        labs = lab_gen.generate(encounters)

        enc_map = {e.encounter_id: e for e in encounters}
        from datetime import timedelta
        for lab in labs:
            enc = enc_map[lab.encounter_id]
            delta = (enc.date - lab.collection_date).days
            assert 0 <= delta <= 2, (
                "Collection date (lab. collection_date) too far from encounter (enc.date)"
            )

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            LabGenerator(encounter_lab_fraction=0.0)

# ----------------------
# NoteGenerator
# ----------------------

class TestNoteGenerator:
    def _setup(self):
        patients = _SAMPLE_PATIENTS
        enc_gen = EncounterGenerator(seed=10)
        encounters = enc_gen.generate(patients, _START, _END)
        lab_gen = LabGenerator(seed=10, encounter_lab_fraction=1.0)
        labs = lab_gen.generate(encounters)
        patient_index = {p.patient_id: p for p in patients}
        labs_by_enc: dict[str, list] = {}

        for lab in labs:
            labs_by_enc.setdefault(lab.encounter_id, []).append(lab)

        return encounters, patient_index, labs_by_enc

    def test_one_note_per_encounter(self):
        encounters, patients, labs = self._setup()
        note_gen = NoteGenerator(seed=10)
        notes = note_gen.generate(encounters, patients, labs)
        assert len(notes) == len(encounters)

    def test_note_types_are_valid(self):
        encounters, patients, labs = self._setup()
        note_gen = NoteGenerator(seed=10)
        notes = note_gen.generate(encounters, patients, labs)
        valid_types = ("soap", "discharge_summary", "referral")
        for note in notes:
            assert note.note_type in valid_types

    def test_inpatient_gets_discharge_summary(self):
        encounters, patients, labs = self._setup()
        note_gen = NoteGenerator (seed=10)
        notes = note_gen.generate(encounters, patients, labs)
        note_by_enc = {n.encounter_id: n for n in notes}

        for enc in encounters:
            if enc.encounter_type == "inpatient":
                assert note_by_enc[enc.encounter_id].note_type == "discharge_summary"

    def test_notes_have_non_empty_text(self):
        encounters, patients, labs = self._setup()
        note_gen = NoteGenerator(seed=10)
        notes = note_gen.generate(encounters, patients, labs)

        for note in notes:
            assert len (note.text.strip()) > 50, "Note text too short"

    def test_notes_contain_synthetic_disclaimer(self):
        encounters, patients, labs = self._setup()
        note_gen = NoteGenerator(seed=10)
        notes = note_gen.generate(encounters, patients, labs)
        for note in notes:
            assert "synthetic" in note.text.lower(), "Missing synthetic disclaimer"
