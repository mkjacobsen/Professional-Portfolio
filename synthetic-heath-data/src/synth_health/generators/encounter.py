"""
generators/encounter.py - Synthetic encounter and diagnosis generator.

Each patient receives 1-8 encounters distributed randomly across a 
configurable date range.  Encounter types follow an approximate real-world 
distribution. Diagnoses are samples from a curated set of common ICD-10 codes.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from synth_health.schema import Encounter, ICD10Code, Patient

# ----------------------------------------------------------------------------
# Curated ICD-10-CM code catalogue (~30 common ambulatory/inpatient diagnoses)
# ----------------------------------------------------------------------------

COMMON_ICD10_CODES: list[ICD10Code] = [
    # Cardiovascular
    ICD10Code("I10", "Essential (primary) hypertension"),
    ICD10Code("I25.10", "Atherosclerotic heart disease of native coronary artery without angina"),
    ICD10Code("I48.91", "Unspecified atrial fibrillation"),
    ICD10Code("I50.9", "Heart failure, unspecified"),
    ICD10Code("I63.9", "Cerebral infarction, unspecified"),
    # Endocrine/Metabolic
    ICD10Code("E11.9", "Type 2 diabetes mellitus without complications"),
    ICD10Code("E78.5", "Hyperlipidemia, unspecified"),
    ICD10Code("E66.09", "Other obesity due to excess calories"),
    ICD10Code("E03.9", "Hypothyroidism, unspecified"),
    ICD10Code("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
    # Mental Health
    ICD10Code("F41.1", "Generalized anxiety disorder"),
    ICD10Code("F32.9", "Major depressive disorder, single episode, unspecified"),
    ICD10Code("F10.10", "Alcohol use disorder, mild"),
    ICD10Code("F41.9", "Anxiety disorder, unspecified"),
    ICD10Code("F33.0", "Major depressive disorder, recurrent, mild"),
    # Gastrointenstinal
    ICD10Code("K21.0", "Gastro-esophageal reflux disease with esophagitis"),
    ICD10Code("K57.30", "Diverticulosis of large intestine without performation or abscess"),
    ICD10Code("K92.1", "Melena"),
    ICD10Code("K76.0", "Fatty (change of) liver, not elsewhere classified"),
    # Respiratory
    ICD10Code("J45.909", "Unspecified asthma, uncomplicated"),
    ICD10Code("J44.1", "Chronic obstrutive pulmonary disease with acute exacerbation"),
    ICD10Code("J06.9", "Acute upper respiratory infection, unspecified"),
    ICD10Code("J18.9", "Pneumonia, unspecified organism"),
    # Musculoskeletal
    ICD10Code("M54.5", "Low back pain"),
    ICD10Code("M17.11", "Primary osteoarthritis, right knee"),
    ICD10Code("M79.3", "Panniculitis, unspecified"),
    # Genitourinary
    ICD10Code("N39.0", "Urinary tract infection, site not specified"),
    ICD10Code("N18.3", "Chronic kidney disease, stage 3 (moderate)"),
    # Preventive / Screening
    ICD10Code("Z00.00", "Encounter for general adult medial examination without abnormal findings."),
    ICD10Code("Z12.11", "Encounter for screening for malignant neoplasm of colon"),
]

# Encounter type distribution weights (outpatient 70%, inpatient 20%, emergency 10%)

_ENCOUNTER_TYPE_CHOICES = ["outpatient", "inpatient", "emergency"]
_ENCOUNTER_TYPE_WEIGHTS = [0.70, 0.20, 0.10]

class EncounterGenerator: 
    """
    Generates synthetic Encounter records for a list of patients.

    Parameters

    seed: 
        RNG seed.

    encounters_per_patient_range:
        (min, max) inclusive number of encounters each patient receives.
    
    max_diagnoses_per_encounter:
        Upper bound of disnoses per encounter (min is always 1). 
    """

    def __init__(
        self,
        seed: int = 42,
        encounters_per_patient_range: tuple[int, int] = (1, 8),
        max_diagnoses_per_encounter: int = 4,
    ) -> None:
        self._rng = random.Random(seed)
        self._enc_min, self._enc_max = encounters_per_patient_range
        self._max_dx = max_diagnoses_per_encounter

        if self._enc_min < 1 or self._enc_max < self._enc_min: 
            raise ValueError("Invalid encounters_per_patient_range")
        if self._max_dx < 1:
            raise ValueError("max_diagnoses_per_encounter must be >= 1.")

    # ------------------------------
    # Internal Helpers
    # ------------------------------

    def _random_date(self, start: date, end: date) -> date:
        span = (end - start).days
        if span <= 0:
            return start
        return start + timedelta(days=self._rng.randint(0,span))

    def _sample_diagnoses(self, n: int) -> list[ICD10Code]:
        """Sample *n* distince diagnoses from the catalogue."""
        return self._rng.sample(COMMON_ICD10_CODES, min(n, len(COMMON_ICD10_CODES)))

    # ------------------------------
    # Public API
    # ------------------------------

    def generate(
        self,
        patients: list[Patient],
        start_date: date,
        end_date: date,
    ) -> list[Encounter]:
        """
        Generate encounters for each patient in *patients*.

        Parameters

        patients:
            List of patient records to generate encounters for.
        start_date:
            Earliest possible encounter date (inclusive).
        end_date:
            Latest possible encounter date (inclusive).

        Returns

        list[Encounter]
            All encounters across all patients, unsorted.
        """

        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        encounters: list[Encounter] = []
        for patient in patients:
            n_enc = self._rng.randint(self._enc_min, self._enc_max)

            for _ in range(n_enc):
                enc_type = self._rng.choices(
                    _ENCOUNTER_TYPE_CHOICES,
                    weights=_ENCOUNTER_TYPE_WEIGHTS,
                    k=1
                )[0]

                n_dx = self._rng.randint(1, self._max_dx)
                diagnoses = self._sample_diagnoses(n_dx)
                enc = Encounter(
                    patient_id = patient.patient_id,
                    date=self._random_date(start_date,end_date),
                    encounter_type=enc_type,
                    diagnoses=diagnoses,
                )
                encounters.append(enc)

        return encounters