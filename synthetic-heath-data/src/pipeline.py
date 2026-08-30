"""
pipeline.py - Orchestration layer for full synthetic patient record generation.

SynthHealthPipline wires together all four generators (patient, encounter
lab, notes) and exposes helpers to export results as pandas Dataframs or CSV files.
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from synth_health.generators.encounter import EncounterGenerator
from synth_health.generators.lab_values import LabGenerator
from synth_health.generators.notes import NoteGenerator
from synth_health.generators.patient import PatientGenerator
from synth_health.schema import (
    ClinicalNote,
    Encounter,
    LabResult
    Patient
)

class SynthHealthPipeline:
    """
    End-to-end synthetic healthcare data generation pipeline.

    Parameters

    n_patients:
        Number of synthetic patients to generate.
    seed:
        Master RNG seed; propagated to all sub-generators for reproducibility.
    start_date: 
        Earliest encounter date. Defaults to 3 years before today.
    end_date:
        Latest encounter date. Defaults to today.
    encounter_lab_fraction:
        Fraction of encounters that receive lab results (0.0 - 1.0].
    male_fraction:
        Fraction of patients assigned sex="M" (0.0 - 1.0].
    age_min, age_max:
        Inclusive patient age range in years.
    """

    def __init__(
        self,
        n_patients: int = 100,
        seed: int=42,
        start_date: date | None=None,
        end_date: date | None=None,
        encounter_lab_fraction: float = 0.60,
        male_fraction: float = 0.50,
        age_min: int = 18,
        age_max: int = 90,
    ) -> None:
        self.n_patients=n_patients
        self.seed = seed
        self.start_date = start_date or date(date.today().year-3, 1, 1)
        self.end_date = end_date or date.today()

        # Seed global RNGs
        random.seed(seed)
        np.random.seed(seed)

        self._patient_gen = PatientGenerator(
            seed=seed,
            male_fraction=male_fraction,
            age_max=age_max,
            age_min=age_min,
        )

        self._encounter_gen = EncounterGenerator(seed=seed)

        self._lab_gen = LabGenerator(
            seed=seed,
            encounter_lab_fraction=encounter_lab_fraction,
        )

        self._note_gen = NoteGenerator(seed=seed)

        self._results: dict | None = None

    # ---------------------------
    # Core Pipeline
    # ---------------------------

    def run(self) -> dict:
        """
        Execute the full generation pipeline.

        Returns

        dict with keys: "patients", "encounters", "labs", "notes"
        Each value is a list of the corresponding dataclass instances.
        """

        patients: list[Patient] = self._patient_gen.generate(self.n_patients)

        encounters: list[Encounter] = self._encounter_gen.generate(
            patients, self.start_date, self.end_date
        )

        labs: list[LabResult] = self._lab_gen.generate(encounters)

        # Build lookup structures for note generator
        patient_index: dict[str, Patient] = {p.patient_id: p for p in patients}
        labs_by_encounter: dict[str, list[LabResult]] = {}
        for lab in labs:
            labs_by_encounter.setdefault(lab.encounter_id, []).append(lab)

        notes: list[ClinicalNote] = self._note_gen.generate(
            encounters, patient_index, labs_by_encounter
        )

        self._results = {
            "patients": patients,
            "encounters": encounters,
            "labs": labs,
            "notes": notes,
        }

        return self._results

    # -------------------------
    # Export Helpers
    # -------------------------

    def to_dataframes(self) -> dict(str, pd.DataFrame):
        """
        Convert pipeline results to pandas DataFrames.

        Returns
        
        dict[str, pd.DataFrame] with keys: "patients", "encounters", "labs", "notes"

        Raises

        RuntimeError if `run()` has not been called first.
        """

        if self._results is None:

            raise RuntimeError("Call run() before to _dataframes()")
        
        return {
            "patients": self._patients_df(self._results["patients"]),
            "encounters": self._encounters_df(self._results["encounters"]),
            "labs": self._labs_df(self._results["labs"]),
            "notes": self._notes_df(self._results["notes"])
        }

    def to_csv(self, output_dir: str | Path) -> None:
        """
        Write one CSV file per entity ot *output_dir*.

        Parameters

        output_dir: 
            Destination directory. Created if it does not exist.

        Raises

        RuntimeError if `run()` has not been called first.
  
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dfs = self.to_dataframes()
        for name, df in dfs.items():
            dest = output_dir / f"{name}.csv"
            df.to_csv(dest, index=False)


    # --------------------------------------
    # DataFrame Builders
    # --------------------------------------

    @staticmethod
    def _patients_df(patients: list[Patient]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "patient_id": p.patient_id,
                    "first_name": p.first_name,
                    "last_name": p.last_name,
                    "dob": p.dob,
                    "sex": p.sex,
                    "race": p.race,
                    "zip_code": p.zip_code,
                    "age": p.age,
                }
                for p in patients
            ]
        )

    @staticmethod
    def _encounters_df(encounters: list[Encounter]) -> pd.DataFrame:

        rows = []

        for enc in encounters:
            rows.append(
                {
                    "encounter_id": enc.encounter_id,
                    "patient_id": enc.patient_id,
                    "date": enc.date,
                    "encounter_type": enc.encounter_type,
                    #Flatten diagnoses to pipe-separated string for tabular format
                    "diagnosis_codes": "|".join(dx.code for dx in enc.diagnoses),
                    "diagnosis_descriptions": "|".join(dx.description for dx in enc.diagnoses),   
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _labs_df(labs: list[LabResult]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "result_id": lab.result_id,
                    "patient_id": lab.patient_id,
                    "encounter_id": lab.encounter_id,
                    "test_name": lab.test_name,
                    "value": lab.value,
                    "unit": lab.unit,
                    "reference_low": lab.reference_low,
                    "reference_high": lab.reference_high,
                    "collection_date": lab.collection_date,
                    "is_abnormal": lab.is_abnormal,
                }
                for lab in labs
            ]
        )

    @staticmethod
    def _notes_Df(notes: list[ClinicalNote]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "note_id": note.note_id,
                    "patient_id": note.patient_id,
                    "encounter_id": note.encounter_id,
                    "note_type": note.note_type,
                    "created_at": note.created_at,
                    "text": note.text,
                }
                for note in notes
            ]
        )
    