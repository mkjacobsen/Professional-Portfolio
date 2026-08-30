"""
test_pipeline.py - Integration tests for SynthHealthPipeline.

Verifies output shapes, referential integrity, and DataFrame exports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import SynthHealthPipeline

@pytest.fixture(scope="module")
def pipeline_output():
    """Run a small pipeline once and share results across tests in this module."""
    pipeline = SynthHealthPipeline(n_patients=10, seed=42)
    return pipeline.run()

@pytest.fixture(scope="module")
def pipeline_dfs():
    pipeline = SynthHealthPipeline(n_patients=10, seed=42)
    pipeline.run()
    return pipeline.to_dataframes()

# ---------------------------------
# run() output shapes
# ---------------------------------

class TestPipelineRun:
    def test_returns_all_keys(self, pipeline_output):
        assert set(pipeline_output.keys()) == {"patients", "encounters", "labs", "notes"}

    def test_patient_count(self, pipeline_output):
        assert len(pipeline_output ["patients"]) == 10

    def test_encounters_exist(self, pipeline_output):
        assert len(pipeline_output ["encounters"]) >= 10 # at least 1 per patient

    def test_labs_exist(self, pipeline_output):
        # With 60% coverage there should besome labs
        assert len(pipeline_output["labs"]) > 0

    def test_notes_count_equals_encounters(self, pipeline_output):
        assert len(pipeline_output ["notes"]) == len(pipeline_output["encounters"])

# ---------------------------------
# Referentialintegrity
# ---------------------------------

class TestReferentialIntegrity:
    def test_encounter_patient_ids_all_valid(self, pipeline_output):
        patient_ids = [p.patient_id for p in pipeline_output ["patients"]]
        for enc in pipeline_output["encounters"]:
            assert enc.patient_id in patient_ids, (
                f"Encounter {enc.encounter_id} has unknown patient_id {enc.patient_id}"
            )

    def test_lab_patient_ids_all_valid(self, pipeline_output):
        patient_ids = (p.patient_id for p in pipeline_output ["patients"])
        for lab in pipeline_output["labs"]:
            assert lab.patient_id in patient_ids, (
                f"Lab {lab. result_id} has unknown patient_id {lab.patient_id}"
            )

    def test_lab_encounter_ids_all_valid(self, pipeline_output):
        encounter_ids = (e.encounter_id for e in pipeline_output["encounters"])
        for lab in pipeline_output["labs"]:
            assert lab.encounter_id in encounter_ids,(
                f"Lab {lab. result_id} has unknown encounter_id {lab.encounter_id}"
            )

    def test_note_patient_ids_all_valid(self, pipeline_output):
        patient_ids = (p.patient_id for p in pipeline_output["patients"])
        for note in pipeline_output["notes"]:
            assert note.patient_id in patient_ids

    def test_note_encounter_ids_all_valid(self,pipeline_output):
        encounter_ids = (e.encounter_id for e in pipeline_output ["encounters"])
        for note in pipeline_output["notes"]:
            assert note.encounter_id in encounter_ids

# -----------------------------------
# DataFrame shape and column checks
# -----------------------------------

class TestToDataFrames:
    def test_returns_four_dataframes(self, pipeline_dfs):
        assert set(pipeline_dfs.keys()) == {"patients", "encounters", "labs", "notes"}

    def test_patients_df_columns(self, pipeline_dfs):
        expected = {
            "patient_id", "first_name", "last_name", "dob",
            "sex", "race", "zip_code", "age",
        }

        assert expected.issubset(set(pipeline_dfs["patients"].columns))

    def test_encounters_df_columns(self, pipeline_dfs):
        expected = {
            "encounter_id", "patient_id", "date",
            "encounter_type","diagnosis_codes", "diagnosis_descriptions",
        }

        assert expected.issubset(set(pipeline_dfs["encounters"].columns))

    def test_labs_df_columns(self, pipeline_dfs):
        expected = {
            "result_id", "patient_id", "encounter_id",
            "test_name", "value","unit",
            "reference_low", "reference_high",
            "collection_date", "is_abnormal",
        }

        assert expected.issubset(set(pipeline_dfs["labs"].columns))

    def test_notes_df_columns(self, pipeline_dfs):
        expected = {
            "note_id", "patient_id",
            "text", "encounter_id", "note_type","created_at",
        }

        assert expected.issubset(set(pipeline_dfs["notes"].columns))

    def test_row_counts_consistent(self, pipeline_dfs, pipeline_output):
        assert len(pipeline_dfs["patients"]) == 10
        assert len(pipeline_dfs["encounters"]) == len(pipeline_output["encounters"])
        assert len(pipeline_dfs["notes"]) == len(pipeline_output["notes"])

# ---------------------
# CSV export
# ---------------------

class TestToCSV:
    def test_writes_four_csv_files(self, tmp_path, pipeline_output):
        pipeline = SynthHealthPipeline(n_patients=10, seed=42)
        pipeline.run()
        pipeline.to_csv(tmp_path)

        for name in ("patients", "encounters", "labs","notes"):
            csv_file = tmp_path / f"{name}.csv"
            assert csv_file.exists(), f"{name}.csv not found"

            df = pd.read_csv(csv_file)
            assert len(df) > 0, f"{name}.csv is empty"

    def test_csv_creates_directory_if_missing(self, tmp_path):
        new_dir = tmp_path / "sub" / "output"
        pipeline = SynthHealthPipeline(n_patients=5, seed=1)
        pipeline.run ()
        pipeline.to_csv(new_dir)
        assert new_dir.is_dir()

# -----------------------------
# Reproducibility
# -----------------------------

class TestReproducibility:

    def test_same_seed_same_first_patient(self):
        r1 = SynthHealthPipeline(n_patients=5, seed=7).run()
        r2 = SynthHealthPipeline(n_patients=5, seed=7).run()
        p1 = r1["patients"][0]
        p2 = r2["patients"][0]

        assert p1.first_name == p2.first_name
        assert p1.last_name == p2.last_name
        assert p1.dob == p2.dob

    def test_to_dataframes_raises_before_run(self):
        pipeline = SynthHealthPipeline(n_patients=5, seed=0)
        with pytest.raises(RuntimeError, match="run()"):
            pipeline.to_dataframes ()
