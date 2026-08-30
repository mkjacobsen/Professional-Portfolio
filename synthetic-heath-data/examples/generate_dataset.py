"""
generate_dataset.py - Example script: generate a 500-patient synthetic datset.

Usage: 
    pip install -e ../ (from the synthetic-health-data/root)
    python examples/generate_dataset.py
    
Outputs four CSV files to ./output/ an prints a summary to stdout.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# Allow running from the examples/ directory without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synth_health.pipeline import SynthHealthPipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

def main() -> None:
    print("=" * 60)
    print("Synthetic Health Data Generator")
    print("All data is fully synthetic. No real patient information.")
    print("=" * 60)
    print()

    print("Initializing pipeline (n_patients=500, seed=42)...")
    pipeline = SynthHealthPipeline(n_patients=500, seed=42)

    print("Running generation pipeline...")
    results = pipeline.run()

    patients = results["patients"]
    encounters = results["encounters"]
    labs = results["labs"]
    notes = results["notes"]

    print()
    print("-" * 40)
    print("GENERATION SUMMARY")
    print("-" * 40)
    print(f"  Patients      : {len(patients):>6,}")
    print(f"  Encounters    : {len(encounters):>6,}")
    print(f"  Lab Results   : {len(labs):>6,}")
    print(f"  Notes         : {len(notes):>6,}")
    print()

    dfs = pipeline.to_dataframes()

    print("-" * 40)
    print("SCHEMA (column names per entity)")
    print("-" * 40)
    for name, df in dfs.items():
        cols = ", ".join(df.columns.tolist())
        wrapped = textwrap.fill(cols, width=56, subsequent_indent="               ")
        print(f". (name:<12): {wrapped}")

    print()

    print("-" * 40)
    print("SAMPLE ROWS - patients (first 3)")
    print("-" * 40)
    print(
        dfs["patients"][["patient_id", "first_name", "last_name", "dob", "sex", "race"]]
        .head(3)
        .to_string(index=False)
    )
    print()

    print("-" * 40)
    print("SAMPLE ROWS - labs (first 5)")
    print("-" * 40)
    print(
        dfs["labs"][["test_name","value","unit","is_abnormal"]]
        .head(5)
        .to_string(index=False)
    )
    print()

    print("-" * 40)
    print("SAMPLE NOTE TEXT (first 500 chars)")
    print("-" * 40)
    first_note = dfs["notes"]["text"].iloc[0]
    print(first_note[:500])
    print("...")
    print()

    print(f"Writeing CSVs to: {OUTPUT_DIR}")
    pipeline.to_csv(OUTPUT_DIR)
    for name in ("patients", "encounters", "labs", "notes"):
        path = OUTPUT_DIR / f"{name}.csv"
        print(f". Wrote {path.name} ({path.stat().st_size:,} bytes)") 

    print()
    print("Done. All output is fully synthetic and contains no real patient data.")   

if __name__ == "__main__":
    main()