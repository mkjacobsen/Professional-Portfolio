"""
generators/lab_values.py - Synthetic laboratory result generator.

Uses numpy distributions (normal/lognormal) to produce realistic-looking lab values.
A configurable fraction of encounters receive labs.  Diabetic patients have elevated
out-of-range probability for glucose and HbA1c.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from synth_health.schema import Encounter, LabResult

# ICD-10 prefix codes that indicate a diabetic patient
_DIABETES_CODES = ("E11", "E10", "E13")

def _has_diabetes(encounter: Encounter) -> bool:
    return any(
        dx_code_startswith(tuple(_DIABETES_CODES)) for dx in encounter.diagnoses
    )

# -------------------------------------------------
# Lab Test Specifications
# -------------------------------------------------

class LabSpec(NamedTuple):
    """Specification for a single lab test's value distribution."""

    test_name: str
    unit: str
    reference_low: float
    reference_high: float

    # Parameters for a normal distribution representing "typical healthy" values
    mean: float
    std: float

    # If True, draw from lognormal instead; mean/std are in log-space
    lognormal: bool = False

    # Probability that a non-diabetic patient's value is out of reference range
    base_abnormal_prob: float = 0.10

_LAB_SPECS: list[LabSpec] = [
    LabSpec(
        test_name="Glucose",
        unit="mg/dL",
        reference_low=70.0,
        reference_high=100.0,
        mean=85.0,
        std=10.0,
        base_abnormal_prob=0.15,
    ),
    LabSpec(
        test_name="HbA1c",
        unit="%",
        reference_low=4.0,
        reference_high=5.6,
        mean=5.1,
        std=0.4,
        base_abnormal_prob=0.10,
    ),
    LabSpec(
        test_name="HbA1c",
        unit="%",
        reference_low=4.0,
        reference_high=5.6,
        mean=5.1,
        std=0.4,
        base_abnormal_prob=0.10,
    ),
    LabSpec(
        test_name="Total Cholesterol",
        unit="mg/dL",
        reference_low=125.0,
        reference_high=200.0,
        mean=185.0,
        std=25.0,
        base_abnormal_prob=0.12,
    ),
    LabSpec(
        test_name="LDL",
        unit="mg/dL",
        reference_low=0.0,
        reference_high=100.0,
        mean=90.0,
        std=20.0,
        base_abnormal_prob=0.15,
    ),
    LabSpec(
        test_name="HDL",
        unit="mg/dL",
        reference_low=40.0,
        reference_high=60.0,
        mean=52.0,
        std=10.0,
        base_abnormal_prob=0.10,
    ),
    LabSpec(
        test_name="Systolic BP",
        unit="mmHg",
        reference_low=90.0,
        reference_high=120.0,
        mean=115.0,
        std=12.0,
        base_abnormal_prob=0.18,
    ),
    LabSpec(
        test_name="Creatinine",
        unit="mg/dL",
        reference_low=0.6,
        reference_high=1.2,
        mean=0.9,
        std=0.15,
        base_abnormal_prob=0.08,
    ),
    LabSpec(
        test_name="TSH",
        unit="mIU/L",
        reference_low=0.4,
        reference_high=4.0,
        mean=1.8,
        std=0.8,
        lognormal=True,
        base_abnormal_prob=0.08,
    )
]

class LabGenerator:
    """
    Generates synthetic LabResult records for a subset of encounters.

    Parameters

    seed:
        RNG seed.
    encounter_lab_fraction:
        Fraction of encounters that receive lab results. Default 0.60.
    diabetic_abnormal_multiplier:
        Multiplier applied to abnormal probability for glucose/HbA1c
        when a patient has a diabetes diagnosis.  Default 3.0.
    """

    def __init__(
        self,
        seed: int = 42,
        encounter_lab_fraction: float = 0.60,
        diabetic_abnormal_multiplier: float = 3.0,
    ) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        if not 0.0 < encounter_lab_fraction <= 1.0:
            raise ValueError("encounter_lab_fraction must be in (0,1]")
        if diabetic_abnormal_multiplier < 1.0:
            raise ValueError("diabetic_abnormal_multiplier must be >= 1.0")

        self._lab_fraction = encounter_lab_fraction
        self._diabetic_mult = diabetic_abnormal_multiplier

# ------------------------
# Internal Helpers
# ------------------------

    def _sample_value(self, spec: LabSpec, diabetic: bool) -> float:
        """
        Draw a single lab value from the spec's distribution.

        With probability `abnormal_prob`, a value is drawn from the 
        tails of the distribution (outside reference range) to simulate pathology.
        """

        abnormal_prob = spec.base_abnormal_prob
        if diabetic and spec.test_name in ("Glucose", "HbA1c"):
            abnormal_prob = min(0.90, abnormal_prob * self._diabetic_mult)
        
        if self._rng.random() < abnormal_prob:
            # Draw abnormal: nudge mean toward an out-of-range value
            if spec.lognormal:
                # Shift log-mean upward for high side
                val = float(self._np_rng.lognormal(
                    mean = np.log(spec.mean) + 0.5, sigma = spec.std
                ))

            else:
                # Bias toward the high tail
                shift = (spec.reference_high - spec.reference_low) * 0.5
                val = float(self._np_rng.normal(
                    loc = spec.reference_high + shift, scale = spec.std
                ))

            # Clamp to a physiologically plausible absolute floor

            val = max(val, max(0.0, spec.reference_low + 0.3))

        else:
            # Draw a normal (in-range) value
            if spec.lognormal:
                val = float(self._np_rng.lognormal(
                    mean=np.log(spec.mean), sigma=spec.std
                ))
            
            else:
                val = float(self._np_rng.normal(loc=spec.mean, scale=spec.std))

            # Soft-clamp in-range draws to keep them sensible
            val = float(np.clip(val, spec.reference_low * 0.8, spec.reference_high * 1.2))

        return round(val, 2)

    def _collection_date(self, enc_date: date) -> date:
        """Labs are typically collected on the encounter date or up to 2 days prior."""

        offset = self._rng.randint(-2, 0)
        return enc_date + timedelta(days=offset)

    # --------------------
    # Public API
    # --------------------

    def generate(self, encounters: list[Encounter]) -> list[LabResult]:
        """
        Generate lab resutls for a random subset of *encounters*.
        
        Each selected encounter receives resutls for all lab tests in the catalogue.

        Parameters

        encounters:
            Encounter records to potentially annotate with labs.
        
        Returns

        list[LabResult]
        """

        results: list[LabResult] = []

        for enc in encounters:
            if self._rng.random() > self._lab_fraction:
                continue

            diabetic = _has_diabetes(enc)
            collect_date = self._collection_date(enc.date)

            for spec in _LAB_SPECS:
                value = self._sample_value(spec, diabetic)
                results.append(
                    LabResult(
                        patient_id=enc.patient_id,
                        encounter_id=enc.encounter_id,
                        test_name=spec.test_name,
                        value=value,
                        unit=spec.unit,
                        reference_low=spec.reference_low,
                        reference_high=spec.reference_high,
                        collection_date=collect_date,
                    )
                )

        return results

    
