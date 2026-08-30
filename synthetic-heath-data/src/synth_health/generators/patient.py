"""
generators/patient.py - Synthetic patient demographic generator.

Uses Faker for names and zip codes.  Sex, age, and race distributions are
configurable but default to approximate US population parameters.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from faker import Faker

from synth_health.schema import Patient

# Approximate US Census race/enthnicity distribution (2020 ACS)
_DEFAULT_RACE_DISTRIBUTION: dict[str, float] = {
    "White": 0.596,
    "Hispanic or Latino": 0.185,
    "Black or African American": 0.134,
    "Asian": 0.059,
    "American Indian or Alaska Native": 0.013,
    "Native Hawaiian or Pacific Islander": 0.003,
    "Other": 0.010,
}

class PatientGenerator:
    """
    Generates synthetic Patient records with realistic demographic distributions.

    Parameters

    seed:
        RNG seed for reproducibility.
    male_fraction:
        Fraction of patients assigned sex='M'. Remaining are 'F'. Default 0.5.
    age_min, age_max:
        Inclusive age range in years.  Default = 18-90.
    race_distribution:
        Mapping of race label -> probability weight.  Values need not sum to 1;
        they are normalized internally.  Defaults to approximate US Census 2020.
    """

    def __init__(
        self,
        seed: int = 42,
        male_fraction: float = 0.5,
        age_min: int = 18,
        age_max: int = 90,
        race_distribution: dict[str, float] | None = None,
    ) -> None:
        self._rng = random.Random(seed)

        # Use a per-instance Generator so multiple PatientGenerators
        # with the same seed are independent of each other.

        self._faker = Faker()
        self._faker.seed_instance(seed)

        if not 0.0 <= male_fraction <= 1.0:
            raise ValueError("male fraction must be between 0 and 1")
        if age_min < 0 or age_max <= age_min:
            raise ValueError("Invalid age range: age_min must be < age_max and >= 0")

        self._male_fraction = male_fraction
        self._age_min = age_min
        self._age_max = age_max

        dist = race_distribution or _DEFAULT_RACE_DISTRIBUTION
        total = sum(dist.values())
        self._race_labels = list(dist.keys())
        self._race_weights = [v / total for v in dist.values()]

    # ------------------------------
    # Internal Helpers
    # ------------------------------

    def _random_dob(self) -> date:
        """
        Return a random date of birth consistent with the configured age range.
        """

        today = date.today()
        age_days_min = self._age_min * 365
        age_days_max = self._age_max * 365+364 #include the full oldest year
        delta_days = self._rng.randint(age_days_min, age_days_max)
        return today - timedelta(days=delta_days)

    def _random_sex(self) -> str:
        return "M" if self._rng.random() < self._male_fraction else "F"

    def _random_race(self) -> str:
        return self._rng.choices(self._race_labels, weights=self._race_weights, k=1)[0]

    def _random_name(self, sex:str) -> tuple[str, str]:
        if sex == "M":
            first = self._faker.first_name_male()
        else:
            first = self._faker.first_name_female()
        last = self._faker.last_name()

        return first, last

    # ---------------------------
    # Public API
    # ---------------------------

    def generate(self, n: int) -> list[Patient]:
        """
        Generate *n* synthetic Patient records.

        Parameters

        n: 
            Number of patients to generate.  Must be positive.
        
        Returns

        list[Patient]
        """

        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")

        patients: list[Patient] = []

        for _ in range(n):
            sex = self._random_sex()
            first, last = self._random_name(sex)
            patients.append(
                Patient(
                    first_name=first,
                    last_name=last,
                    dob=self._random_dob(),
                    sex=sex,
                    rae=self._random_race(),
                    zip_code=self._faker.zipcode(),
                )
            )

        return patients