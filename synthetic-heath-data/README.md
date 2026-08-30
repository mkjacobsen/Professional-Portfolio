# Synthetic Health Data Generator

A configurable Python pipeline that generates realistic-but-entirely-fictional patient records - demographics, clinical encounters, laboratory results, and free-text clinical notes. The project is designed for use in MI development, data engineering testing, and pipeline prototyping where realistic healthcare data shapes are needed but no actual patient data may be used.

Synthetic healthcare data matters because real clinical datasets carry significant legal constraints (HIPAA and equivalents), access barriers, and residual re-identification risk even after de-identification. Having a generator that produces structurally authentic records - correct field types, realistic value distributions, referential integrity across entities - lets engineers build and test data pipelines, schema validators, ML Feature extractors, and NLP preprocessing code without touching any protected information.

This project contains **no real patient information of any kind**. All names, dates, diagnoses, lab values, and notes arecomputationally generated from configurable probability distributions and fixed templates.

## Table of contents

1. [Architecture] (#architecture)
2. [Quickstart] (#quickstart)
3. [Pipeline API reference](pipeline-api-reference)
4. [Generator API reference](#generator-api-reference)
5. [Schema reference](#schema-reference)
6. [Lab tests reference](#lab-tests-reference)
7. [ICD-10 codes reference](#icd-10-codes-reference)
8. [Reproducibility](reproducibility)
9. [Extending the generators](#extending-the-generators)
10. [Running tests](running-tests)
11. [Known limitations](#known-limitations)
12. [Data disclaimer](#data-disclaimer)

# Architecture
The project is organized in three layers: schema, generators, and pipeline. Each layer depends only on the layers below it, so generators are independently usable or composable through the pipeline.

src/synth_health/
|-- schema.py Dataclass definitions (Patient, Encounter, LabResult,ClinicalNote, ICD10Code) with post_init validation.
|-- generators/
|   |-- patient.py PatientGenerator - uses Faker for names/zips; configurable sex, age range, and US Census-weighted race distribution.
|   |-— encounter.py EncounterGenerator - assigns 1-8 encounters per patient sampled from a curated catalogue of 30 common ICD-10 codes.
|   |-- lab_values.py LabGenerator - draws values from numpy normal/Lognormal distributions; elevates out-of-range probability for diabetic patients on Glucose and HDAIC.
|   |-— notes.py NoteGenerator - template-based SOAP notes, discharge summaries, and referral letters; each template interpolates patient age, sex,diagnoses, and lab findings.
|-- pipeline.py SynthHealthPipeline - wires all generators together, manages RNG seeds, and exposes run () / to dataframes () / to_csv () for easy downstream use.

**schema py** defines five Python dataclasses that serve as the canonical data model. Each dataclass validates its inputs in_post_init and exposes any derived values as properties (e.g. `Patient.age`, `LabResult.is_abnormal`). All primary keys are UUID strings not auto-generated via `uuid.uuid4()` if not supplied by the caller.

**generators/** contains four independent classes, each responsible for one entity type. Every generator accepts an explicit `seed` parameter and constructs its own seeded `random.Random` instance (and a `numpy.random.Generator" where applicable) so that multiple generators can be instantiated independently without interfering with each other's sequences.

**pipeline.py** is the orchestration layer. `SynthHealthPipeline` instantiates all four generators, runs them in order (patients first, then encounters, then labs and notes which depend on the encounter list), and wires the output of each stage into the input of the next. It also provides convenience methods for exporting results as pandas DataFrames or writing CV files.

# Quickstart
```bash
# Install in editable mode from the project root
cd synthetic-health-data
pip install -e .
# Generate a 500-patient dataset (CVs written to examples/output/)
python examples/generate_dataset.py
```

Expected terminal output:

```
============================================================
Synthetic Health Data Generator
All data is fully synthetic. No real patient information.
============================================================

Initialising pipeline (n_patients-500, seed=42).
Running generation pipeline...

-------------------------------------------------------------
GENERATION SUMMARY
-------------------------------------------------------------

Patients 500
Encounters: 2,xxx
Lab Results: x,xxx
Notes : 2,xxx

-------------------------------------------------------------
SCHEMA (column names per entity)
-------------------------------------------------------------

patients    : patient_id, first_name, last_name, dob, sex, race, zip_code, age
encounters  : encounter_id, patient_id, date, encounter_type, diagnosis_codes, diagnosis_descriptions
labs        : result_id, patient_id, encounter_id, test name, value, unit, reference_low, reference_high, collection date, is_abnormal
notes       : note_id, patient id, encounter_id, note_type, created_at, text

***
Done. All output is fully synthetic and contains no real patient data.
***
```

The exact encounter/lab/note counts vary because each patient receives a random number of encounters (1-8 by default) and 608 of encounters receive labs.

** Using the pipeline in code
```python
from synth health import Synthealthpipeline
pipeline = SynthHealthPipeline(n_patients=200, seed=42)
results = pipeline.run()

# Work with dataclass objects
patients - results["patients"] # list [Patient]
encounters - results ["encounters"] # list [Encounter]
labs - results ["labs"] # list [LabResult]
notes - results ["notes"] # list[ClinicalNote]

# Or get pandas DataFrames
dfs - pipeline. to_dataf rames ()
print (dfs ["labs"]. head ())

# Or write CSV files
pipeline.to_csv ("./output")
```


## Pipeline API reference

### SynthHealthPipeline
```python
class SynthHealthPipeline:
    def __init__(
        self,
        n_patients: int = 100,
        seed: int = 42,
        start_date: date | None = None,
        end_date: date | None = None,
        encounter_lab_fraction: float = 0.60,
        male_fraction: float = 0.50,
        age_min: int = 18,
        age_max: int = 90,
    ) -> None: ...
```

#### '__init__' parameters
| Parameter | Type | Default | Description |
| --------- | ----- | ------- | ---------- |
| `n_patients` | `int` | `100` |  Number of synthetic patients to generate. |
| `seed` | `int` | `42` | Master RNG seed. Propagated to all sub-generators so the entire output is reproducible. |
| `start_date` | `date` \| None` | `None'`| Earliest possible encounter date. When "None defaults to January 1 of three calendar years before today. I\|
| `end_date` | `date \| None` | `None` | Latest possible encounter date. When "None", defaults to today. |
| `encounter_lab_fraction` | `float` | `0.60` | Fraction of encounters that receive lab results. Must be in the range `(0.0, 1.0]`.|
| `male_fraction` | `float` | '0.50` | Fraction of patients assigned sex='M'. The remaining fraction are assigned `sex='F'`. Must be in `[0.0, 1.0]`. |
| `age max` | `int` | 18 | Minimum patient age in years (inclusive). |
| `age_max` | `int` | 90 | Maximum patient age in years (inclusive). |

#### `run()`
```python
def run (self) -> dict:
```

Executes the full generation pipeline in order: patients, then encounters, then labs, then notes. Results are cached on the instance so subsequent calls to `to_dataframes()` or `to_csv()` do not re-run generation.

Returns a `dict` with four keys:

| Key | Value type | Description |
| --- | ----------- | ---------- |
| `"patients"` | `list[Patient]` | All generated patient records. |
| `"encounters"` | `list[Encounter]` | All encounters across all patients, unsorted. |
| `"labs"` | `list[LabResult]` | Lab results for the subset of encounters selected by encounter_lab_fraction`. |
| `notes` | `list[ClinicalNote]` | One clinical noteper encounter. |

#### `to_dataframes()`

```python
def to_dataframes(self) -> dict[str, pd.Dataframe]:
```

Converts the cached pipeline results to pandas DataFrames.  Raises `RuntimeError` if `run()` has not been called first.

Returns a `dict[str, pd.DataFrame] with keys `"patients"`, `"encounters"`, `"labs"`, and `"notes"`.  The diagnosis list in `encounters` is flattened to two pipe-separated sting columns (`diagnosis_codes` and `diagnosis_descriptions`).

#### `to_csv()`

```python
def to_csv(self, output_dir: str | Path) -> None:
```

Write one CSV file per entity to `output_dir`. The director is created (including any missing parents) if it does not exist.  File names are `patients.csv`, `encounters.csv`, `labs.csv`, and `notes.csv`.  Raises `RuntimeError` if `run()` has not been called first.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `output_dir` | `str \| Path` | Destination directory. Created with `mkdir(parents=True, exist_ok=True)` if absent. |


## Generator API Reference

Each generator can be used independently of the pipeline.

### `PatientGenerator`

```python
from synth_health.generators.patient import PatientGenerator

class PatientGenerator:

    def __init__(
        self,
        seed: int = 42,
        male_fraction: float = 0.5,
        age_min: int = 18,
        age_max: int = 90,
        race_distribution: dict[str, float] | None = None,
    ) -> None: ...

    def generate(self, n: int) -> list[Patient]: ...
```

Generates synthetic `Patient` demographic records.  Name are produced by Faker using sex-matches first names.  Zip codes are also produced by Faker.  Sex, age, and race are drawn from configurable distributions.


#### Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `seed` | `int` | `42` | RNG seed for both `random.Random` and Faker's per-instance seed. |
| `male_fraction` | `float` | `0.5` | Probability that a patient is assigned `sex='M'`. Must be in `[0.0, 1.0]`. |
| `age_min` | `int` | `18` | Minimum patient age in years. Must be `>= 0` and less than `age_max`.|
| `age_max` | `int` | `90` | Maximum patient age in years.  Must be greater than `age_min`.|
| `race_distribution` | `dict[str, float] \| None` | `None` | Mapping of race label to probability weight.  Weights are normalized internally and need not sum to 1.  When `None`, defaults to approximate US Census 2020 ACS propotions. |

The default race ditribtuion is:

| Race | Weight |
| ---- | ------ |
| White | 0.596 |
| Hispanic or Latino | 0.185 |
| Black or African American | 0.134 |
| Asian | 0.059 |
| American Indian or Alask Native | 0.013 |
| Native Hawaiian or Pacific Islander | 0.003 |
| Other | 0.010 |

** Useage example: **
```python
gen = PatientGenerator(seed=0, male_fraction=0.4, age_min=30, age_max=70)
patients = gen.generate(100) # list[Patient], len == 100
```

### `EncounterGenerator`

```python
from synth_health.generators.encounter import EncounterGenerator
class EncounterGenerator:
    def __init__:(
        self,
        seed: int = 42,
        encounters_per_patient_range: tuple [int, int] = (1, 8),
        max_diagnoses_per_encounter: int = 4,
    ) -> None: ...

    def generate (
        self,
        patients: list [Patient],
        start_date: date,
        end date: date,
    ) -> list[Encounter]:
```

Generates synthetic 'Encounter' records for a list of patients. Each patient receives a random number of encounters drawn uniformly from `encounters_per_patient_range`. Encounter types follow a fixed distribution (outpatient 70%, inpatient 20%, emergency 10%).  Diagnoses are samples without replacement from the 30-code `COMMON_ICD10_CODES` catalogue.

#### Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `seed` | `int` | `42` | RNG seed |
| `encounters_per_patient_range` | `tuple[int,int]` | `(1,8)` | Inclusive `(min,max)` number of encounters per patient. `min` must be `>=1`.|
| `max_diagnoses_per_encounter` | `int` | `4` | Upper bound on the number of diagnoses attached to a single encounter. Minimum is always 1. |

**Usage example:**

```python
gen = EncounterGenerator(seed=0, encounters_per_patient_range=(2,5))
encounters = gen.generate(patients, date(2022,1,1), date(2024,12,31))
```

### `LabGenerator`
```python
from synth_health.generators.lab_values import LabGenerator
class LabGenerator:
    def __init__:(
        self,
        seed: int = 42,
        encounter_lab_fraction: float = 0.60,
        diabetic_abnormal multiplier: float = 3.0,
    ) -> None:

    def generate (self, encounters: list [Encounter]) -> list [LabResult]: ...
```

Generates syntheticLabResult records for a random subset of encounters. Each selected encounter receives one result for every test in the built-in `_LAB_SPECS` catalogue (8 tests). Values are drawn from normal or lognormal distributions parameterised per test. A separate abnormal-value pathway is triggered with probability `base_abnormal_prob` per test, biasing the draw toward the high tail of the reference range.
For encounters where the patient has a diabetes diagnosis (ICD-10 prefix E10, E11, or E13), the abnormal probability for Glucose and HbA1c is multiplied by `diabetic_abnormal` multiplier (capped at 0.90). Collection dates are set to the encounter date or up to two days prior.

#### Parameters
| Parameter | Type | Default | Description |
| `seed` | `int` | `42` | RNG seed for both 'random.Random" and numpy.random.default_rng. |
| `encounter_lab_fraction` | `float` | `0.60` | Fraction of encounters that receive lab results. Must be in `(0.0, 1.0]`. |
| `diabetic_abnormal multiplier` | `float` | `3.0` | Multiplier applied to the base abnormal probability for Glucose and HbA1c when a diabetes diagnosis is present. Must be `>=1.0`. |

**Usage example:**

```python
    gen = LabGenerator(seed=0, encounter_lab_fraction=0.75)
    labs = gen.generate(encounters) #list[LabResult]
```

### `NoteGenerator'
```python
    from synth_health.generators.notes import NoteGenerator
    
    class NoteGenerator:
        def __init__ (self, seed: int = 42) -> None: ...

        def generate:(
            self,
            encounters: list [Encounter],
            patients: dict(str, Patient),
            Labs: dict[str, list [LabResult]],
        ) -> list[ClinicalNote]: ...
```

Generates one ClinicalNote per encounter using fixed string templates. The note type is selected deterministically by encounter type: inpatient encounters always produce a discharge summary; outpatient and emergency encounters randomly produce a SOAP note or referral letter. Patient name, age, sex, diagnoses, and any available lab findings are interpolated into the template at render time. Note creation timestamps are set to a random hour on the encounter date (7:00-18:59).

#### Parameters

| Parameter | Type | Default | Description |
| `seed` | `int` | `42` | RNG seed used when selecting content from the template content banks (chief complaints, plan texts, etc.). |

**Usage example:**
```python
    patient_index = (p.patient_id: p for p in patients)
    labs_by encounter = ()
    for lab in labs:
        labs_by_encounter.setdefault(lab.encounter_id, []).append(lab)

    gen = NoteGenerator(seed=0)
    notes = gen.generate(encounters, patient_index, labs_by_encounter)
```

## Schema reference
All classes are defined in 'src/synth_health/schema.py' as standard Python `dataclasses`. Primary key fields default to a freshly generated `uuid.uuid4()` string if not suppied.

### `ICD10Code`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `code` | `str` | ICD-10-CM diagnosis code (e.g., "I10"). Must not be empty. |
| `description` | `str` | Human-readable description (e.g., "Essential (primary) hypertension"). Must not be empty. |

### `Patient` 

| Field | Type | Description |
| ----- | ---- | ----------- |
| `patient_id` | `str` | UUID primary key. Auto-generated if omitted. |
| `first_name` | `str` | Given name, produced by Faker. |
| `last_name` | `str` | Family name, produced by Faker. |
| `dob` | `date` | Date of birth. Must be strictly in the past (validation: `dob < date.today()`). |
| `sex` | `Literal["M", "F"]` | Biological sex.  Must be exactly "M" or "F". |
| `race` | `str` | Race/ethnicity label drawn from the configured distrubtion (2020 US Census). |
| `zip_code` | `str` | US ZIP code produced by Faker. Must not be empty. |
| `age` | `int` (property) | Current age in whole years, computed from `dob` and today's date. Read-only.|


### `Encounter`
| Field | Type | Description |
| ----- | ---- | ----------- |
| `encounter_id` | `str` | UUID primary key. Auto-generated if omitted. |
| `patient_id` | `str` | Foreign key referencing `Patient.patient_id`. |
| `date` | `date` | Date of the clinical encounter. |
| `encounter type` | `Literal["outpatient", "inpatient", "emergency"]` | Must be one of
the three listed values.|
| `diagnoses` | `list[ICD10Code]` | One ormore diagnosis codes. Must be a `list`; an empty list is permitted by the dataclass but generators always produce at least one diagnosis. |

### `LabResult`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `result_id` | `str` | UUID Primary Key. Auto-generated if omitted. |
| `patient_id` | `str` | Foreign key referencing `Patient.patient_id`. |
| `encounter_id` | `str` | Foreign key referencing `Encounter.encounter_id`. |
| `test_name` | `str` | Name of lab test (e.g. 'Glucose') |
| `value` | `float` | Measured result value |
| `unit` | `str` | `Unit of measure (e.g., 'mg/dL') |
| `reference_low` | `float` | Lower bound of normal range.  Must be less than `reference_high`. |
| `reference_high` | `float` | Upper bound of normal range. Must be greater than `reference_low`. |
| `collection_date` | `date` | Date of collection |
| `is_abnormal` | `bool` (property) | `True` when value outside reference range (`>reference_high` or `<reference_low`). Read_only.|

Validation: `reference_low` must be strictly less than `reference_high`; construction raises `ValueError` otherwise.

### `ClinicalNote`

| Field | Type | Description |
| ----- | ---- | ----------- | 
| `note_id` | `str` | UUID primary key. Auto-generated if omitted. |
| `patient_id` | `str` | Foreign key referencing `Patient.patient_id`.|
| `encounter_id` | `str` | Foreign key referencing `Encounter.encounter_id`. |
| `note_type` | `Literal["soap","discharge_summary","referral"]` | Template type used. Must be one of the three listed values. |
| `text` | `str` | Full rendered note text. Must not be blank (whitespace-only raises ValueError). |
| `created_at` | `datetime` | Timestamp of note creation, set to a random time on the encounter date. |

----

## Lab Tests Reference

Each test is defined as a `LabSpec` named tuple in `_LAB_SPECS` inside `lab_values.py`.  The distribution column indicates whether values are drawn from a normal or lognormal distribution.  The conditions column lists diagnoses that increase the probability of an out-of-range result.

| Test | Normal Range | Unit | Distribution | Out-of-range Increasing Conditions |
| ---- | ------------ | ---- | ------------ | ---------------------------------- |
| Glucose | 70-100 | mg/dL | Normal | Diabetes (E10.x, E11.x, E13.x) - 3x multiplier |
| HbA1c | 4.0-5.6 | % | Normal | Diabetes (E10.x, E11.x, E13.x) - 3x multiplier |
| Total Cholesterol | 125-200 | mg/dL | Normal | None |
| LDL | 0-100 | mg/dL | Normal | None |
| HDL | 40-60 | mg/dL | Normal | None
| Systolic BP | 90 - 120 | mmHg | Normal | None |
| Creatinine | 0.6-1.2 | mg/dL | Normal | None |
| TSH | 0.4-4.0 | mIU/L | Lognormal | None |

The base probability of drawing and abnormal value (used when the diabetic multiplier does not apply) is : Gluecose 15%, HbA1c 10%, Total Cholesterol 12%, LDL 15%, HDL 10%, Systolic BP 18%, Creatinine 8%, TSH 8%. 

---

# ICD-10 Codes Reference

The following 30 codes are defined in `COMMON_ICD10_CODES` in src/synth_health/generators/encounter.py.Diagnoses are sampled from this list for every generated encounter.

### Cardiovascular
| Code | Description |
| ---- | ----------- |
| I10 | Essential (primary) hypertension |
| I25.10 | Atherosclerotic heart disease of native coronary artery without angina |
| I48.91 | Unspecified atrial fibrillation |
| I50.9 | Heart failure, unspecified |
| I63.9 | Cerebral infarction, unspecified |
### Endocrine / Metabolic
| Code | Description |
| ---- | ----------- |
| E11.9 | Type 2 diabetes mellitus without complications |
| E78.5 | Hyperlipidemia, unspecified |
| E66.09 | Other obesity due to excess calories |
| E03.9 | Hypothyroidism, unspecified |
| E11.65 | Type 2 diabetes mellitus with hyperglycemia |
### Mental Health
| Code | Description |
| ---- | ----------- |
| F41.1 | Generalized anxiety disorder |
| F32.9 | Major depressive disorder, single episode, unspecified |
| F10.10 | Alcohol use disorder, mild |
| F41.9 | Anxiety disorder, unspecified |
| F33.0 | Major depressive disorder, recurrent, mild |
### Gastrointestinal
| Code | Description |
| ---- | ----------- |
| K21.0 | Gastro-esophageal reflux disease with esophagitis |
| K57.30 | Diverticulosis of large intestine without perforation or abscess |
| K92.1 | Melena |
| K76.0 | Fatty (change of) liver, not elsewhere classified |
### Respiratory
| Code | Description |
| ---- | ----------- |
| J45.909 | Unspecified asthma, uncomplicated |
| J44.1 | Chronic obstructive pulmonary disease with acute exacerbation l
| J06.9 | Acute upper respiratory infection, unspecified |
| J18.9 | Pneumonia, unspecified organism |
### Musculoskeletal
| Code | Description |
| ---- | ----------- |
| M54.5 | Low back pain |
| M17.11 | Primary osteoarthritis, right knee |
| M79.3 | Panniculitis, unspecified |
### Genitourinary
| Code | Description |
| ---- | ----------- |
| N39.0 | Urinary tract infection, site not specified |
| N18.3 | Chronic kidney disease, stage 3 (moderate) |
### Preventive / Screening
| Code | Description |
| ---- | ----------- |
| 200.00 | Encounter for general adult medical examination without abnormal findings |
| 212.11 | Encounter for screening for malignant neoplasm of colon |

---

## Reproducibility

Every generator stores its own seeded `random.Random` instance (and, where applicable, a `numpy.random.default_rng` instance) constructed from the `seed` argument passed at initialization.  The pipeline propagates the same master seed to all four generators:

```python
self._patient_gen = PatientGenerator (seed=seed, ...)
self.encounter_gen = EncounterGenerator (seed=seed)
self.lab_gen = LabGenerator (seed=seed, ...)
self.note_gen = NoteGenerator (seed=seed)
```

Additionally, the global random module and global numpy. random" state are both seeded at pipeline construction time (random.seed (seed), np.random.seed (seed)) to guard against any library code that draws from the global RNG rather than an explicit instance.

Given the same `seed` and the same parameter values, two pipeline runs will produce byte-for-byte identical output.

**When reproducibility can break:**

- **Different Faker versions** - Faker's name and zip-code databases change between releases.  Upgrading `faker` will produce different names and zip codes for the same seed.  Pin `faker` in your lockfile if exact reproducibility across environments is required.
- **Different NumPy versions** - The `numpy.random.default_rng` bit-generator is stable within a major NumPy series, but the underlying distributino algorithms can change between NumPy major versions.  Pin `numpy` to avoid drift.
- **Changing pipeline parameters** - Altering any constructor argument changes the sequence of draws and all downstream output, even if the seed is the same.

## Extending the generators

### Adding a new lab test
Open `src/synth_health/generators/notes.py` and append a new `LabSpec` entry to the `_LAB_SPECS` list:

```python
LabSpec(
    test_name="Sodium",
    unit="mEq/L",
    reference_low=136.0,
    reference_high=145.0,
    mean=140.0,
    std=2.5,
    lognormal=False,
    base_abnormal_prob=0.06,
), ...
```

The new test will automatically be included in every encounter that receives labs.  No changes to the pipeline or schema are needed.

### Adding a new ICD-10 code

open `src/synth_health/generators/encounter.py` and append a new `ICD10Code1 to the `COMMON_ICD10_CODES` list:

```python
ICD10Code("C34.10", "Malignant neoplasm of upper lobe, bronchus or lung, unspecified side"),
```

The new code becomes part of the diagnosis pool samples for all future encounters.

### Adding a new note template

Open `src/synth_health/generators/notes.py`. Add a new top-level string constant for the template, add any content-bank lists the template draws from, and then add a `_‹type›` method to `NoteGenerator`. Finally, update `_select_note_type` to return the new type for appropriate encounter types, and add a corresponding branch in `generate()` to dispatch the new renderer. 

The new note type will also need to be added to the `note_type` `Literal` in `schema.py` and its `__post_init__` validation tuple to avoid `ValueError` at construction time.

---

# Running tests

```bash
pip install -e ". [dev]"
pytest tests/ -v
```

The test suite contains three files:

- ** tests/test_schema.py ** - Unit tests for all five schema dataclasses. Covers valid construction, UID auto-generation, the age"andis_abnormal computed properties, and all _post_init •validation rules (future dob, invalid sex, inverted reference range, blank note text, invalid encounter_type, invalid note_type).

- ** tests/test_generators.py ** - Tests for each generator in isolation. Covers correct output counts, ageand sex distributional properties,encounter type validity,diagnosis presence, lab fraction compliance, value plausibility (non-negative, not wildly out of range), collection date proximity to encounter date, note type assignment rules (inpatient always gets discharge summary), note text non-emptiness, and the presence of the synthetic disclaimer in every note. Also tests that each generator raises ValueError for invalid constructor arguments.

- ** tests/test_pipeline.py ** - Integration tests for `SynthHealthPipeline`. Covers output dictionary shape, patient/encounter/lab/note counts, referential integrity across al1 four entity types (every encounter/lab/note foreign key resolves to a known patient or encounter), `DataFrame` column sets for all four entities, csv file creation and non-emptiness, directory auto-creation by `to_csv()`, and reproducibility (same seed produces identical first patient across two independent pipeline runs). Also verifies that `to_dataframes()` raises RuntimeError before `run()` is called.

## Known limitations

- **Not a clinical data model.** The schema is a simplified custom model. There is no FHIR compliance, no HL7 messaging support, and no adherence to OMOP or any other standard clinical data standard.

- **Fixed note templates.** All three note types (SOAP, discharge summary, referral) use a single string template each. Content is varied by selecting phrases from fixed lists, not by generative text. This produces limited linguistic diversity, which may be insufficient for LP tasks that require diverse free-text variation.

- **Lab distributions are approximations.** The mean, standard deviation, and abnormal probability values in `_LAB_SPECS` are reasonable approximations but have not been validated against clinical reference literature. They should not be used for any purpose that requires clinically accurate value distributions.

- **No pediatric patients.** The default age min' is 18. The age min" parameter can be set lower, but no adjustments are made to lab reference ranges, encounter types, or note content for pediatric patients.

- **No longitudinal patient model.** Encounters are independently generated for each patient with no memory of prior encounters. There is no modelling of disease progression, treatment response, or temporal correlation between successive visits.

## Data disclaimer

All data produced by this project is **fully synthetic and fictitious**. No real patient records, names, addresses,diagnoses, or clinical data are used at any point in the generation process. This project is intended solely for software demonstration, data engineering testing, and MI pipeline development purposes.