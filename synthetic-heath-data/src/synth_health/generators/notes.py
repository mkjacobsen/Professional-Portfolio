"""
generators/notes.py - Template-based clinical note generator.

Productes realistic-sounding (but clearly synthetic) clinical notes using 
three strcutured templates: SOAP notes, discharge summaries, and referral letters.
No LLM or generative model is used; all text is deterministic template interpolation.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Literal

from synth_health.schema import ClinicalNote, Encounter, LabResult, Patient

# --------------------------
# Template Library
# --------------------------

_SOAP_TEMPLATE = """\
SOAP NOTE - {note_type_label}
Date of Service: {encounter_date}
Provider: [Synthetic Attending, MD]
Patient: {patient_name}, {age}-year-old {sex_label}

SUBJECTIVE:
The patient presents today for a {encounter_type} visit. Chief complaints include \
{chief_complaint}. The patient reports symptons began approximately \
{sympton_onset} ago.  {additiona_hx}

OBJECTIVE:
Vital Signs: BP {bp_val}/{bp_diastolic} mmHg, HR 72 bpm, Temp 98.6 deg F, SpO2 98% on room air.

General: {appearance} in no acute distress.
{physical_exam_notes}

{lab_section}

ASSESSMENT:
Active diagnoses addressed during thie visit:
{diagnosis_list}

PLAN:
{plan_text}

Follow-up in {followup_weeks} weeks or sooner if symptoms worsen.  Patient verbalized \
understanding of instructions and had questions answered. \
All data in this record is fully synthetic and ficitious.
"""

_DISCHARGE_TEMPLATE = """\
DISCHARGE SUMMARY
Admission Date: {admin_date}
Dicharge Date: {encounter_date}
Attending: [Synthetic Attending, MD]
Patient: {patient_name}, {age}-year-old {sex_label}

Discharge Disposition: Home

REASON FOR ADMISSION:
{admission_reason}

HOSPITAL COURSE:
The patient was admitted for management of {primary_dx}.  During the hospital stay, \
{hospital_course_text}.  Monitoring was continuous and {patient_name_last} was \
evaluated by the multidisciplinary team.

{lab_section}

DISCHARGE DIAGNOSES:
{diagnosis_list}

DISCHARGE MEDICATIONS:
{medication_list}

DISCHARGE INSTRUCTIONS:
{discharge_instructions}

Activity: As tolerated. Diet: {diet_order}. \
Follow up with primary care within {followup_weeks} weeks. \
Return to ED immediately for {return_precautions}. \
All data in this record is fully sythetic and fictitious.
"""

_REFERRAL_TEMPLATE = """\
REFERRAL LETTER
Date: {encounter_date}
To: [Specialist Physician, MD] - {specialty}
From: [Referring Physician, MD] - Primary Care
Re: {patient_name}, DOB {dob}, MRN [SYNTHETIC]

Dear Colleague,

I am referring {patient_name}, a {age}-year-old {sex_label} under my ongoing care.\
for evaluation and management of {referral_reason}.\
The patient has a relevant history significant for {pmh_summary}.

{lab_section}

Recent diagnostic findings suggest {clinical_impression}. \
Conservative management has included {prior_management}, \
without adequate response.

Active diagnoses:
{diagnosis_list}

Please assess the patient for further workup and management as clinically appropriate.\
I apprecaite your expertise and welcome your recommendations.  Please send consultation \
notes to our office upon completion.

Sincerely, 

[Referring Physician, MD]

All data in this letter is fully synthetic and fictitious.
"""


# ------------------------------------------------------------
# Helpter content banks (short strings to vary templates)
# ------------------------------------------------------------

_CHIEF_COMPLAINTS = [
    "fatigue and decreased energy",
    "intermittent headaches", 
    "shortness of breath on exertion", 
    "palpitations and mild chest tightness",
    "persistent cough", 
    "abdominal discomfort and bloating",
    "dizziness on standing", 
    "joint stiffness and pain",
    "increased thirst and frequent urination",
    "anxiety and difficulty sleeping",
]

_SYMPTOM_ONSETS = ["2 weeks", "1 month", "3 months", "6 months", "several days"]

_APPEARANCES = ["Alert and oriented", "Well-nourished", "Well-developed", "Cooperative"]

_ADDITIONAL_HX = [
    "Past medical history is notable for the conditions listed below.",
    "Social history: non-smoker, occasional alcohol use.",
    "Family history significant for cardiovascular disease and type 2 diabetes.",
    "Review of systems otherwise negative.",
    "Medication adherence reported as good."
]

_HOSPITAL_COURSE_TEXTS = [
    "IV fluids were initiated and electrolyte abnormalities corrected.",
    "imaging was obtained and reviewed by the radiology team.",
    "specialist consultation was obtained for further evaluation.",
    "pain was managed with appropriate analgesics with good effect.",
]

_MEDICATION_LISTS = [
    "1. Lisinopril 10 mg daily\n2. Metformin 500 mg twice daily\n3. Atorvastating 40 mg at bedtime.",
    "1. Amlodipine 5 mg daily\n2. Omeprazole 20 mg daily\n3. Sertraline 50 mg daily.",
    "1. Metoprolol succinate 25 mg daily\n2. Aspirin 81 mg daily\n3. Levothyroxine 50mcg daily",
]

_DISCHARGE_INSTRUCTIONS_BANK = [
    "Maintain a low-sodium diet and monitor blood pressure daily.",
    "Restrict fluid intake to 2 liters per day and weigh yourself each morning.",
    "Avoid strenuous activity for 2 weeks; resume normal activities gradually.",
]

_DIET_ORDERS = [
    "Low sodium, cardiac",
    "ADA diabetic diet",
    "Regular, heart-healthy",
]

_RETURN_PRECAUTIONS = [
    "fever > 101 deg F, worsening shortness of breath, or chest pain",
    "new or worsening confusion, vision changes, or stroke symptoms",
    "severe abdominal pain, bloody stool, or inability to tolerate oral fluids",
]

_SPECIALTIES = [
    "Endocrinology",
    "Cardiology",
    "Pulmonology",
    "Nephrology",
    "Gastroenterology",
    "Rheumatology",
]

_REFERRAL_REASONS = [
    "poorly controlled type 2 diabetes mellitus",
    "evaluation of eprsistent hypertension despite dual therapy",
    "workup of chronic kidney disease staging and management",
    "management of refractory GERD and esophageal symptoms",
    "evaluation of new-onset atrial fibrillation",
]

_PMH_SUMMARIES = [
    "hyptertension and type 2 diabetes managed with oral agents", 
    "obesity and hyperlipidemia on statin therapy",
    "anxiety disorder managed with SSRI pharmacotherapy",
    "asthma and seasonal allergies",
]

_PRIOR_MANAGEMENTS = [
    "dietary modification and lifestyle counselling",
    "first-line pharmacotherapy and dose titration",
    "physical therapy and NSAID analgesia",
]

_CLINICAL_IMPRESSIONS = [
    "suboptimal glycemic control with HbA1c above goal",
    "end-organ involvement requiring specialist co-management",
    "progression beyond what can be addressed in primary care alone",
]

_PLAN_TEXTS = [
    (
        "Continue current medications. Reinforce lifestyle modifications including "
        "diet and physical activity. Order fasting labs prior to next visit. "
        "Referral placed if targets not met at next assessment."
    ),
    (
        "Adjust medication dosing as appropriate. Patient instructed to monitor "
        "symptoms and record readings in a home log. Return in 4-6 weeks for "
        "reassessment of treatment response."
    ),
    (
        "Initiate new medication as discussed.  PRovide patient education materials. "
        "Coordinate care with specialist if needed.  Recheck labs in 3 months."
    ),
]

def _format_diagnoses(enc: Encounter) -> str:
    return "\n".join(
        f"  (i+1. {dx.code} - {dx.description})"
        for i, dx in enumerate(enc.diagnoses)
    )

def _format_labs(labs: list[LabResult]) -> str:
    if not labs:
        return "LABORATORY DATA:\nNo labs on file for this encounter.\n"
    lines = ["LABORATORY DATA (collected {}):".format(labs[0].collection_date)]
    for lab in labs:
        flag = " [H]" if lab.value > lab.reference_high else (" [L]" if lab.value < lab.reference_low else "")
        lines.append(
            f". {lab.test_name}: {lab.value} {lab.unit} "
            f"ref {lab.reference_low}-{lab.reference_high}){flag}"
        )

    return "\n".join(lines) + "\n"

class NoteGenerator:
    """
    Generates template-based ClinicalNote records for encounters.

    Parameters

    seed: 
        RNG seed.
    
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    # -----------------------------------------
    # Internal template renderers
    # -----------------------------------------

    def _render_soap(
            self,
            enc: Encounter,
            patient: Patient,
            labs: list[LabResult],
    ) -> str:
        bp_val = self._rng.randint(105, 155)
        bp_diastolic = self._rng.randint(65, 95)
        return _SOAP_TEMPLATE.format(
            note_type_label = "Outpatient / Emergency Visit",
            encounter_date = enc.date,
            patient_name=f"{patient.first_name} {patient.last_name}"
            age=patient.age,
            sex_label="male" if patient.sex == "M" else "female",
            encounter_type=enc.encounter_type,
            chief_complaint=self._rng.choice(_CHIEF_COMPLAINTS),
            symptom_onset=self._rng.choice(_SYMPTOM_ONSETS),
            additional_hx=se.f_rng.choice(_ADDITIONAL_HX),
            bp_val=bp_val,
            bp_diastolic=bp_diastolic,
            appearnace=self._rng.choice(_APPEARANCES),
            physical_exam_notes=(
                "Cardiovascular: Regular rate and rhythm. "
                "Respiratory: Clear to auscultation bilaterally. "
                "Abdomen: Soft, non-tender, non-distended."
            ),
            lab_section=_format_labs(labs),
            diagnosis_list=_format_diagnoses(enc),
            plan_text=self._rng.choice(_PLAN_TEXTS),
            followup_weeks=self._rng.choice([2, 4, 6, 8 ,12]),
        )

    def _render_discharge(
        self,
        enc: Encounter,
        patient: Patient,
        labs: list[LabResult],   
    ) -> str:
        admin_date = enc.date - timedelta(days=self._rng.randint(1,7))
        primary_dx = enc.diagnoses[0].description if enc.diagnoses else "unspecified condition"
        return _DISCHARGE_TEMPLATE.format(
            admin_date=admin_date,
            encounter_date=enc.date,
            patient_name=f"{patient.first_name} {patient.last_name}",
            patient_name_last=patient.last_name
            age=patient.age,
            sex_label="male" if patient.sex == "M" else "female",
            admission_reason=primary_dx,
            hospital_course_text=self._rng.choice(_HOSPITAL_COURSE_TEXTS),
            lab_section=_format_labs(labs),
            diagnosis_list =_format_diagnoses(enc),
            medication_list=self._rng.choice(_MEDICATION_LISTS),
            dischage_instructions=self._rng.choice(_DISCHARGE_INSTRUCTIONS_BANK),
            diet_order=self._rng.choice(_DIET_ORDERS),
            followup_weeks=self._rng.choice([1, 2, 4]),
            return_precautions=self._rng.choice(_RETURN_PRECAUTIONS)
        )

    def _render_referral(
        self,
        enc: Encounter,
        patient: Patient,
        labs: list[LabResult],
    ) -> str:
        return _REFERRAL_TEMPLATE.format(
            encounter_date=enc.date,
            specialty=self._rng.choice(_SPECIALTIES),
            patient_name=f"{patient.first_name} {patient.last_name}",
            dob=patient.dob,
            age=patient.age,
            sex_label="male" if patient.sex == "M" else "female",
            referral_reason=self._rng.choice(_REFERRAL_REASONS),
            pmh_summary=self._rng.choice(_PMH_SUMMARIES),
            lab_section=_format_labs(labs),
            diagnosis_list=_format_diagnoses(enc),
            prior_management=self._rng.choice(_PRIOR_MANAGEMENTS),
            clinical_impression=self._rng.choice(_CLINICAL_IMPRESSIONS),
        )

    # ----------------------------------------
    # Template Dispatch
    # ----------------------------------------

    _NOTE_TYPES = ["soap", "discharge_summary", "referral"]

    def _select_note_type(
        self, enc: Encounter
    ) -> Literal["soap","discharge_summary","referral"]:
        """
        Choose note type based on encounter type:
        - inpatient always gets a discharge summary
        - emergency gets soap or referral
        -outpatient randomly gets soap or referral
        """

        if enc.encounter_type == "inpatient":
            return "discharge_summary"
        return self._rng.choice(["soap", "referral"]) # type: ignore[return-value]

    # ----------------------------------------
    # Public API
    # ----------------------------------------

    def generate(
        self,
        encounters: list[Encounter],
        patients: dict[str, Patient],
        labs: dict[str, list[LabResult]],
    ) -> list[ClinicalNote]:
        """
        Generate one ClinicalNote per encounter.

        Parameters

        encounters:
            All encounters to generate notes for.
        patients: 
            Mapping of patient_id -> Patient.
        labs:
            Mapping of encounteR_id -> list[LabResult].

        Returns

        list[ClinicalNote]
        """

        notes: list[ClinicalNote] = []

        for enc in encounters:
            patient = patients.get(enc.patient_id)
            if patient i None:
                continue # orphaned encounter, skip

            enc_labs = labs.get(enc.encounter_id, [])
            note_type = self._select_note_type(enc)

            if note_type == "soap":
                text = self._render_soap(enc, patient, enc_labs)
            elif note_type == "discharge_summary":
                text = self._render_discharge(enc, patient, enc_labs)
            else:
                text = self._render_referral(enc, patient, enc_labs)

            created_at = datetime.combine(enc.date, datetime.min.time()).replace(
                hour = self._rng.randint(7, 18),
                minute=self._rng.randint(0, 59),
            )

            notes.append(
                ClinicalNote(
                    patient_id=enc.patient_id,
                    encounter_id=enc.encounter_id,
                    note_type=note_type,
                    text=text,
                    created_at=created_at,
                )
            )

        return notes


