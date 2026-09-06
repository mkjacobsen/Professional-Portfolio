"""HR Document Generator with Injected PII

Each generator method returns a ``(document_text, ground_truth)`` tuple where
``ground_truth`` is a list of dicts describing every PII value that was
deliberately inserted:

    {"text": "Jane Dow", "pii_type": "person_name", "start": 42, "end": 50}

Offsets are computed precisely by building documents as a sequence of
``(segment_text, pii_type_or_None)`` tuples, then joining them while
accumulating the running character position.  This is more reliable than
``str.find()`` because it handles the case where the same value happens to
appear more than once in the template prose.
"""

from __future__ import annotations

import email
import random
from faker import Faker

from .types import PIIType

# Type alias for ground-truth records
GroundTruth = list[dict]

# Document Builder Helper

class _DocBuilder:
    """Accumulate text segments and track PII offsets precisely."""

    def __init__(self) -> None:
        self._parts: list[tuple[str, str | None]] = [] # (text, pii_type_value | None)

    def add(self, text: str, pii_type: PIIType | None = None) -> "_DocBuilder":
        """Append a text segment, optionally tagged as PII."""
        self._parts.append((text, pii_type.value if pii_type is not None else None))
        return self

    def build(self) -> tuple[str, GroundTruth]:
        """Join all segments and return ``(full_text, ground_truth_list)``"""
        full_text = ""
        ground_truth: GroundTruth = []
        for segment, pii_type_value in self._parts:
            if pii_type_value is not None:
                start = len(full_text)
                end = start + len(segment)
                ground_truth.append(
                    {"text": segment, "pii_type": pii_type_value, "start": start, "end": end}
                )

            full_text += segment
        return full_text, ground_truth


# Main Injector Class

class HRDocumentInjector:
    """Generate realistic HR documents with deterministic, seeded PII

    Parameters

    seed: 
        Random seed passed to both :class: `~faker.Faker` and Python's 
        ``random`` module, ensuring reproducible output across runs.
    """

    def __init__(self, seed: int = 42) -> None:
        _ROLES = {
             "Senior Software Engineer": "Engineering",
             "Data Analyst": "Operations",
             "Product Manager": "Marketing",
             "HR Business Partner": "Human Resources",
             "Financial Analyst": "Finance"
        }

        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)

        self.name = self.fake.name()
        self.street = self.fake.street_address()
        self.city = self.fake.city()
        self.state = self.fake.state_abbr()
        self.zipcode = self.fake.zipcode()
        self.address = f"{self.street}, {self.city}, {self.state} {self.zipcode}"
        self.ssn = self.fake.ssn()
        self.dob_date = self.fake.date_of_birth(minimum_age=22, maximum_age=65)
        self.dob_str = f"DOB: {self.dob_date.strftime("%m/%d/%Y")}"
        self.phone = self.fake.numerify("###-###-####")
        self.email = self.fake.email()
        self.emergency_name = self.fake.name()
        self.emergency_phone = self.fake.numerify("###-###-####")
        self.prev_employer = self.fake.company()
        
        self.start_date = self.fake.date_between(start_date="+14d", end_date="+60d").strftime("%B %d, %Y")
        self.salary = f"${random.randint(70,200) * 1_000:,}"

        self.hr_email = self.fake.company_email()
        self.hr_phone = self.fake.numerify("###-###-####")
        self.company = self.fake.company()
        self.role = random.choice(list(_ROLES.keys()))
        self.department = _ROLES[self.role]
        self.employee_id = self.fake.bothify("EMP-####??").upper()
        self.manager = self.fake.name()

    # Document Generators

    def offer_letter(self) -> tuple[str, GroundTruth]:
        """~300-word employment offer letter with seeded PII."""
        b = _DocBuilder()
        b.add(f"{self.company}\nOFFER OF EMPLOYMENT\n\nDear ")
        b.add(self.name, PIIType.PERSON_NAME)
        b.add(",\n\nWe are pleased to extend this offer of employment for the position of ")
        b.add(f"{self.role} at {self.company}. Your mailing address on file is ")
        b.add(self.address, PIIType.ADDRESS)
        b.add(f".\n\nYour employment will commence on {self.start_date}, reporting directly to "
              f"{self.manager}.  Your starting annual compensation will be {self.salary}, paid "
              f"bi-weekly.  You will also be eligible for the standard benefits package "
              f"beginning on your first day of employment.\n\n"
              f"As part of the onboarding process, you will be required to complete an"
              f"I-9 Employment Eligibility Verification form.  Please provide your SSN: "
        )
        b.add(self.ssn, PIIType.SSN)
        b.add(
             f" for I-9 and payroll processing purposes. This information will be handled "
             f"in strict accordance with applicable privacy regulations.\n\n"
             f"This offer is contingent upon successful completion of a background check "
             f"and reference verification.  Please sign and return this letter within five "
             f"(5) business days to confirm your acceptance.\n\n"
             f"Should you have any questions, please do not hesitate to contact our HR "
             f"department at "
        )
        b.add(self.hr_email, PIIType.EMAIL)
        b.add(" or by phone at ")
        b.add(self.hr_phone, PIIType.PHONE)
        b.add(
             f".\n\nWe look forward to welcoming you to the {self.company} team.\n\n"
             f"Sincerely,\nHuman Resources\n{self.company}"
        )
        return b.build()

    def intake_form(self) -> tuple[str, GroundTruth]:
        """~200-word employee intake / new-hire information form."""
        b = _DocBuilder()
        b.add(
            "EMPLOYEE INTAKE FORM\nPlease complete all fields in full. \n\n"
            f"Employee ID: {self.employee_id} \n"
            f"Department: {self.department} \n\n"
            f"--- PERSONAL INFORMATION ---\n"
            f"Full Name: "
        )
        b.add(self.name, PIIType.PERSON_NAME)
        b.add("\n")
        b.add(self.dob_str, PIIType.DATE_OF_BIRTH)
        b.add(
            f"\n \n--- CONTACT INFORMATION ---\n"
            f"Primary Phone: "
        )
        b.add (self.phone, PIIType.PHONE)
        b.add ("\nWork Email: ")
        b.add (self.email, PIIType.EMAIL)
        b.add (
            f"\n \n--- EMERGENCY CONTACT ---\n"
            f"Contact Name: "
        )
        b.add (self.emergency_name, PIIType.PERSON_NAME)
        b.add (f"\nRelationship: ")
        b.add (random. choice (["Spouse", "Parent", "Sibling", "Friend"]))
        b.add (
            f"\n\n--- ACKNOWLEDGEMENT ---\n"
            f"By submitting this form, I confirm that the information provided above"
            f"is accurate and complete to the best of my knowledge. I understand that"
            f"any material misrepresentation may result in disciplinary action up to"
            f"and including termination of employment. \n\n Date: ______________"
            f"Signature: ______________"
        )
        return b.build()

    def background_check_consent(self) -> tuple[str, GroundTruth]:
        """~150-word background check authorization and consent form"""
        b = _DocBuilder()
        b.add(
            f"BACKGROUND CHECK AUTHORIZATION AND CONSENT FORM\n\n"
            f"I, "
        )
        b.add(self.name, PIIType.PERSON_NAME)
        b.add(
            f", hereby authorize {self.company} and its designated consumer reporting "
            f"agency (CRA) to conduct a thorough background investigation as a "
            f"condition of my employment or continued employment.\n\n"
            f"--- APPLICANT INFORMATION ---\n"
            f"Full Legal Name: "
        )
        b.add(self.name, PIIType.PERSON_NAME)
        b.add(f"\nSocial Security Number: ")
        b.add(self.ssn, PIIType.SSN)
        b.add(f"\n")
        b.add(self.dob_str, PIIType.DATE_OF_BIRTH)
        b.add(f"\nCurrent Address: ")
        b.add(self.address, PIIType.ADDRESS)
        b.add(
            f"\n \n--- EMPLOYMENT HISTORY ---\n"
            f"Most Recent Employer: "
        )
        b.add(self.prev_employer, PIIType. ORGANIZATION)
        b.add(
            f"\n\nThe background check may include, but is not limited to: criminal "
            f"history, employment verification, education verification, credit history,"
            f"and reference checks. The investigation will be conducted in accordance "
            f"with the Fair Credit Reporting Act (FCRA) and all applicable state laws.\n\n"
            f"I understand that a copy of any report obtained may be made available - "
            f"to me upon written request. I release (company), its agents, and all "
            f"persons providing information from any liability in connection with this "
            f"investigation. \n\n"
            f"Signature: ______________ Date: \n"
        )
        return b.build()

    def generate_batch (self, n: int = 10) -> list[tuple[str, GroundTruth]]:
        """Return *n* documents of randomly chosen types with injected PII. """
        generators = [
            self. offer_letter,
            self. intake_form,
            self.background_check_consent,
        ]
        return [random.choice(generators)() for _ in range(n)]