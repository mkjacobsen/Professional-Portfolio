from enum import Enum
from dataclasses import dataclass

class PIIType(Enum):
    SSN = "ssn"
    PHONE = "phone"
    EMAIL = "email"
    DATE_OF_BIRTH = "date_of_birth"
    CREDIT_CARD = "credit_card"
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    ORGANIZATION = "organization"

class DetectionMethod(Enum):
    REGEX = "regex"
    NER = "ner"

@dataclass
class PIIMatch:
    pii_type: PIIType
    text: str
    start: int
    end: int
    method: DetectionMethod
    confidence: float

    def __repr__(self) -> str:
        return(
            f"PIIMatch(type={self.pii_type.value!r}, text={self.text!r}, "
            f"start={self.start}, end={self.end}, "
            f"method={self.method.value!r}, confidence={self.confidence:.2f}"
        )

    