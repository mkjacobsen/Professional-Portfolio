"""Regex-based structural PII matchers.

Each pattern targets a well-defined format (SSN, phone, email, etc.) where
the structure itself is sufficient evidence of PII - no surrounding context
needed.  These matchers assign confidence = 1.0 because a valid structural
match is essentially never a false positive in an HR/legal document
"""

import re
from .types import PIIMatch, PIIType, DetectionMethod

# Compiled Patterns

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b") #Social Security Number

_PHONE_PATTERN = re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b") #North American Phone Number

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9._\-]+\.[A-Za-z]{2,}\b") #Email address similar to RFC-5321

_DOB_PATTERN = re.compile(
    r"{?:DOB:?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}}",
    r"|date\s+of\s+birth:?\s*\w+\s+\d{1,2},?\s*\d{4})",
    re.IGNORECASE,
) # Date of Birth with front indicator

_CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
) # Credit card formatted in standard Visa, Mastercard, Discover, or AmEx pattern

# Mapping keeps interation order consistent and simplified new type addition
_PATTERNS: list[tuple[PIIType, re.Pattern]] = [
    (PIIType.SSN, _SSN_PATTERN),
    (PIIType.PHONE, _PHONE_PATTERN),
    (PIIType.EMAIL, _EMAIL_PATTERN),
    (PIIType.DATE_OF_BIRTH, _DOB_PATTERN),
    (PIIType.CREDIT_CARD, _CREDIT_CARD_PATTERN),
]

class RegexMatcher:
    """Apply all structural regex patterns to a text and return PII matches.
    
    Matches from different patterns are collected, then any overlapping spans 
    within the *same* pass are resolved by keeping the match with the earlier
    start position (ties broken by longer match).  This prevents a single token
    like an email or phone number from being returned twice is patterns overlap.
    """

    def __init__(self) -> None:
        self._patterns: list[tuple[PIIType, re.Pattern]] = list(_PATTERNS)

    # Public API

    def match(self, text: str) -> list[PIIMatch]:
        """Return all non-overlapping PII matches found in *text*.
        
        All regex matches have confidence = 1.0.
        Results are sorted by start offset.
        """

        raw: list[PIIMatch] = []
        for pii_type, pattern in self._patterns:
            for m in pattern.finditer(text):
                raw.append(
                    PIIMatch(
                        pii_type = pii_type,
                        text = m.group(),
                        start = m.start(),
                        end = m.end(),
                        method = DetectionMethod.REGEX,
                        confidence = 1.0,
                    )
                )

        return _resolve_overlaps(sorted(raw, key=lambda x: (x.start, -(x.end - x.start))))

# Helpers

def _resolve_overlaps(matches: list[PIIMatch]) -> list[PIIMatch]:
    """Given matches sorted by (start, -length), keep the first (longest)
    match at each postiion and discard any later match that overlaps it."""

    result: list[PIIMatch] = []
    for candidate in matches:
        if result and candidate.start < result[-1].end:
            # Overlaps the previous accepted match - skip it.
            continue
        result.append(candidate)

    return result

