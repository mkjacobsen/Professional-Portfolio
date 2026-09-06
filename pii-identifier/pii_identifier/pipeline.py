"""Hybrid PII detection pipeline

Combines a high-precision regex layer with a high-recall NER layer.  The merge
step resolves conflice in favour of reges: wherever a regex match and an NER
match share any overlapping characters, the regex match is kept and the NER match
is discarded.  This prevents NER from "couble-counting" entities that are already caught
by a deterministic rule (e.g. SSNs tagged by spaCy as CARDINAL).
"""

from __future__ import annotations

from .patterns import RegexMatcher
from .ner import NERMatcher
from .types import PIIMatch, PIIType, DetectionMethod

class PIIIdentifier:
    """Two-stage identification pipeline for HR/legal documents.
    
    Parameters:
    
    use_ner: 
        Set to ``False`` to run regex only (useful for environment where 
        spaCy is not available or for pure-speed use cases).
    ner_model:
        spaCy model name passed through to :class:`ner.NERMatcher`.
    min_confidence:
        Matches below this threshold are discarded before returning results. 
        Regex matches always have confidence 1.0 so this only affects NER.    
    """

    def __init__(
        self,
        use_ner: bool = True,
        ner_model: str = "en_core_web_sm",
        min_confidence: float = 0.7,
    ) -> None:
        self.regex_matcher = RegexMatcher()
        self.ner_matcher: NERMatcher | None = NERMatcher(ner_model) if use_ner else None
        self.min_confidence = min_confidence


    # Public API

    def identify(self, text: str) -> list [PIIMatch]:
        """Detect all PII in *text* and return a deduplicated, sorted list.
        Algorithm
        1. Run regex matcher - high-confidence structural matches.
        2. Run NER matcher- contextual entity matches.
        3. Merge: drop any NER match whose span overlaps a regex match.
        4. Filter by min confidence
        5. Sort by start offset.
        """
        regex_matches = self.regex_matcher.match(text)

        if self.ner_matcher is not None:
            ner_matches = self.ner_matcher.match(text)
            ner_filtered = _remove_overlapping_ner(regex_matches, ner_matches)

        else:
            ner_filtered = []

        combined = regex_matches + ner_filtered
        above_threshold = [m for m in combined if m.confidence >= self.min_confidence]
        return sorted (above_threshold, key-lambda m: m.start)

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Replace every PII span in *text* wdth *replacement*.
        Processes matches right-to-left so that earlier character offsets
        remain valid as we mutate the string.
        """
        matches = self.identify(text)

        # Sort descending by start to process right-to-left.
        for match in sorted (matches, key=lambda m: m.start, reverse=True) :
            text = text[:match.start] + replacement + text[match.end:]
        return text

    def report (self, text: str) -> dict:
        """Return a structured report dict for *text*.
        Keys
        matches:
            Full list of :class: ~pil_identifier.types.PIIMatch" objects.
        pii_types_found:
            Deduplicated list of PII type value strings (e.g. ("ssn", "email")).
        total_count:
            Number of distinct PII matches.
        redacted_text
            The original text with all PII replaced by [REDACTED) 
        
        """
        matches = self.identify(text)

        return {
            "matches": matches,
            "pii_types_found": sorted({m.pii_type.value for m in matches}),
            "total_count": len(matches),
            "redacted_text": self.redact(text),
        }
# Internal helpers

def _spans_overlap(a: PIIMatch, b: PIIMatch) -> bool:
    """ Return True if two matches share at least one character."""
    return max(a.start, b.start) < min(a.end, b.end)

def _remove_overlapping_ner(
    regex_matches: list[PIIMatch], 
    ner_matches: list[PIIMatch],
) -> list[PIIMatch]:
    """Return only the NER matches that do*not*overlap any regex match."""
    kept: list [PIIMatch] = []

    for ner_match in ner_matches:
        if not any(_spans_overlap(ner_match, rm) for rm in regex_matches):
            kept.append(ner_match)

    return kept