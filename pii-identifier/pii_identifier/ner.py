"""spaCy NER wrapper for contextural PII detection.

Named-entity recognition excells at identifying PII that has no rigid
structural form: person names, addresses embedded in prose, and organization
names.  It complements the regex layer by handling the "soft" surface forms
that rules alone cannot capture.
"""

import spacy
from spacy.language import Language

from .types import PIIMatch, PIIType, DetectionMethod

# spaCy entity lables -> PIIType mapping
# We intentinoally skip DATE, CARDINAL, MONEY, PERCENT, TIME, ORDINAL, etc.
# because in HR documents these appear constantly as non-PII (salaries, dates, durations)
# and generate far too many false positives.
_LABEL_MAP: dict[str, PIIType] = {
    "PERSON": PIIType.PERSON_NAME,
    "GPE": PIIType.ADDRESS, #Geo-political entity (cities, states, countries)
    "LOC": PIIType.ADDRESS, #None-geo-political entity locations (regions, bodies of water)
    "FAC": PIIType.ADDRESS, #Facilities (buildings, airports, bridges)
    "ORG": PIIType.ORGANIZATION,
}

# Default confidence when spaCy does not expose a per-entity probability.
_DEFAULT_CONFIDENCE = 0.85

class NERMatcher:
    """ Wrap a spaCy pipeline and return PIIMatch objects for relevant entities.
    
    Parameters:
    
    model:
        spaCy model name. Defaults to "en_core_web_sm".  You can swap in a larger
        model for higher recall on ambiguous names and addresses
        without changing any other code.
    """

    def __init__(self, model: str="en_core_web_sm") -> None:
        self.nlp: Language = spacy.load(model)

    # Public API

    def match(self, text: str) -> list[PIIMatch]:
        """Run spaCy NER on *text* and return mapped PII matches.
        
        Only entity labels present in ``_LABEL_MAP`` are returned; all other
        spaCy entity types are silently dropped.  Confidence is taken from the 
        span's ``kb_id_`` score if available, otherwise ``_DEFAULT_CONFIDENCE``.
        """

        doc = self.nlp(text)
        results = list[PIIMatch] = []

        for ent in doc.ents:
            pii_type = _LABEL_MAP.get(ent.label_)
            if pii_type is None:
                continue
            confidence = _entity_confidence(ent)

            results.append(
                PIIMatch(
                    pii_type = pii_type,
                    text = ent.text,
                    start = ent.start_char,
                    end = ent.end_char,
                    method = DetectionMethod.NER,
                    confidence = confidence
                )
            )

        return results

# Helpers

def _entity_confidence(ent) -> float: # type: ignore[no-untyped-def]
    """Extract a confidence score from a spaCy span, with a safe fallback.
    
    spaCy's ``en_core_web_sm`` does not attach per-entity probabilities by
    default.  The transformer-based models do expose span scores via
    ``doc.spans`` pipelines, but that API varies.  We keep this simple:
    return the default confidence unless the span exposes something better.
    """

    # Some custom pipelines store the socre as a float in kb_id_
    try:
        score = float(ent.kb_id_)
        if 0.0 < score <= 1.0:
            return score
    except (ValueError, TypeError):
        pass
    return _DEFAULT_CONFIDENCE

