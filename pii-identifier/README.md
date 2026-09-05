# pii-identifier

`pii-identifier` is a hybrid PII detection pipeline designed for HR and legal documents.  It combines deterministic regex matchers for high-precision structural detection with a spaCy named-entity recognition (NER) layer for contextual entity detection, then merges the results with regex taking priority on overlapping spans.

The library targets the specific mix of PII found in employment documents: Social Security Numbers, phone numbers, email addresses, dates of birth, credit card numbers, person names, addresses, and organization names.  Each detection strategy is optimised for the types of entities it handles best, and the merge step nesures that the two sources complement rather than duplicate each other.  

`pii-identifier` is intended as a development tool and research aid.  The `HRDocumentInjector` component generates realistic, fully synthetic HR documents with known ground-truth PII, making it straight forward to measure precision and recall as you tune models or patterns. 

## Why hybrid?

| Approach | Strength | Weakness |
| -------- | -------- | -------- | 
| Regex only | Near-perfect precision on formatted tokens (SSN, phone, email) | Misses names, addresses, and organization names that have no fixed structure |
| NER only | Catches contextual entities regardless of surface form | Noisy on structured PII; DATE, CARDINAL, and MONEY false-positives are common in HR documents |
| **Hybrid** | High precision on structured PII plus contextual coverage for prose entities | Requires a merge step to reconcile the two result sets |

The merge strategy is deliberatly assymetric.  Regex matches are treated as ground truth: wherever a regex match and an NER match share any overlapping characters, the regex match is kept and the NER match is discarded.  This prevents NER from double-counting entities that are already caught by a deterministic rule - for example, an SSN that spaCy might additionally tag as a CARDINAL, or a phone number it tags as a CARDINAL or QUANTITY.  NER adds recall exclusively for the entity types that have no structural pattern: person names, geographic addresses embedded in prose, and organization names.

--

## Architecture

```
                         Input Text
                              |
            +---------------------------------+
            v                                 v 
    +-----------------+             +--------------------+
        RegexMatcher                      NERMatcher
         (patterns)                        (spaCy)
    +-----------------+             +--------------------+
            | SSN, Phone,                     | PERSON, GPE
            | Email, DOB,                     | LOC, FAC, ORG,
            | Credit Card                     | (PERSON_NAME,
            | confidence = 1.0                | ADDRESS, ORG)
            |                                 | confidence = 0.85
            +-----------------+---------------+
                              v
                    +------------------+
                        Merge Step. <- NER matches that overlap a regex
                                       match are discarded (regex wins) 
                    +------------------+
                              v
                    Filter by min_confidence
                              |
                              v
                    Sorted List (PIIMatch)
```

**RegexMatcher** (`patterns.py`) applies five compiled regular expression - one each for SSN, North-American phone number,s email addresses, date-of-birth labels, and the major credit card networks - to the input text.  All regex matches receive `confidence = 1.0`. Within a single pass, if two patterns produce overlapping spans, the match with the earlier start position is kept (ties broken by preferring the longer match).  Results are returned by start offset.

**NERMatcher** `ner.py` wraps a spaCy pipeline and maps a subset of spaCy entity labels to `PIIType` values.  Only `PERSON`, `GPE`, `LOC`, `FAC`, and `ORG` are forwarded; all other spaCy labels (DATE, CARDINAL, MONEY, PERCENT, TIME, etc.) are silently dropped because in HR documents these produce far too many false positives.  Each NER match receives `confidence = 0.85` by default; if a custom pipeline stores a per-entity score in `ent.kb_id_`, that value is used instead.

**Merge Step** `pipeline.py` iterates over every NER match and discards any that share at least one character with a regex match.  The surviving NER matches aer concatenated with the regex matches, filtered by `min_confidence`, and sorted by start offset.  The merge is one-directional: regex resutls are never suppresed by NER.

**PIIIdentifier** `pipeline.py` is the public entry point.  It owns one `RegexMatcher` and one optional `NERMatcher`, runs the merge internally, and exposes three public methods: `identify()`, `redact()`, and `report()`. 

---

## Prerequisites and Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

The spaCy model download is optional for basic regex-only usage.  Tests that require the model are decorated with `@pytest.mark.requires_spacy_model` and are automatically skipped (with a descriptive message) when the model is not installed, so the test suite still passes in environments where the download has not been run.

For higher NER accuracy - especially on ambiguous names and address fragments - a larger model can be used without changing any other code:

```bash
python -m spacy download en_core_web_lg

# or if GPU resources are available:
python -m spacy download en_core_web_trf
```


---

## Quickstart

```bash
python examples/identify_pii.py
```

The script generates one offer letter, one employee intake form, and one background-check consent form using `HRDocumentInjector`, then runs `PIIIdentifier` on each document and prints a formatted report to the terminal. 

The output for each document contains four sections:

1. **Original text with PII highlighted in red** - ANSI escape codes wrap each detected span so that PII stands out in the terminal.
2. **Detected PII Table** - columns for PII type, matched text (truncated at 28 characters), detection method (`REGEX` in cyan or `NER` in yellow), and confidence score.
3. **Redacted text preview** - The first eight lines of the document with every PII span replaced by `[REDACTED]`.
4. **Precision/Recall vs Ground Truth** - counts of injected vs detected values and precision/recall scores colored green (>=80%) or yellow (<80%). An aggregate summary across all three documents is printed at the end.

---

## Programmatic usage

```python
from pii_identifier import PIIIdentifier

identifier = PIIIdentifier()

text = "Please send your SSN (123-45-6789) to hr@company.com"
report = identifier.report(text)

print(report["pii_types_found"]) # ['email','ssn']
print(report["total_count"]) # 2
print(report["redacted_text"]) # "Please send your SSN ([REDACTED]) to [REDACTED]."

# or just get the redacted string directly:
redacted = identifier.redact(text)

# or iterate over matches:
for match in identifier.identify(text):
    print(match)
```

---

## PII Type Reference

