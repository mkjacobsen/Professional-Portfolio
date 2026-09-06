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

All eight `PIIType` enum members are listed below with their detection method, the underlying patter or spaCy label, and a representative match.

| PIIType | Enum Value | Detection Method | Pattern / Entity Label | Example Match |
| ------- | ---------- | ---------------- | ---------------------- | ------------- |
| `SSN` | `"ssn"` | Regex | `\b\d{3}-\d{2}-\d{4}\b` | `123-45-6789` |
| `PHONE` | `"phone"` | Regex | Optional `+1` country code, otpional parentheses around area code, digits separated by `-`, `.`, or space  | `(555) 123-4567`, `+1 800.555.1234` |
| `EMAIL` | `"email"` | Regex | RFC-5321-compatible: local part `[A-Za-z0-9._%+\-]+, `@`, domain with TLD of two or more characters | `user@example.com` |
| `DATE_OF_BIRTH` | `"date_of_birth"` | Regex | `DOB:?` followed by MM/DD/YYYY` or `M-D-YY`, **or** `date of birth:?` (case-insensitive) followed by `Month DD, YYYY` | `DOB: 01/15/1980`, `Date of Birth: January 15, 1980` |
| `CREDIT_CARD` | `"credit_card"` | Regex | Visa ('4xxx'), Mastercard ('51-55xx'), Discover ('6011'), or Amex ('34xx'/'37xx') prefix, followed by three groups of four digits with optional spaces or hyphens | `4111-1111-1111-1111` | 
| `PERSON_NAME` | `"person_name"` | NER | spaCy `PERSON` | `Jane Doe` |
| `ADDRESS` | `"address"` | NER | spaCy `GPE` (cities, states, countries), `LOC` (locations, bodies of water), `FAC` (buildings, airports) | `Washington, DC` |
| `ORGANIZATION` | `"organization"` | NER | spaCy `ORG` | `Acme Corp` |

Note: The credit card pattern targets 16-digit card numbers.  American Express uses 15 digits and is partially matched by the `34xx`/`37xx` prefix check, but full 15-digit Amex support would require a separate pattern variant.

---

## API Reference: PIIIdentifier

```python
class PIIIdentifier:
    def __init__(
        self,
        user_ner: bool = True,
        ner_model: str = "en_core_web_sm",
        min_confidence: float = 0.7,
    ) -> None: ...
    ...
```


### Constructor Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- | 
| `use_ner` | `bool` | `True` | Set to `False` to run regex patterns only.  Useful in environments where spaCy is not available or when throughput matters more than recall of contextual entities. |
| `ner_model` | `str` | `"en_core_web_sm"` | spaCy model name forwarded to `NERMatcher`.  Swap to `..._lg` or `..._trf` for higher NER accuracy.  Has no effect when `use_ner=False`. |
| `min_confidence` | `float` | `0.7` | Matches with `confidence < min_confidence` are removed before results are returned.  Because all regex matches have `confidence=1.0` this threshold only affects NER matches (default to `0.85`). Set to `0.9` or higher to suppress lower-confidence NER detections; set to `0.0` to disable filtering entirely. |

### Public Methods

#### `identify(text: str) -> list[PIIMatch]`

Detects all PII in `text` and returns a deduplicated list of `PIIMatch` objects sorted by `start` offset.  Internally runs the regex matcher, optionally runs the NER matcher, merges the two results sets (see section 10), filters by `min_confidence`, and sorts.

#### `redact(text: str, replacement: str = "[REDACTED"]) -> str`

Returns a copy of `text` with every detected PII span replaced by `replacement`. The replacement string defaults to `"[REDACTED]"` but any string may be supplied.  Matches are processed right-to-lef by start offset so that earlier character positions remain valid after each substitution.

#### `report(text: str) -> dict`

Runs `identify()` and `redact()` and returns a single dict with the following keys:

| Key | Type | Description |
| --- | ---- | ----------- |
| `matches` | `list[PIIMatch]` | Full list of `PIIMatch` objects, sorted by start index. |
| `pii_types_found` | `list[str]` | Sorted, deduplicated list of `PIIType` vaule strings (e.g. `["email", "ssn"]`) |
| `total_count` | `int` | Total number of distinct PII matches (`len(matches)`) |
| `redacted_text` | `str` | The original text with all PII spans replaced by the value of redacted parameter |

---

## API reference: HRDocumentInjector

```python
class HRDocumentInjector:
    def __init__(self, seed: int = 42) -> None:
```

`HRDocumentInjector` generates realistic synthetic HR documents with deterministic, seeded PII for testing and benchmarking. It uses the [Faker ](https://faker.readthedocs.io/) library for plausible names,addresses, and other values. Both `Faker.seed()` and Python's `random.seed()` are called with seed, ensuring reproducible output across runs.

Each generator method returns a (str, list[dict]) tuple where the first element is the document text and the second is the ground-truth list. Each ground-truth entry has the shape:

```python
{"text": "Jane Doe", "pii_type": "person_name", "start": 42,"end": 50}
...
```

Offsets are computed by building the document as a sequence of tagged segments rather than by searching for values after the fact, so offsets are exact even when the same value appears more than once.

### Document generators

#### `offer_letter() -> tuple[str, list[dict]]`

Generates an approximately 300-word employment offer letter. Injects: PERSON_NAME" (candidate), 'ADDRESS' (mailing address), "SSN', "EMAIL'
(HR contact), PHONE" (HR contact).


#### `intake_form() -> tuple[str, list[dict]]`

Generates an approximately 200-word employee intake / new-hire information form. Injects: PERSON_NAME" (employee and emergency contact - two entries), DATE_OF_BIRTH, PHONE (employee and emergency contact - two entries), "EMAIL'.

#### `background_check_consent() -> tuple[str, list[dict]]`

Generates an approximately 250-word background-check authorization and consent form. Injects: PERSON_NAME' (two entries - opening sentence and applicant information section), "SSN', "DATE_OF_BIRTH, "ADDRESS",  "ORGANIZATION" (previous employer).

#### `generate_batch(n: int = 10) -> list[tuple[str, list[dict]]]`

Returns in documents of randomly chosen types (offer letter, intake form, or background-check consent) with injected PII. Useful for bulk evaluation.

## Merge algorithm

### Overlap definition

Two spans overlap when they share at least one character:

```
max(start_a, start_b) < min(end_a, end_b)
```

Adjacent spans where one ends exactly where the other begins end (start_b) do not overlap by this definition. This is consistent with Python's half-open slice convention.

### Cross-layer resolution (regex vs NER)

After both matchers have run, `_remove_overlapping_ner()` in `pipeline.py` iterates over every NER match and discards it if it overlaps with any regex match. Regex matches are never suppressed. This is a one-directional filter: the regex layer has veto power over the NER layer for any shared region of text. The motivation is that a valid structural pattern match (e.g., an SSN `d(3)-ld(2)-\d(4)') is a far more reliable signal than an NER label, and allowing both to survive would double-count the same PII value.

### Within-layer resolution (regex vs regex)

Within the regex pass, RegexMatcher.match () collects all raw matches from all five patterns, then sorts them by (start, -length) before passing them to `_resolve_overlaps()`. That function walks the sorted list and, whenever a candidate match's start offset falls before the end of the previously accepted match, skips the candidate. The effect is: the match with the earlier start position wins; when two matches start at the same position, the longer one wins. Only one match survives per overlapping group.

### Within-layer resolution (NER vs NER)

spaCy's tokeniser enforces non-overlapping entity spans internally, so the NER layer never produces two overlapping matches. No additional deduplication is needed for NER-only matches.

### Redaction Offset Stability

`redact()` calls `identify()` and then iterates over the matches in reverse start-offset order (largest start first).  Each replacement of `text[match.start:match.end]` with the replacement string changes the length of `text`, but only affects character positions to the right of `match.end`. Because subsequent loop iterations always target positions further left in the string, all remaining start and end offsets remain correct.

---

## Extending the Pipeline

### Adding a new regex pattern

Open `pii_identifier/patterns.py` and append a new compiled pattern to `_PATTERNS`. If the PII type does not already exist in `PIIType`, add it to `pii_identifier/types.py` first.

```python
# In types.py - add to PIIType enum:
PASSPORT = "passport"

# In patterns.py - add compiled pattern:
import re
from .types import PIIType

_PASSPORT_PATTERN = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")

# Append to _PATTERNS list:
_PATTERNS: list[typle[PIIType, re.Pattern]] = [
    ...
    (PIIType.PASSPORT, _PASSPORT_PATTERN),
]
```

The new pattern is picked up automatically by `RegexMatcher` on the next instantiation - no other changes are needed.

### Adding a new NER label mapping

Open src/pii identifier/ner.py and add the spacy label and its corresponding `PIIType` to `_LABEL_MAP`. For example, to map the spacy `NORP` label (nationalities, religious groups, political groups) to a new PIlType:

```python
# In types py - add to PIIType enum:
DEMOGRAPHIC = "demographic"

# In ner.py - add to _LABEL_MAP:
_LABEL_MAP: dict[str, PIIType] = [
    "PERSON": PIIType. PERSON_NAME,
    "GPE": PIIType. ADDRESS,
    "LOC": PIIType.ADDRESS,
    "FAC": PIIType. ADDRESS,
    "ORG": PIIType. ORGANIZATION,
    "NORP": PIIType. DEMOGRAPHIC, # new entry
]
```

### Swapping in a larger spaCy model

```python
from pii identifier import PIIIdentifier

#Higher recall for names and addresses at the cost of load time:
identifier = PIIIdentifier(ner_model="en_core_web_lg")
```

Download the model first:

```bash
python -m spacy download en_core_web_lg
```

The transformer-based 'en_core web_trf` model gives the best NER accuracy if GPU resources are available. No code changes are required beyond passing the model name.

### Disabling the NER layer

Pass 'use_ner=False" to run the rege layer only. This is useful when spacy is not installed, in latency-sensitive environments, or when you want to measure the contribution of each layer independently.

```python
identifier = PIlIdentifier (use_ner=False)
```

### Subclassing for custom merge logic

`PIIIdentifier` can be subclassed to override the merge step. Override `identify()` and call `self.regex_matcher.match()` and `self.ner_matcher.match()` directly to implement alternative conflict-resolution strategies.

# Running tests
```bash
pip install -e "[dev]"
pytest tests/ -v
```
Tests that require `en_core_web_sm` carry the `@pytest.marker.requires_spacy_model` marker. If the model is absent, those tests are automatically skipped with the message:

SKIPPED - spacy en_core_web_sm not installed. Run: Python -m spacy download en_core_web_sm. 

All other tests run without the model. The test suite is organised as follows:

| File | What it covers |
| `tests/test_patterns.py` | `RegexMatcher`: one class per PII type (SSN, PHONE, EMAIL, DATE_OF_BIRTH, CREDIT_CARD), plus overlap resolution and sort-order guarantees. No spaCy required. |
| `tests/test_ner.py` | `NERMatcher`: entity label mapping, confidence range, method tag, and suppression of noisy spacy labels (DATE, MONEY). Requires en_core_web_sm`.|
| `tests/test_pipeline.py` | `PIIIdentifier`: `identify()`, `redact()`, `report()`, `min_confidence` filtering, regex-beats-NER overlap resolution, and the `_spans_overlap` helper. Most tests run without spaCy via the 'use_ner=False` fixture; overlap-resolution tests require the model. |
| `tests/test_injector.py` | `HRDocumentInjector`: return types, ground-truth offset accuracy, expected PII types per document type, and batch generation. No spacy required. |

## Known Limitations

- **spaCy model precision** - `en_core_web_sm` is a small model trained for general English texxt.  It has lower precision than the larger or transformer-based models and will produce false positives for `PERSON_NAME` and `ORGANIZATION` on common words or phrases that happen to look like proper nouns in context.  Expect elevated false-positive rates on short or unusually formatted documents.

- **ADDRESS detection is coarse** - The NER layer maps `GPE`, `LOC`, and `FAC` labels to `ADDRESS`.  This means city names, state abbreviations, and country names are flagged, but full street addresses written in prose (e.g. "123 Mainstreet, Suite 4") may not be detected if spaCy does not segment them as a single entity.  Conversely, any city or country name - including those used non-personally (e.g., "the Washington summit") - will be flagged.

- **DATE_OF_BIRTH requires a label prefix** - The DOB regex only fires when the date is preceded by `DOB:`, `DOB`, or `Date of Birth:` (case -insensitive).  Bare date strings, such as `01/15/1980` appearing without a label are not detected.  This is intentional - unlabelled dates in HR documents are extremely common as non-PII (hire dates, review dates, effective dates) and matching them would produce a large number of false positives.

- **English only** - All regex patterns and the spaCy model are tuned for English-language documents.  Non-English names, addresses formatter per non-US conventions, and non-US phone number formats are not reliably detected.

- **Not a compliance tool** - This library is a development and research aid.  It should not be used as the sole PII detection mechanism for regulatory compliance purposes (GDPR, CCPA, HIPPA, etc.).  Production compliance use cases require audited, validated tooling, legal reivew, and ongoing model monitoring.