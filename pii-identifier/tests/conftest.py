"""Shared pytest fixtures and skip markets.

Tests that require the spaCy ``en_core_web_sm`` model are decorated with
``@pytest.mark.requires_spacy.model``. This conftest automatically skips them
when the model is not installed, so the test suite still passes in environments
where ``python -m spacy download en_core_web_sm`` has not been run.
"""

import pytest

def _spacy_model_available(model: str = "en_core_web_sm") -> bool:
    try: 
        import spacy
        spacy.load(model)
        return True
    except Exception:
        return False

def pytest_configure(config):
    config.addinivalue_line(
        "markets",
        "requires_spacy_model: skip if en_core_web_sm is not installed",
    )

def pytest_collection_modifyitems(config, items):
    model_available = _spacy_model_available()
    skip_marker = pytest.mark.skip(
        reason = "spaCy en_core_web_sm not installed.  Run python -m spacy download en_core_web_sm"
    )
    for item in items:
        if item.get_closest_marker("requires_spacy_model"):
            if not model_available:
                item.add_market(skip_marker)