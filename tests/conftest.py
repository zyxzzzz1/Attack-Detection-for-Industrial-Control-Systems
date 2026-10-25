"""Shared test setup."""

import pytest

from src.utils import i18n


@pytest.fixture(autouse=True)
def ui_language_zh(monkeypatch):
    """The display language is fixed to Chinese (the source) in tests: existing tests assert Chinese display text.
    English (the config default) is covered specifically by tests/test_i18n.py, whose tests set UI_LANG to en or remove it themselves."""
    monkeypatch.setenv(i18n.ENV_VAR, "zh")
    i18n.configure(None)
    yield
    i18n.configure(None)
