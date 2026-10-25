"""Display language of the UI and report material (ui.lang in the config: en / zh, default en; the environment variable UI_LANG overrides it).

Affects display text only, never any computation, field name or number. Display text in code and templates is always written with Chinese as the source, as t("中文");
zh returns it unchanged, en looks it up in config/i18n_en.yaml (keys are the Chinese source, values English). A missing English translation raises an error; the tests make sure none is missing.
Text with numbers uses format placeholders (e.g. t("共 {n} 条报警").format(n=3)); the numbers themselves never go into translations.
"""

import os
from functools import lru_cache

import yaml

from src.utils.paths import DEFAULT_CONFIG, ROOT

LANGS = ("en", "zh")
DEFAULT_LANG = "en"
ENV_VAR = "UI_LANG"
CATALOG = ROOT / "config" / "i18n_en.yaml"
DISCLAIMER_ZH = "资产映射与漏洞数据来自实验环境，仅用于演示"

_configured: str | None = None   # In-process default set by configure() (from the config passed by the caller)


@lru_cache(maxsize=1)
def _catalog() -> dict[str, str]:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def _config_lang() -> str | None:
    if not DEFAULT_CONFIG.exists():
        return None
    return (yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")).get("ui") or {}).get("lang")


def _check(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"ui.lang 只能是 {' / '.join(LANGS)}，现在是 {lang!r}")
    return lang


def configure(cfg: dict | None) -> None:
    """Set the default language from a config the caller has already loaded (the environment variable still takes precedence). With cfg None, revert to reading the default config.yaml."""
    global _configured
    _configured = _check((cfg.get("ui") or {}).get("lang") or DEFAULT_LANG) if cfg is not None else None


def get_lang() -> str:
    return _check(os.environ.get(ENV_VAR) or _configured or _config_lang() or DEFAULT_LANG)


def t(text: str, lang: str | None = None) -> str:
    """Display text: text is the Chinese source."""
    lang = _check(lang) if lang else get_lang()
    if lang == "zh":
        return text
    try:
        return _catalog()[text]
    except KeyError:
        raise KeyError(f"config/i18n_en.yaml 缺少译文：{text!r}") from None


def t_data(text, lang: str | None = None):
    """Display text from data files (subsystem names, asset roles, etc.): use the translation if there is one, otherwise return it unchanged."""
    lang = _check(lang) if lang else get_lang()
    if lang == "zh" or not isinstance(text, str):
        return text
    return _catalog().get(text, text)


def disclaimer(lang: str | None = None) -> str:
    return t(DISCLAIMER_ZH, lang)
