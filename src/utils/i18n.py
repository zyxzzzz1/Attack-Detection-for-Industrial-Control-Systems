"""界面与报告素材的显示语言（config 的 ui.lang：en / zh，默认 en；环境变量 UI_LANG 可覆盖）。

只影响显示文字，不影响任何计算、字段名和数字。代码和模板里的显示文字一律以中文为源写成 t("中文")；
zh 原样返回，en 查 config/i18n_en.yaml（键为中文源文，值为英文）。英文缺译时直接报错，由测试保证不会漏。
带数字的文字用 format 占位符（如 t("共 {n} 条报警").format(n=3)），数字本身不进译文。
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

_configured: str | None = None   # configure() 设定的进程内默认值（来自调用方传入的 config）


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
    """用调用方已经读入的 config 设定默认语言（环境变量仍然优先）。cfg 为 None 时恢复为读默认的 config.yaml。"""
    global _configured
    _configured = _check((cfg.get("ui") or {}).get("lang") or DEFAULT_LANG) if cfg is not None else None


def get_lang() -> str:
    return _check(os.environ.get(ENV_VAR) or _configured or _config_lang() or DEFAULT_LANG)


def t(text: str, lang: str | None = None) -> str:
    """显示文字：text 是中文源文。"""
    lang = _check(lang) if lang else get_lang()
    if lang == "zh":
        return text
    try:
        return _catalog()[text]
    except KeyError:
        raise KeyError(f"config/i18n_en.yaml 缺少译文：{text!r}") from None


def t_data(text, lang: str | None = None):
    """来自数据文件的显示文字（子系统名称、资产角色等）：有译文就用，没有就原样返回。"""
    lang = _check(lang) if lang else get_lang()
    if lang == "zh" or not isinstance(text, str):
        return text
    return _catalog().get(text, text)


def disclaimer(lang: str | None = None) -> str:
    return t(DISCLAIMER_ZH, lang)
