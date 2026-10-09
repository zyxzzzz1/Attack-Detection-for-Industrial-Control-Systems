"""测试的公共设置。"""

import pytest

from src.utils import i18n


@pytest.fixture(autouse=True)
def ui_language_zh(monkeypatch):
    """显示语言在测试中固定为中文（源文）：已有测试断言的是中文显示文字。
    英文（config 的默认值）由 tests/test_i18n.py 专门覆盖，那里的测试自行把 UI_LANG 设为 en 或删掉。"""
    monkeypatch.setenv(i18n.ENV_VAR, "zh")
    i18n.configure(None)
    yield
    i18n.configure(None)
