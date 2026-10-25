"""Label isolation: under src/ only whitelisted files may reference the label module or contain the string "attack" (main-line decision 2)."""

import re

from src.utils.paths import ROOT

# Whitelist: the label module itself, the final evaluation, the attribution hit-rate evaluation. The latter two are created in later stages; their absence before then does not affect this test.
WHITELIST = {"src/eval/labels.py", "src/evaluate.py", "src/attribution.py"}
LABEL_MODULE_RE = re.compile(r"eval\.labels|eval\s+import\s+[^\n]*\blabels\b|^\s*(from|import)\s+\.?labels\b", re.M)


def violations(root=ROOT) -> list[str]:
    bad = []
    for path in sorted((root / "src").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in WHITELIST:
            continue
        text = path.read_text(encoding="utf-8")
        if "attack" in text.lower():
            bad.append(f'{rel}: 出现字符串 "attack"')
        if LABEL_MODULE_RE.search(text):
            bad.append(f"{rel}: 引用了标签模块")
    return bad


def test_only_whitelisted_files_touch_labels():
    assert violations() == []


def test_scanner_catches_violations(tmp_path):
    (tmp_path / "src" / "data").mkdir(parents=True)
    (tmp_path / "src" / "eval").mkdir()
    (tmp_path / "src" / "eval" / "labels.py").write_text("COL = 'attack'\n", encoding="utf-8")
    (tmp_path / "src" / "data" / "a.py").write_text("from src.eval.labels import read_labels\n", encoding="utf-8")
    (tmp_path / "src" / "data" / "b.py").write_text("from src.eval import metrics, labels\n", encoding="utf-8")
    (tmp_path / "src" / "data" / "c.py").write_text("cols = ['Attack_P1']\n", encoding="utf-8")
    (tmp_path / "src" / "data" / "ok.py").write_text("label_cols = cfg['label_cols']\n", encoding="utf-8")
    bad = violations(tmp_path)
    assert [b.split(":")[0] for b in bad] == ["src/data/a.py", "src/data/b.py", "src/data/c.py"]


def test_whitelist_is_not_wider_than_decided():
    assert WHITELIST == {"src/eval/labels.py", "src/evaluate.py", "src/attribution.py"}
