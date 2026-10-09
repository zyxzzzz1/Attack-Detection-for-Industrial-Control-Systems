"""标签隔离：src/ 下只有白名单文件可以引用标签模块或出现字符串 "attack"（主线补充决策 2）。"""

import re

from src.utils.paths import ROOT

# 白名单：标签模块本身、最终评估、归因命中率评估。后两个文件在后续阶段创建，创建前不存在也不影响本测试。
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
