"""列出代码和模板里全部需要翻译的显示文字（中文源文），用于检查 config/i18n_en.yaml 是否有遗漏或多余。

- Python：t("...") 调用的第一个参数（字符串字面量，含相邻字面量拼接）；
- 模板：{{ _("...") }}；
- 间接传给 t() 的常量（先存进变量或表里的源文）在 INDIRECT 中登记；
- 数据文件里的显示文字（资产映射中的名称、说明、角色，mock 报警的场景说明）经 t_data() 翻译，单独列出。

用法：
    python -m scripts.i18n_keys            # 打印缺少和多余的键
"""

import ast
import json
import re
import sys

import yaml

from src.utils.i18n import CATALOG, DISCLAIMER_ZH
from src.utils.paths import ROOT

PY_DIRS = ("src", "scripts", "app")
TEMPLATE_RE = re.compile(r'_\("((?:[^"\\]|\\.)*)"\)')


def indirect_keys() -> set[str]:
    from scripts.make_ablation_report import HEATMAP_METRICS
    from scripts.make_cases import CASES
    from src.fusion.render_card import STATUS_TEXT, UNDERESTIMATED
    keys = {DISCLAIMER_ZH, UNDERESTIMATED, *STATUS_TEXT.values(), *(title for _, title in HEATMAP_METRICS),
            *(title for title, _ in CASES), "真实报警（LSTM-AE）", "mock 报警",
            "已评估", "可能低估", "未评估：上游资产均未扫描"}   # app/streamlit_app.py 的 PATH_RISK_TEXT
    return keys


def python_keys() -> set[str]:
    keys = set()
    for d in PY_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t" and node.args
                        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    keys.add(node.args[0].value)
    return keys


def template_keys() -> set[str]:
    keys = set()
    for path in (ROOT / "templates").glob("*.j2"):
        for m in TEMPLATE_RE.finditer(path.read_text(encoding="utf-8")):
            keys.add(m.group(1).encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
                     if "\\" in m.group(1) else m.group(1))
    return keys


def data_keys() -> set[str]:
    """数据文件里含中文的显示文字。"""
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["fusion"]
    amap = yaml.safe_load((ROOT / cfg["asset_map"]).read_text(encoding="utf-8"))
    texts = [s.get(k) for s in amap["subsystems"].values() for k in ("name", "note")]
    texts += [a.get("role") for a in amap["assets"].values()]
    texts += [a.get("mock_case") for a in json.loads((ROOT / cfg["mock_alerts_json"]).read_text(encoding="utf-8"))]
    return {x for x in texts if isinstance(x, str) and re.search(r"[一-鿿]", x)}


def required_keys() -> set[str]:
    return python_keys() | template_keys() | indirect_keys() | data_keys()


def main() -> int:
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8")) if CATALOG.exists() else {}
    need = required_keys()
    missing, extra = sorted(need - set(catalog)), sorted(set(catalog) - need)
    for k in missing:
        print(f"[缺少] {k!r}")
    for k in extra:
        print(f"[多余] {k!r}")
    print(f"需要 {len(need)} 条，译文表 {len(catalog)} 条，缺少 {len(missing)}，多余 {len(extra)}")
    return 1 if missing or extra else 0


if __name__ == "__main__":
    sys.exit(main())
