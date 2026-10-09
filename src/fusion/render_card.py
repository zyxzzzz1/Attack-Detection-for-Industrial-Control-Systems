"""告警卡片（claude.md M10）：enriched_alerts.json → Markdown/HTML 卡片 + 按 priority 排序的索引页。

每张卡片包含：告警 ID、起止时间、时长、分数/阈值、Top-k 传感器及占比、受影响子系统及关键度（含 P4 说明）、
入侵路径、路径资产的漏洞（按 r_v 排序，标 KEV/EPSS）、需人工核实（低 QoD 但命中 KEV）、建议动作、
优先级与排名、is_mock 标记、免责声明。

用法：
    python -m src.fusion.render_card [--input JSON] [--output-dir DIR] [--format md|html|both]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from src.utils import i18n
from src.utils.i18n import disclaimer, t, t_data

ROOT = Path(__file__).resolve().parents[2]
# 以下显示文字以中文为源，渲染时按 ui.lang 经 t() 取译文
STATUS_TEXT = {"scanned": "已扫描", "no_vulns": "已扫描，无漏洞", "not_scanned": "未扫描（未评估）"}
UNDERESTIMATED = "该路径存在未评估资产，实际风险可能被低估"
MAX_ACTION_FIXES = 5
CARD_FILE_RE = re.compile(r"^(\d+_.+|index)\.(md|html)$")  # 本模块生成的文件：NN_告警ID.ext 与 index.ext


def fmt_duration(seconds: int) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    parts = [t("{n} 小时").format(n=h) if h else "", t("{n} 分").format(n=m) if m else "",
             t("{n} 秒").format(n=s) if s or not (h or m) else ""]
    return " ".join(p for p in parts if p)


def md_cell(text) -> str:
    """Markdown 表格单元格：转义竖线、压平换行。"""
    return " ".join(str(text).replace("|", "\\|").split())


def build_actions(alert: dict) -> dict:
    """建议动作：steps 为按顺序排列的动作文本；fixes 为 OpenVAS 修复建议（按 r_v 取前几条，按修复文本去重）。"""
    assets = sorted(alert["path_assets"], key=lambda a: (-(a["R_asset"] or 0.0), a["asset"]))
    sep = t("、")
    check = [t("{asset}（R_asset {risk}）").format(asset=a["asset"], risk=f"{a['R_asset']:.3f}")
             for a in assets if (a["R_asset"] or 0.0) > 0]
    unknown = alert["unassessed_assets"]
    steps = []
    if check:
        steps.append(t("优先排查资产：{assets}").format(assets=sep.join(check)))
    if alert["needs_review"]:
        steps.append(t("人工核实 {n} 条命中 KEV 但 QoD 偏低的漏洞是否真实存在").format(n=len(alert["needs_review"])))
    if unknown:
        steps.append(t("路径上未评估（未扫描）的资产：{assets}（PLC 不做主动扫描，请结合日志与配置核查）").format(assets=sep.join(unknown)))
    if any(s["note"] for s in alert["subsystems"]):
        steps.append(t("按子系统说明排查物理耦合的上游过程"))
    if not steps:
        steps.append(t("受影响子系统无关联资产与漏洞，请结合过程数据人工研判"))

    fixes, seen = [], set()
    for v in alert["vulns"]:
        key = " ".join((v["solution"] or "").split())
        if key and key not in seen:
            seen.add(key)
            fixes.append(v)
        if len(fixes) >= MAX_ACTION_FIXES:
            break
    return {"steps": steps, "fixes": fixes}


def exposure(alert: dict) -> str:
    """告警级 E_sub 的评估状态（只用于显示）：assessed / partial（路径上部分资产未扫描）/
    not_assessed（路径上的资产全部未扫描）。"""
    n_unassessed = len(alert["unassessed_assets"])
    if not n_unassessed:
        return "assessed"
    return "not_assessed" if n_unassessed == len(alert["path_assets"]) else "partial"


def shown_subsystems(alert: dict, digits: int = 0) -> list[dict]:
    """列表里显示的子系统：权重按显示精度（百分比的小数位数）四舍五入后为 0 的不显示。"""
    return [s for s in alert["subsystems"] if round(s["weight"] * 100, digits) > 0]


def card_context(alert: dict, meta: dict) -> dict:
    return {
        "a": alert,
        "meta": meta,
        "n_alerts": len(meta["alerts"]),
        "duration": fmt_duration(alert["duration_s"]),
        "ratio": alert["max_score"] / alert["threshold"],
        "actions": build_actions(alert),
        "status_text": STATUS_TEXT,
        "disclaimer": disclaimer(),   # 按当前显示语言，不取 JSON 中保存的那一句
    }


def card_name(alert: dict, ext: str) -> str:
    return f"{alert['rank']:02d}_{alert['alert_id']}.{ext}"


def make_env(templates_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=("html.j2",), default_for_string=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["md"] = md_cell
    env.filters["pct"] = lambda x: f"{x * 100:.1f}%"
    env.filters["f3"] = lambda x: f"{x:.3f}"
    # 资产风险：未评估（None）不显示成 0
    env.filters["risk"] = lambda x: t("未评估") if x is None else f"{x:.3f}"
    env.filters["trd"] = t_data          # 数据文件中的显示文字（子系统名称、说明、mock 场景）
    env.globals["_"] = t                 # 模板中的显示文字：{{ _("中文") }}
    env.globals["html_lang"] = lambda: {"zh": "zh-CN"}.get(i18n.get_lang(), i18n.get_lang())
    env.globals["UNDERESTIMATED"] = UNDERESTIMATED
    env.globals["card_name"] = card_name
    env.globals["exposure"] = exposure
    env.globals["shown_subsystems"] = shown_subsystems
    return env


def render_all(meta: dict, env: Environment, out_dir: Path, formats: list[str], limit: int | None = None) -> list[Path]:
    """limit：只为优先级最高的前 limit 条报警生成卡片并列入索引页（None 或 0 表示全部）。
    卡片上的"第 x / n"仍按全部报警计数。输出目录里上一次生成的卡片和索引页先清掉，避免新旧混在一起。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_file() and CARD_FILE_RE.match(old.name):
            old.unlink()
    written = []
    alerts = sorted(meta["alerts"], key=lambda a: a["rank"])[:limit or None]
    for ext in formats:
        card_t = env.get_template(f"alert_card.{ext}.j2")
        for a in alerts:
            path = out_dir / card_name(a, ext)
            path.write_text(card_t.render(**card_context(a, meta)), encoding="utf-8")
            written.append(path)
        index = out_dir / f"index.{ext}"
        index.write_text(env.get_template(f"index.{ext}.j2").render(
            meta=meta, alerts=alerts, n_total=len(meta["alerts"]), disclaimer=disclaimer()), encoding="utf-8")
        written.append(index)
    return written


def main() -> int:
    p = argparse.ArgumentParser(description="enriched_alerts.json → 告警卡片 + 索引页")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--input", type=Path, help="默认取 config.yaml 的 fusion.enriched_alerts_json")
    p.add_argument("--output-dir", type=Path, help="默认取 config.yaml 的 fusion.cards_dir")
    p.add_argument("--format", choices=["md", "html", "both"], default="both")
    p.add_argument("--limit", type=int, help="只渲染优先级最高的前 N 条，默认取 config.yaml 的 fusion.max_rendered_alerts")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["fusion"]
    meta = json.loads((args.input or ROOT / cfg["enriched_alerts_json"]).read_text(encoding="utf-8"))
    formats = ["md", "html"] if args.format == "both" else [args.format]
    env = make_env(ROOT / cfg["templates_dir"])
    limit = cfg.get("max_rendered_alerts") if args.limit is None else args.limit
    written = render_all(meta, env, args.output_dir or ROOT / cfg["cards_dir"], formats, limit)
    print(f"[保存] {len(written)} 个文件 → {written[0].parent}（is_mock={meta.get('is_mock')}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
