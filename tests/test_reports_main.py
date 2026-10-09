"""M3 / M4 / M5 / I1 的脚本：归因规则（手工样例）、alerts.json 契约、报告来源规则，以及在合成数据的冒烟产物上跑通全部脚本。"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field

from scripts import export_alerts, make_ablation_report, make_figures, make_main_report
from src import attribution, run_model
from src.eval.alarms import alarm_segments, sensor_shares, subsystem_shares, top_k
from src.fusion.prioritize import Alert
from src.utils.paths import ROOT, load_config
from src.utils.reporting import ReportSource, report_problem, smoke_problem
from src.utils.run import ablation_grid

FEATURES = ["P1_A", "P2_B", "P4_C"]


# ---------- 报警段与归因规则（手工样例） ----------

def hand_case():
    n = 50
    scores = np.zeros(n)
    scores[:2] = np.nan
    err = np.full((n, 3), 0.01)
    err[:2] = np.nan
    alarms = {(6, 7): [8.0, 1.0, 1.0],      # 与事件 1（P1）重叠，Top-1 = P1_A → 命中
              (21, 22): [1.0, 5.0, 1.0],    # 只与事件 2（无子系统标签）重叠 → 单列
              (30, 31): [1.0, 1.0, 3.0],    # 不与任何事件重叠 → 单列
              (41, 43): [1.0, 2.0, 7.0]}    # 与事件 3（P2）重叠，Top-1 = P4_C → 未命中，且计入 P4
    for (a, b), e in alarms.items():
        scores[a:b + 1] = 2.0
        err[a:b + 1] = e
    scores[42] = 9.0
    index = {"files": [{"file": "test1.csv.gz", "offset": 0, "rows": n, "start_row": 0, "start_time": "2020-07-07 15:00:00"}],
             "features": FEATURES, "window": 3}
    events = [{"event_id": 1, "part": 0, "file": "test1.csv.gz", "start": 5, "end": 9, "subsystems": ["P1"]},
              {"event_id": 2, "part": 0, "file": "test1.csv.gz", "start": 20, "end": 24, "subsystems": []},
              {"event_id": 3, "part": 0, "file": "test1.csv.gz", "start": 40, "end": 44, "subsystems": ["P2"]}]
    return scores, err, index, events


def test_alarm_segments_and_topk():
    scores, err, index, _ = hand_case()
    segs = alarm_segments(scores, index, 1.0)
    assert [(s["start"], s["end"], s["duration_s"]) for s in segs] == [(6, 7, 2), (21, 22, 2), (30, 31, 2), (41, 43, 3)]
    assert segs[0]["start_time"] == "2020-07-07 15:00:06" and segs[3]["end_time"] == "2020-07-07 15:00:43"
    assert segs[3]["max_score"] == 9.0
    shares = sensor_shares(err, 6, 7)
    assert shares.tolist() == pytest.approx([0.8, 0.1, 0.1]) and shares.sum() == pytest.approx(1)
    assert top_k(shares, FEATURES, 2) == [{"sensor": "P1_A", "share": pytest.approx(0.8)}, {"sensor": "P2_B", "share": pytest.approx(0.1)}]
    assert subsystem_shares(shares, FEATURES) == {"P1": pytest.approx(0.8), "P2": pytest.approx(0.1), "P4": pytest.approx(0.1)}
    assert np.isnan(sensor_shares(err, 0, 1)).all() and top_k(sensor_shares(err, 0, 1), FEATURES, 3) == []
    assert alarm_segments(scores, index, 1.0, merge_gap_s=8)[1]["end"] == 31  # 合并间隔来自同一个 to_segments


def test_attribution_rules_hand_example():
    scores, err, index, events = hand_case()
    rows, summary, event_rows = attribution.attribute_run(scores, err, index, 1.0, 0, events, [3, 5])
    assert [r["category"] for r in rows] == ["evaluable", "overlap_no_subsystem_label", "non_overlap", "evaluable"]
    assert [r["hit"] for r in rows] == [True, None, None, False]
    assert [r["top1_subsystem"] for r in rows] == ["P1", "P2", "P4", "P4"]
    assert rows[0]["event_ids"] == "1" and rows[2]["event_ids"] == "" and rows[3]["event_subsystems"] == "P2"
    assert [t["sensor"] for t in json.loads(rows[3]["top3"])] == ["P4_C", "P2_B", "P1_A"] and len(json.loads(rows[0]["top5"])) == 3
    assert [r["hit_union"] for r in rows] == [True, None, None, False] and [r["n_events_overlap"] for r in rows] == [1, 1, 0, 1]
    assert summary == {"n_alarm_segments": 4, "n_overlap_attack": 3, "n_evaluable": 2, "n_hit": 1, "subsystem_hit_rate": 0.5,
                       "n_multi_event_segments": 0, "n_evaluable_union": 2, "n_hit_union": 1, "subsystem_hit_rate_union": 0.5,
                       "n_top1_p4": 1, "top1_p4_ratio": pytest.approx(1 / 3), "n_overlap_no_subsystem_label": 1,
                       "n_non_overlap": 1, "n_events": 3, "n_events_without_subsystem": 1, "n_events_attributed": 3,
                       "event_hit_rate": 0.5}
    assert [(e["event_id"], e["hit"], e["alarm_seconds"]) for e in event_rows] == [(1, True, 2), (2, None, 2), (3, False, 3)]
    assert event_rows[0]["share_P1"] == pytest.approx(0.8)


def test_attribution_segment_overlapping_several_events_uses_longest_overlap():
    scores, err, index, events = hand_case()
    scores[6:42] = 2.0  # 一段长报警（6~43）跨过三个事件：与事件 1 重叠 4 秒，与事件 2 重叠 5 秒，与事件 3 重叠 4 秒
    err[6:42] = [1.0, 6.0, 1.0]
    rows, summary, _ = attribution.attribute_run(scores, err, index, 1.0, 0, events, [3])
    assert len(rows) == 1 and rows[0]["event_ids"] == "1+2+3" and rows[0]["n_events_overlap"] == 3
    # 主口径：重叠最长的是事件 2（没有子系统标签）→ 不计入命中率；并集口径：P1+P2，Top-1 = P2_B → 命中
    assert rows[0]["primary_event_id"] == 2 and rows[0]["category"] == "overlap_no_subsystem_label" and rows[0]["hit"] is None
    assert rows[0]["union_subsystems"] == "P1+P2" and rows[0]["hit_union"] is True
    assert (summary["n_multi_event_segments"], summary["n_evaluable"], summary["n_evaluable_union"], summary["n_hit_union"]) == (1, 0, 1, 1)

    # 去掉事件 2：事件 1 与事件 3 的重叠时间相同（各 4 秒）→ 取较早的事件 1（P1），Top-1 在 P2 → 主口径未命中，并集口径命中
    rows, summary, _ = attribution.attribute_run(scores, err, index, 1.0, 0, [events[0], events[2]], [3])
    assert rows[0]["primary_event_id"] == 1 and rows[0]["event_subsystems"] == "P1" and rows[0]["hit"] is False
    assert rows[0]["hit_union"] is True and (summary["n_hit"], summary["n_hit_union"]) == (0, 1)
    # 事件 3 延长后重叠更长 → 改按事件 3（P2）判断，主口径命中
    longer = [events[0], {**events[2], "start": 36}]
    rows, _, _ = attribution.attribute_run(scores, err, index, 1.0, 0, longer, [3])
    assert rows[0]["primary_event_id"] == 3 and rows[0]["hit"] is True


def test_attribution_no_alarms_and_no_events():
    scores, err, index, events = hand_case()
    _, summary, event_rows = attribution.attribute_run(scores, err, index, 100.0, 0, events, [3])
    assert summary["n_alarm_segments"] == 0 and np.isnan(summary["subsystem_hit_rate"]) and event_rows == []
    rows, summary, _ = attribution.attribute_run(scores, err, index, 1.0, 0, [], [3])
    assert {r["category"] for r in rows} == {"non_overlap"} and summary["n_non_overlap"] == 4 and np.isnan(summary["top1_p4_ratio"])


def test_label_events_use_union_of_subsystem_labels():
    df = pd.DataFrame({"time": range(12), "attack": [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                       "attack_P1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "attack_P2": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       "attack_P3": [0] * 12})
    index = {"files": [{"file": "f", "offset": 0, "rows": 9, "start_row": 2, "start_time": "2020-01-01"}]}
    events = attribution.label_events(index, {"f": df}, ["attack", "attack_P1", "attack_P2", "attack_P3"])
    # 片段从第 2 行开始：第一个事件只剩后两秒（只有 P2 的标签），第二个事件没有任何子系统标签
    assert [(e["start"], e["end"], e["subsystems"]) for e in events] == [(0, 1, ["P2"]), (4, 5, []), (8, 8, [])]


# ---------- 报告来源规则 ----------

def write_manifest(d, **fields):
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_manifest.json").write_text(json.dumps({"smoke": False, "git_dirty": False, "status": "completed", **fields}))


def test_report_and_smoke_rules(tmp_path):
    for name, fields in {"ok": {}, "smoke": {"smoke": True}, "dirty": {"git_dirty": True}, "unknown": {"git_dirty": None},
                         "failed": {"status": "failed"}, "local": {"local_diagnostic": True}}.items():
        write_manifest(tmp_path / name, **fields)
    assert report_problem(tmp_path / "ok") is None and smoke_problem(tmp_path / "smoke") is None
    for name in ("smoke", "dirty", "unknown", "failed", "local", "missing"):
        assert report_problem(tmp_path / name), name
    assert smoke_problem(tmp_path / "ok") and smoke_problem(tmp_path / "missing")
    assert report_problem(tmp_path / "failed", ("completed", "failed")) is None


def test_report_source_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "art"))
    cfg = load_config()
    cfg["ml"]["report_dir"] = str(tmp_path / "report")
    smoke, official = ReportSource(cfg, True), ReportSource(cfg, False)
    assert (smoke.root, smoke.out) == (tmp_path / "art" / "smoke", tmp_path / "art" / "smoke_reports")
    assert (official.root, official.out) == (tmp_path / "art", tmp_path / "report")


# ---------- alerts.json 契约 ----------

class ContractTopK(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensor: str
    share: float = Field(ge=0, le=1)


class ContractAlert(BaseModel):
    """alerts.json 的字段契约：mock 与真实输出都必须通过。mock 多一个 mock_case，真实多一个 model，其余字段完全相同。"""
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())
    alert_id: str = Field(min_length=1)
    start: str = Field(pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    end: str = Field(pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    max_score: float = Field(ge=0)
    threshold: float = Field(gt=0)
    topk: list[ContractTopK] = Field(min_length=1)
    is_mock: bool
    mock_case: str | None = None
    model: str | None = None


def check_contract(alerts: list[dict], is_mock: bool, sensors: set[str]) -> None:
    assert alerts and len({a["alert_id"] for a in alerts}) == len(alerts)
    for a in alerts:
        parsed = ContractAlert.model_validate(a)
        Alert.model_validate(a)  # 扩展部分实际使用的解析模型
        assert parsed.is_mock is is_mock and pd.Timestamp(parsed.start) <= pd.Timestamp(parsed.end)
        assert (parsed.model is None) == is_mock and (parsed.mock_case is None) != is_mock
        shares = [t.share for t in parsed.topk]
        # 每个 share 四舍五入到 6 位小数，各自最多偏 0.5e-6，所以和的上限按个数放宽
        assert shares == sorted(shares, reverse=True) and sum(shares) <= 1 + 0.5e-6 * len(shares) + 1e-9
        assert {t.sensor for t in parsed.topk} <= sensors and len({t.sensor for t in parsed.topk}) == len(shares)


def hai_sensors() -> set[str]:
    cols = (ROOT / load_config()["fusion"]["hai_columns_txt"]).read_text(encoding="utf-8").split()
    return {c for c in cols if c.startswith("P")}


def test_mock_alerts_match_contract():
    alerts = json.loads((ROOT / "data" / "mock" / "alerts.json").read_text(encoding="utf-8"))
    check_contract(alerts, True, hai_sensors())


def test_real_alert_fields_equal_mock_fields_except_marker():
    mock = json.loads((ROOT / "data" / "mock" / "alerts.json").read_text(encoding="utf-8"))[0]
    scores, err, index, _ = hand_case()
    run = type("Run", (), {"scores": scores, "per_sensor_err": err, "index": index, "features": FEATURES,
                           "threshold": {"threshold": 1.0}})()
    alerts = export_alerts.build_alerts(run, "lstm_ae", 0, 5)
    assert set(alerts[0]) - {"model"} == set(mock) - {"mock_case"}
    assert set(alerts[0]["topk"][0]) == set(mock["topk"][0])
    assert [a["alert_id"] for a in alerts] == ["LSTMAE-00001", "LSTMAE-00002", "LSTMAE-00003", "LSTMAE-00004"]
    assert alerts[3] == {"alert_id": "LSTMAE-00004", "start": "2020-07-07 15:00:41", "end": "2020-07-07 15:00:43",
                         "max_score": 9.0, "threshold": 1.0, "is_mock": False, "model": "lstm_ae",
                         "topk": [{"sensor": "P4_C", "share": 0.7}, {"sensor": "P2_B", "share": 0.2}, {"sensor": "P1_A", "share": 0.1}]}
    check_contract(alerts, False, set(FEATURES))


# ---------- 在合成数据的冒烟产物上跑通全部脚本 ----------

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """合成 HAI 数据 → 冒烟运行（pca、lstm_ae 和 2 个消融组合）→ 一份把报告目录指到临时目录的 config。"""
    root = tmp_path_factory.mktemp("ws")
    cfg = load_config()
    hai = cfg["ml"]["hai"]
    rng = np.random.default_rng(0)
    n = 2000
    (root / "hai").mkdir()
    for k, name in enumerate(hai["train_files"] + hai["test_files"]):
        t = np.arange(n)[:, None]
        x = np.sin(t / 9 + np.arange(4)) + 0.1 * rng.normal(size=(n, 4))
        df = pd.DataFrame(x, columns=["P1_A", "P1_B", "P2_A", "P4_A"])
        df.insert(0, "time", [str(s) for s in pd.date_range(f"2020-03-{k + 1:02d}", periods=n, freq="s")])
        for c in hai["label_cols"]:
            df[c] = 0
        if name == "test1.csv.gz":
            df.loc[1200:1259, ["P1_A", "P1_B"]] += 6.0
            df.loc[1200:1259, ["attack", "attack_P1"]] = 1
        df.to_csv(root / "hai" / name, index=False)
    cfg["ml"]["report_dir"] = str(root / "report")
    cfg["ml"]["alerts"]["output"] = str(root / "official_alerts.json")
    cfg["ml"]["window"]["length"] = 12
    cfg["ml"]["data"]["drop_train_overlap"] = False
    cfg["ml"]["smoke"].update(fraction=0.5, extra_test_clips=[{"file": "test1.csv.gz", "start_row": 1100, "rows": 300}])
    cfg["ml"]["ablation"].update(windows=[12, 20], hiddens=[4, 8])
    cfg["ml"]["models"]["lstm_ae"]["hidden"] = 8
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    mp = pytest.MonkeyPatch()
    mp.setenv("HAI_DATA_DIR", str(root / "hai"))
    mp.setenv("ARTIFACT_DIR", str(root / "art" / "smoke"))
    for args in (["--model", "pca"], ["--model", "lstm_ae"], ["--ablation-index", "0"], ["--ablation-index", "3"]):
        assert run_model.main(["--config", str(config), "--smoke", *args]) == 0
    mp.setenv("ARTIFACT_DIR", str(root / "art"))
    yield root, config
    mp.undo()


def test_scripts_on_smoke_artifacts_write_only_to_smoke_reports(workspace):
    root, config = workspace
    out = root / "art" / "smoke_reports"
    base = ["--config", str(config), "--smoke"]
    assert make_figures.main([*base, "--models", "pca", "lstm_ae", "iforest"]) == 0  # iforest 没跑：跳过
    assert attribution.main([*base, "--models", "pca", "lstm_ae"]) == 0
    assert export_alerts.main(base) == 0
    assert make_ablation_report.main(base) == 0
    names = {p.name for p in out.iterdir()}
    assert {"scores_test1.png", "scores_test1_row1100.png", "scores_test5.png", "score_distribution.png", "training_curves.png",
            "per_subsystem_detection.csv", "attribution.csv", "attribution_summary.csv", "attribution_summary.md",
            "attribution_heatmap_lstm_ae.png", "alerts.json", "ablation.csv", "ablation.md", "ablation_heatmap.png",
            "ablation_quantiles.csv"} <= names
    # 冒烟模式不写正式位置
    assert not (root / "report").exists() or not list((root / "report").iterdir())
    assert not (root / "official_alerts.json").exists()

    det = pd.read_csv(out / "per_subsystem_detection.csv")
    assert set(det["model"]) == {"pca", "lstm_ae"} and set(det["subsystem"]) == {"ALL", "P1"}
    assert (det["n_events"] == 1).all()

    summary = pd.read_csv(out / "attribution_summary.csv")
    assert list(summary.columns) == attribution.SUMMARY_COLUMNS and summary["model"].tolist() == ["pca", "lstm_ae"]
    assert (summary["n_events"] == 1).all() and (summary["n_events_without_subsystem"] == 0).all()
    seg = pd.read_csv(out / "attribution.csv")
    assert set(seg["category"]) <= {"evaluable", "non_overlap", "overlap_no_subsystem_label"}
    evaluable = seg[seg["category"] == "evaluable"]
    assert set(evaluable["model"]) == {"pca", "lstm_ae"} and set(evaluable["top1_subsystem"]) <= {"P1", "P2", "P4"}
    assert (evaluable["event_subsystems"] == "P1").all()
    assert (evaluable["hit"].astype(str) == (evaluable["top1_subsystem"] == "P1").astype(str)).all()
    for row in summary.itertuples():  # 汇总与明细一致
        own = seg[seg["model"] == row.model]
        assert row.n_alarm_segments == len(own) and row.n_non_overlap == (own["category"] == "non_overlap").sum()
        assert row.n_hit == (own["hit"].astype(str) == "True").sum()

    alerts = json.loads((out / "alerts.json").read_text(encoding="utf-8"))
    check_contract(alerts, False, {"P1_A", "P1_B", "P2_A", "P4_A"})
    # 导出的报警与归因用的是同一批报警段
    ae = seg[seg["model"] == "lstm_ae"]
    assert [a["start"] for a in alerts] == ae["start_time"].tolist() and [a["end"] for a in alerts] == ae["end_time"].tolist()
    assert [a["topk"][0]["sensor"] for a in alerts] == ae["top1_sensor"].tolist()

    ab = pd.read_csv(out / "ablation.csv")
    assert ab[["window", "hidden"]].values.tolist() == [[12, 4], [20, 8]] and ab["is_main_config"].tolist() == [False, False]
    assert "缺少的组合（2 个）" in (out / "ablation.md").read_text(encoding="utf-8")
    q = pd.read_csv(out / "ablation_quantiles.csv")
    assert sorted(q["quantile"].unique()) == [0.99, 0.995, 0.999] and len(q) == 6


def test_official_mode_rejects_smoke_artifacts(workspace, capsys):
    """同一批冒烟产物放到正式位置：正式模式一律拒绝，不写报告目录，也不写 alerts.json。"""
    root, config = workspace
    art = root / "art"
    for name in ("pca", "lstm_ae"):
        if not (art / name).exists():
            (art / name).symlink_to(art / "smoke" / name)
    base = ["--config", str(config)]
    assert make_figures.main(base) == 0 and attribution.main(base) == 0 and make_ablation_report.main(base) == 0
    assert export_alerts.main(base) == 1
    assert "冒烟测试的产物" in capsys.readouterr().out
    assert not list((root / "report").iterdir()) and not (root / "official_alerts.json").exists()


def test_ablation_grid_matches_report_expectations():
    grid = ablation_grid(load_config())
    assert len(grid) == 9 and grid[4]["run_name"] == "ablation/lstm_ae_w60_h64"


# ---------- 多随机种子（主表 LSTM-AE 一行的均值 ± 标准差） ----------

def _fake_run(root, run_name, model, seed, overrides, f1, auroc, **manifest_extra):
    d = root / run_name
    d.mkdir(parents=True)
    row = {c: 0.5 for c in make_main_report.MAIN_COLUMNS}
    row.update(model=model, f1=f1, auroc=auroc, n_unscored=295)
    (d / "eval_metrics.json").write_text(json.dumps({"main_row": row}), encoding="utf-8")
    (d / "run_manifest.json").write_text(json.dumps({
        "smoke": False, "git_dirty": False, "status": "completed", "seed": seed, "overrides": overrides,
        **manifest_extra}), encoding="utf-8")


def test_main_report_seed_study_mean_and_std(tmp_path, monkeypatch, capsys):
    """全部种子可用时 LSTM-AE 一行写均值 ± 样本标准差；缺一个或配置不符就维持单种子，不用不完整的集合。"""
    cfg = load_config()
    study = cfg["ml"]["seed_study"]
    assert study["model"] == "lstm_ae" and cfg["ml"]["seed"] in study["seeds"] and len(study["seeds"]) == 5
    root, out = tmp_path / "artifacts", tmp_path / "report"
    monkeypatch.setenv("ARTIFACT_DIR", str(root))
    f1s = [0.40, 0.50, 0.45, 0.35, 0.55]
    _fake_run(root, "iforest", "iforest", 42, [], 0.44, 0.83)
    for seed, f1 in zip(study["seeds"][:-1], f1s):
        name = "lstm_ae" if seed == cfg["ml"]["seed"] else f"seeds/lstm_ae_s{seed}"
        _fake_run(root, name, "lstm_ae", seed, [] if seed == cfg["ml"]["seed"] else [f"ml.seed={seed}"], f1, 0.9)

    args = ["--models", "iforest", "lstm_ae", "--output-dir", str(out)]
    assert make_main_report.main(args) == 0  # 少一个种子
    assert "还不完整" in capsys.readouterr().out and not (out / "main_results_seeds.csv").exists()
    assert "±" not in (out / "main_results.md").read_text(encoding="utf-8")

    last = study["seeds"][-1]
    _fake_run(root, f"seeds/lstm_ae_s{last}", "lstm_ae", last, [f"ml.seed={last}", "ml.window.length=30"], f1s[-1], 0.9)
    assert make_main_report.main(args) == 0  # 多了别的覆盖项：不是同一个配置
    assert "覆盖项不符" in capsys.readouterr().out and not (out / "main_results_seeds.csv").exists()

    manifest = root / f"seeds/lstm_ae_s{last}" / "run_manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "overrides": [f"ml.seed={last}"]}))
    assert make_main_report.main(args) == 0
    md = (out / "main_results.md").read_text(encoding="utf-8")
    expected = f"{np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}"
    line = next(x for x in md.splitlines() if x.startswith("| lstm_ae |"))
    assert expected in line and "0.9000 ± 0.0000" in line
    assert "±" not in next(x for x in md.splitlines() if x.startswith("| iforest |")) and "单个种子" in md
    # csv 主表不变：仍是种子 42 的那一次运行，列也不变
    table = pd.read_csv(out / "main_results.csv")
    assert list(table.columns) == make_main_report.MAIN_COLUMNS
    assert table.set_index("model").loc["lstm_ae", "f1"] == f1s[0]
    seeds = pd.read_csv(out / "main_results_seeds.csv")
    assert seeds["seed"].tolist() == [str(s) for s in study["seeds"]] + ["mean", "std"]
    assert seeds["f1"].iloc[-1] == pytest.approx(np.std(f1s, ddof=1))


# ---------- M5 补充：按事件的主口径、置信区间、按长度分组、探索性归一化 ----------

def test_wilson_interval_and_precision_at_k():
    from src.eval.metrics import precision_at_k, wilson_interval
    lo, hi = wilson_interval(54, 72)
    assert (lo, hi) == (pytest.approx(0.6391, abs=1e-4), pytest.approx(0.8356, abs=1e-4))  # 与手算的 Wilson 区间一致
    assert wilson_interval(0, 3)[0] == 0.0 and wilson_interval(3, 3)[1] == 1.0 and wilson_interval(3, 3)[0] < 1.0
    assert all(np.isnan(v) for v in wilson_interval(0, 0))
    assert precision_at_k([True, False, True, True], 2) == 0.5 and precision_at_k([True, False], 10) == 0.5
    assert np.isnan(precision_at_k([], 5))


def test_event_first_segment_rule_and_duration_groups():
    """事件 3 有两个报警段：第一个指向 P2（命中），第二个更长、指向 P4；主口径只看第一个，合并口径被第二个主导。"""
    scores, err, index, events = hand_case()
    events[2] = {**events[2], "start": 36}   # 事件 3（P2）提前到第 36 秒开始
    scores[37:39] = 2.0                      # 事件 3 里先出现的一个短报警段（37~38），之后才是原来的 41~43
    err[37:39] = [1.0, 8.0, 1.0]
    rows, summary, event_rows = attribution.attribute_run(scores, err, index, 1.0, 0, events, [3])
    event_rows = attribution.add_first_segment(event_rows, rows)
    e3 = next(e for e in event_rows if e["event_id"] == 3)
    assert e3["first_top1_subsystem"] == "P2" and e3["first_hit"] is True and e3["first_duration_s"] == 2
    assert e3["top1_subsystem"] == "P4" and e3["hit"] is False            # 合并口径（补充）
    assert next(e for e in event_rows if e["event_id"] == 2)["first_hit"] is None  # 无子系统标签的事件不计入分母
    full = attribution.full_summary(summary, rows, event_rows)
    assert (full["n_events_first_evaluable"], full["n_events_first_hit"], full["n_events_merged_hit"]) == (2, 2, 1)
    assert full["event_first_hit_rate"] == 1.0 and full["event_hit_rate"] == 0.5
    assert 0 < full["event_first_hit_rate_ci_low"] < 1 and full["subsystem_hit_rate_ci_high"] <= 1

    assert [attribution.duration_group(d, [10, 60]) for d in (1, 9, 10, 60, 61)] == \
        ["<10 秒", "<10 秒", "10–60 秒", "10–60 秒", ">60 秒"]
    groups = attribution.by_duration(rows, [10, 60])
    assert [g["duration_group"] for g in groups] == ["<10 秒", "10–60 秒", ">60 秒"]
    assert sum(g["n"] for g in groups) == summary["n_evaluable"] and np.isnan(groups[2]["rate"])


def test_exploratory_val_normalisation_changes_top1_only_through_baseline():
    """基线大的传感器（平时误差就高）归一化后让位；不给基线时与原来完全相同。"""
    err = np.array([[4.0, 3.0, 1.0], [4.0, 3.0, 1.0]])
    plain = sensor_shares(err, 0, 1)
    assert top_k(plain, FEATURES, 1)[0]["sensor"] == "P1_A" and plain.sum() == pytest.approx(1)
    normed = sensor_shares(err, 0, 1, baseline=np.array([4.0, 1.0, 1.0]))   # 相对平时的倍数：1、3、1
    assert top_k(normed, FEATURES, 1)[0]["sensor"] == "P2_B" and normed.tolist() == pytest.approx([0.2, 0.6, 0.2])
    zero = sensor_shares(err, 0, 1, baseline=np.array([0.0, 1.0, 1.0]))     # 基线不为正的传感器不参与
    assert zero[0] == 0 and zero.sum() == pytest.approx(1)


# ---------- I1：融合排序评估 ----------

def test_fusion_ranking_eval_precision_at_k_and_alignment():
    from src.eval import fusion_ranking
    alerts = [{"alert_id": f"A-{i}", "start": f"2020-01-01 00:00:0{i}", "end": f"2020-01-01 00:00:0{i}"} for i in range(4)]
    alarms = pd.DataFrame({"start_time": [a["start"] for a in alerts], "end_time": [a["end"] for a in alerts],
                           "overlaps_event": [True, False, True, False]})
    truth = fusion_ranking.truth_by_alert(alerts, alarms)
    assert truth == {"A-0": True, "A-1": False, "A-2": True, "A-3": False}
    with pytest.raises(ValueError):   # 条数不同：不是同一次运行
        fusion_ranking.truth_by_alert(alerts[:3], alarms)
    with pytest.raises(ValueError):   # 时间对不上
        fusion_ranking.truth_by_alert(alerts, alarms.assign(end_time=alarms["end_time"].iloc[::-1].to_numpy()))

    ranked = {"raw_score": [("A-0", 9.0), ("A-2", 8.0), ("A-1", 1.0), ("A-3", 0.5)],
              "alpha=1": [("A-1", 2.0), ("A-0", 2.0), ("A-3", 2.0), ("A-2", 0.1)]}
    df = fusion_ranking.evaluate_rankings(ranked, truth, [2, 10]).set_index(["ranking", "k"])
    assert df.loc[("raw_score", 2), "precision"] == 1.0 and df.loc[("alpha=1", 2), "precision"] == 0.5
    assert df.loc[("alpha=1", 2), "n_tied_with_kth"] == 3 and df.loc[("alpha=1", 2), "n_shared_with_raw_score"] == 1
    assert df.loc[("raw_score", 10), "precision"] == 0.5 and df.loc[("raw_score", 10), "base_rate"] == 0.5
    assert list(df.reset_index().columns) == fusion_ranking.COLUMNS


def test_fusion_ranking_module_does_not_read_labels():
    """融合排序评估不在标签白名单内：真值只能来自 evaluate 写出的 eval_alarms.csv。"""
    text = (ROOT / "src" / "eval" / "fusion_ranking.py").read_text(encoding="utf-8")
    assert "eval_alarms.csv" in text and "labels" not in text.replace("标签", "")


# ---------- I1：案例挑选 ----------

def test_case_rules_and_observations():
    """案例按固定规则挑选；"放在一起看"的每一条只在数据确实如此时才写。"""
    from types import SimpleNamespace as NS

    from scripts import make_cases
    rows = [NS(category="non_overlap", hit=None), NS(category="evaluable", hit=False), NS(category="evaluable", hit=True)]
    assert [next(i for i, r in enumerate(rows) if rule(r)) for _, rule in make_cases.CASES] == [2, 1, 0]

    def alert(rank, score, subs, asset="ews_01"):
        return {"rank": rank, "max_score": score, "threshold": 1.0,
                "subsystems": [{"id": k, "weight": w} for k, w in subs.items()],
                "vulns": [{"nvt_oid": "1.2.3", "asset": asset}]}
    picked = {1: (NS(event_subsystems="P2"), alert(1, 900.0, {"P2": 1.0})),
              2: (NS(event_subsystems="P1"), alert(50, 23.0, {"P1": 0.7, "P4": 0.3})),
              3: (NS(event_subsystems=float("nan")), alert(28, 2.2, {"P1": 1.0}))}
    text = "\n".join(make_cases.observations(picked))
    assert "70% 的权重放在 P1" in text and "封顶" in text and "同一台已扫描的资产（ews_01）" in text and "P4" in text
    # 误报排在真实攻击之后、漏洞来自不同资产时，对应的两条不出现
    picked[3] = (picked[3][0], alert(60, 2.2, {"P1": 1.0}, asset="hmi_main"))
    text = "\n".join(make_cases.observations(picked))
    assert "封顶" not in text and "同一台" not in text
    assert make_cases.observations({}) == []


# ---------- R1：关键数字 ----------

def test_key_numbers_match_source_tables_and_progress_copy():
    """key_numbers.md 里的检测数字与主表一致；PROGRESS.md 顶部的关键数字是它的原样副本（两处不许各写各的）。"""
    report = ROOT / load_config()["ml"]["report_dir"]
    text = (report / "key_numbers.md").read_text(encoding="utf-8")
    main = pd.read_csv(report / "main_results.csv")
    for row in main.itertuples():
        assert f"{row.model}：precision {row.precision:.4f}，recall {row.recall:.4f}，F1 {row.f1:.4f}，AUROC {row.auroc:.4f}" in text
    seeds = pd.read_csv(report / "main_results_seeds.csv").set_index("seed")
    for col in ("f1", "auroc", "event_recall"):
        assert f"{seeds.loc['mean', col]:.4f} ± {seeds.loc['std', col]:.4f}" in text
    assert f"{seeds.loc['mean', 'false_alarms_per_hour']:.2f} ± {seeds.loc['std', 'false_alarms_per_hour']:.2f}" in text
    md = (report / "main_results.md").read_text(encoding="utf-8")
    line = next(x for x in md.splitlines() if x.startswith("| lstm_ae |"))
    assert line.count("±") == 4                      # 主表该行：F1、AUROC、事件检出率、误报率
    for group in ("### 检测", "### 归因", "### 融合", "### 数据问题"):
        assert group in text
    assert "不使用 point-adjust" in text and "事后分析，不替换主结果" in text
    assert text in (ROOT / "PROGRESS.md").read_text(encoding="utf-8")


# ---------- R1：各次运行的入库副本（reports/main/runs/） ----------

def _snapshot_dir():
    from scripts import snapshot_runs
    cfg = load_config()
    return cfg, snapshot_runs, ROOT / cfg["ml"]["report_dir"] / snapshot_runs.SNAPSHOT_SUBDIR


def test_run_snapshots_are_official_and_key_numbers_reproducible():
    """入库副本里每次运行都是集群完整运行；关键数字只用入库的文件就能重新生成，且与入库的版本相同。"""
    from scripts import make_key_numbers
    cfg, snapshot_runs, runs = _snapshot_dir()
    names = snapshot_runs.run_names(cfg)
    assert len(names) == 5 + 9 + 4 and len(set(names)) == len(names)
    for name in names:
        assert report_problem(runs / name) is None, name
        assert (runs / name / "eval_metrics.json").exists() and (runs / name / "threshold.json").exists()
    assert make_key_numbers.build(cfg) == (ROOT / cfg["ml"]["report_dir"] / "key_numbers.md").read_text(encoding="utf-8")
    assert not [p for p in runs.rglob("*") if p.is_file() and p.name not in snapshot_runs.FILES]  # 不带大文件和日志


def test_run_snapshots_contain_no_personal_or_cluster_identifiers():
    """仓库可能公开：入库副本里不出现用户名、集群路径、主机名；hostname / node 一律是占位符。"""
    cfg, snapshot_runs, runs = _snapshot_dir()
    rules = snapshot_runs.scrub_rules()   # 真实的用户名和路径在不入库的 cluster.local.yaml 里；没有它时只剩登录主机名
    assert ("<cluster-host>" in {r for _, r in rules})

    def node_values(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in snapshot_runs.NODE_KEYS:
                    yield v
                else:
                    yield from node_values(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from node_values(v)

    for path in runs.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for needle, _ in rules:
            assert needle not in text, (path, needle)
        assert not snapshot_runs.USER_PATH_RE.search(text), path
        if path.name == "run_manifest.json":
            values = set(node_values(json.loads(text)))
            assert values <= {snapshot_runs.NODE_PLACEHOLDER, None, ""}, (path, values)
            assert snapshot_runs.NODE_PLACEHOLDER in values

    # 脱敏函数本身：键名命中的替换，字符串里的用户名和路径替换，其余不动
    sample = {"hostname": "nodeA", "slurm": {"node": "nodeA", "job_id": "1"}, "note": "/data/users/someone/x", "seed": 42}
    assert snapshot_runs.scrub_nodes(sample) == {"hostname": "<cluster-node>", "slurm": {"node": "<cluster-node>", "job_id": "1"},
                                                 "note": "/data/users/someone/x", "seed": 42}
    assert snapshot_runs.scrub_text('"/data/users/someone/x" /home/abc/y', []) == '"<user-path>/x" <user-path>/y'
    assert snapshot_runs.scrub_text("in /opt/secret/dir by cid123", [("/opt/secret/dir", "<cluster-path>"), ("cid123", "<cluster-user>")]) \
        == "in <cluster-path> by <cluster-user>"


def test_run_snapshots_match_originals_in_artifact_dir():
    """入库副本 = 对产物目录中的原件做脱敏的结果；没有命中脱敏规则的文件与原件逐字节相同。
    产物目录不入库：原件不在时（集群上、全新的克隆）跳过。"""
    cfg, snapshot_runs, runs = _snapshot_dir()
    root = ROOT / cfg["ml"]["artifact_dir"]
    rules = snapshot_runs.scrub_rules()
    checked = identical = 0
    for name in snapshot_runs.run_names(cfg):
        for f in snapshot_runs.FILES:
            original, copy = root / name / f, runs / name / f
            if not original.exists():
                assert not copy.exists() or not (root / name).exists(), f"{name}/{f}: 入库副本没有对应的原件"
                continue
            assert copy.read_bytes() == snapshot_runs.redacted(original, rules).encode("utf-8"), f"{name}/{f}"
            if f != "run_manifest.json":
                assert copy.read_bytes() == original.read_bytes(), f"{name}/{f}"   # 只有 manifest 需要脱敏
                identical += 1
            else:
                # manifest：除 hostname / node 外逐项相同
                a, b = json.loads(original.read_text(encoding="utf-8")), json.loads(copy.read_text(encoding="utf-8"))
                assert snapshot_runs.scrub_nodes(a) == b and a["git_commit"] == b["git_commit"] and a["stages"] == b["stages"]
            checked += 1
    if checked == 0:
        pytest.skip("产物目录中没有原件（artifacts/ 不入库），无法核对")
    assert checked == 88 and identical == 70


# ---------- 英文化：图表与表格在两种显示语言下数值一致 ----------

def test_figures_and_tables_identical_between_languages(workspace, monkeypatch):
    """同一批冒烟产物，中文、英文各生成一次全部图表：csv / json 逐字节相同（只有图上的文字不同）；
    图上的文字里出现的数字相同；英文图上没有中文，中文图上有中文。"""
    import hashlib
    import re
    from collections import Counter

    from matplotlib.figure import Figure
    from matplotlib.text import Text

    from src.utils import i18n
    root, config = workspace
    out = root / "art" / "smoke_reports"
    base = ["--config", str(config), "--smoke"]
    captured: dict[str, list[str]] = {}
    original = Figure.savefig

    def recording_savefig(fig, path, *a, **k):
        texts = [x.get_text() for x in fig.findobj(Text) if x.get_text().strip()]
        captured.setdefault(lang, []).extend([f"## {getattr(path, 'name', path)}", *texts])
        return original(fig, path, *a, **k)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    digests = {}
    for lang in ("zh", "en"):
        monkeypatch.setenv(i18n.ENV_VAR, lang)
        assert make_figures.main([*base, "--models", "pca", "lstm_ae"]) == 0
        assert attribution.main([*base, "--models", "pca", "lstm_ae"]) == 0
        assert make_ablation_report.main(base) == 0
        # .md 是中文文档，不在英文化范围内，但内容同样不应随显示语言变化
        digests[lang] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(out.iterdir()) if p.suffix in (".csv", ".json", ".md")}
        assert all(p.stat().st_size > 0 for p in out.glob("*.png"))
    assert digests["zh"] == digests["en"] and len(digests["en"]) >= 8
    num = re.compile(r"\d+(?:\.\d+)?")
    cjk = re.compile(r"[\u4e00-\u9fff]")
    zh, en = "\n".join(captured["zh"]), "\n".join(captured["en"])
    assert len(captured["zh"]) == len(captured["en"]) > 50          # 同样多的文字对象
    assert Counter(num.findall(zh)) == Counter(num.findall(en))      # 刻度、格子里的数值、标题里的数字都相同
    assert cjk.search(zh) and not cjk.search(en), cjk.findall(en)[:10]
    assert "Anomaly score" in en and "Training curves" in en and "smoke-test data" in en and "Point-wise F1" in en
