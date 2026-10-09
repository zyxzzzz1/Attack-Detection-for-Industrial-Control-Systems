"""一个模型的完整运行入口：准备数据 → 训练 → 打分 → 阈值 → 评估。集群作业脚本（cluster/*.sbatch）只调用这里。

每次运行在 产物目录/<运行名>/ 下写 run_manifest.json、run.log 和各阶段的产物。带 --resume 时，已有 checkpoint
就从断点继续，没有就从头开始，所以作业被中断后原样重新提交即可。

阶段（--stages，默认按顺序执行全部阶段；每个阶段也可以单独执行，需要的东西从运行目录读取）：
    prepare    读取 train 文件、切分、fit 预处理，保存 scaler.joblib 与 preprocess.json（不读取 test）
    train      在 train 部分上拟合模型，val_es 只用于早停（src/train.py）
    diagnose   只用 val_thr 的重尾特征诊断；单个传感器占比超限时在读取 test 之前停下（src/diagnose.py）
    score      给 val_thr 和 test 打分（只读传感器列，src/score.py）
    threshold  阈值 = val_thr 分数的分位数（src/threshold.py）
    evaluate   唯一读取 test 标签的阶段：逐点与事件级指标（src/evaluate.py）

退出码：0 完成；3 诊断超限，停下等待决定（run_manifest.json 的 status 为 stopped_for_review）。

用法：
    python -m src.run_model --model lstm_ae [--resume] [--set ml.window.length=30 ...]
    python -m src.run_model --model lstm_ae --ablation-index 0 --resume     # 消融网格的第 0 个组合
    python -m src.run_model --model pca --smoke                               # 1% 数据、1 个 epoch，只检查流程
    python -m src.run_model --list-ablation                                   # 打印消融网格与任务号
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src import diagnose, evaluate, score, threshold, train
from src.data.preprocess import prepare_train
from src.utils.context import RunContext
from src.utils.paths import DEFAULT_CONFIG, load_config
from src.utils.run import (RunManifest, ablation_grid, apply_overrides, config_hash, limit_threads, resolve_num_workers,
                           run_dir, seed_everything, setup_logging)

EXIT_STOPPED_FOR_REVIEW = 3
# 本地诊断运行允许的阶段：都不读取 test 文件
LOCAL_DIAGNOSTIC_STAGES = ("prepare", "train", "diagnose")


def stage_prepare(ctx: RunContext) -> dict:
    """读取 train 文件并 fit 预处理（只看 train 部分），保存 scaler 与摘要。不读取 test 文件。"""
    ctx.data = prepare_train(ctx.cfg, ctx.fraction)
    summary = ctx.data.summary()
    summary["windows"] = {split: len(ctx.dataset(split)) for split in ctx.data.segments}
    if min(summary["windows"].values()) == 0:
        raise RuntimeError(f"有数据部分切不出任何窗口：{summary['windows']}（数据太少或窗口太长）")
    ctx.data.pre.save(ctx.directory / "scaler.joblib")
    (ctx.directory / "preprocess.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    ctx.log.info("数据：%s 行，特征 %d 个，窗口 %s", summary["rows"], summary["n_features"], summary["windows"])
    return {"rows": summary["rows"], "n_features": summary["n_features"], "windows": summary["windows"]}


# 阶段表：按执行顺序排列。diagnose 必须排在 score 之前（超限时要在读取 test 之前停下）。
STAGES = {"prepare": stage_prepare, "train": train.run, "diagnose": diagnose.run, "score": score.run,
          "threshold": threshold.run, "evaluate": evaluate.run}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行一个模型的完整流程（本地冒烟或集群作业）")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", help="模型名，取值见 config 的 ml.models")
    p.add_argument("--run-name", help="运行目录名（相对产物根目录），默认等于模型名")
    p.add_argument("--stages", default="all", help=f"逗号分隔的阶段，默认 all；可选：{','.join(STAGES)}")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="覆盖配置项，如 --set ml.window.length=30，可重复")
    p.add_argument("--ablation-index", type=int, help="消融网格的任务号（SLURM_ARRAY_TASK_ID），自动设置覆盖项与运行名")
    p.add_argument("--list-ablation", action="store_true", help="打印消融网格后退出")
    p.add_argument("--resume", action="store_true", help="有 checkpoint 就从断点继续，没有就从头开始")
    p.add_argument("--local-diagnostic", action="store_true",
                   help=f"本地诊断运行：只执行 {','.join(LOCAL_DIAGNOSTIC_STAGES)}，不读取 test；manifest 标记 "
                        "local_diagnostic=true，不能用于报告。运行目录默认为 local_diagnostic/<模型名>")
    p.add_argument("--smoke", action="store_true", help="冒烟测试：1%% 数据、1 个 epoch；数字不得写进报告")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.list_ablation:
        for g in ablation_grid(cfg):
            print(f"{g['index']}\t{g['run_name']}\t{' '.join(g['overrides'])}")
        return 0

    overrides, run_name = list(args.overrides), args.run_name
    if args.ablation_index is not None:
        grid = ablation_grid(cfg)
        if not 0 <= args.ablation_index < len(grid):
            p.error(f"--ablation-index 超出范围 0~{len(grid) - 1}")
        combo = grid[args.ablation_index]
        args.model = args.model or cfg["ml"]["ablation"]["model"]
        if args.model != cfg["ml"]["ablation"]["model"]:
            p.error(f"消融只针对 {cfg['ml']['ablation']['model']}")
        overrides = combo["overrides"] + overrides
        run_name = run_name or combo["run_name"]
    if not args.model:
        p.error("需要 --model")
    if args.model not in cfg["ml"]["models"]:
        p.error(f"未知模型 {args.model!r}，可选：{', '.join(cfg['ml']['models'])}")
    stages = list(STAGES) if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    if args.local_diagnostic:
        if args.stages == "all":
            stages = list(LOCAL_DIAGNOSTIC_STAGES)
        elif not set(stages) <= set(LOCAL_DIAGNOSTIC_STAGES):
            p.error(f"--local-diagnostic 只允许阶段 {','.join(LOCAL_DIAGNOSTIC_STAGES)}（不读取 test）")
        if args.smoke:
            p.error("--local-diagnostic 用全量 train 数据，不能与 --smoke 同用")
        run_name = run_name or f"local_diagnostic/{args.model}"
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        p.error(f"未知阶段 {unknown}，可选：{', '.join(STAGES)}")

    cfg = apply_overrides(cfg, overrides)
    if args.smoke:
        cfg = apply_overrides(cfg, [f"ml.train.max_epochs={cfg['ml']['smoke']['max_epochs']}"])
    directory = run_dir(cfg, run_name or args.model)
    log = setup_logging(directory)
    seed_everything(cfg["ml"]["seed"])
    threads = limit_threads()
    workers = resolve_num_workers(cfg["ml"]["runtime"]["num_workers"])

    manifest = RunManifest(directory)
    previous = manifest.data
    if (args.resume and args.stages == "all" and previous.get("status") == "completed"
            and previous.get("config_hash") == config_hash(cfg) and not previous.get("smoke") and not args.smoke):
        # --dependency=afterany 串接的后一段作业：前一段已经用同样的配置跑完，无事可做
        log.info("运行 %s 已经用相同的配置完成（%s），无事可做", run_name or args.model, previous.get("finished_at"))
        return 0
    manifest.start(cfg, model=args.model, run_name=run_name or args.model, smoke=args.smoke, overrides=overrides,
                   argv=argv)
    manifest.update(torch_threads=threads, num_workers=workers, local_diagnostic=bool(args.local_diagnostic))
    d = manifest.data
    log.info("运行 %s（模型 %s%s）目录 %s", d["run_name"], args.model, "，冒烟测试" if args.smoke else "", directory)
    log.info("提交 %s%s，配置哈希 %s，覆盖项 %s", d["git_commit"], "（工作区有未提交改动）" if d["git_dirty"] else "",
             d["config_hash"][:12], overrides or "无")
    log.info("主机 %s，设备 %s（%s），CPU %d 个，python %s，%s", d["hostname"], d["device"], d["device_name"],
             d["cpus_available"], d["python"], d["versions"])
    if d["slurm_job_id"]:
        log.info("SLURM 作业 %s，array 任务 %s，节点 %s，GPU %s", d["slurm_job_id"], d["slurm_array_task_id"], d["node"],
                 d["gpu_nvidia_smi"])

    ctx = RunContext(cfg=cfg, model=args.model, run_name=d["run_name"], directory=directory, smoke=args.smoke,
                     resume=args.resume, manifest=manifest, log=log)
    try:
        for name in stages:
            t0 = time.time()
            log.info("阶段 %s 开始", name)
            result = STAGES[name](ctx) or {}
            manifest.stage(name, seconds=round(time.time() - t0, 2), **result)
            log.info("阶段 %s 完成，用时 %.1f 秒", name, time.time() - t0)
    except diagnose.DiagnosisStop as e:
        manifest.finish("stopped_for_review")
        log.warning("停下等待决定：%s", e)
        return EXIT_STOPPED_FOR_REVIEW
    except BaseException as e:  # 包括被调度器终止（KeyboardInterrupt / SystemExit）：记录后原样抛出
        manifest.finish("failed" if isinstance(e, Exception) else "interrupted")
        log.error("运行中止：%r", e)
        raise
    manifest.finish("completed")
    log.info("运行完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
