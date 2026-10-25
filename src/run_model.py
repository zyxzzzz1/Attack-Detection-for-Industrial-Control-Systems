"""Entry point for a complete run of one model: prepare data -> train -> score -> threshold -> evaluate. The cluster job scripts (cluster/*.sbatch) call only this.

Each run writes run_manifest.json, run.log and each stage's artifacts under <artifact dir>/<run name>/. With --resume, an existing checkpoint
is resumed and otherwise the run starts from scratch, so an interrupted job can simply be resubmitted unchanged.

Stages (--stages; by default all stages run in order; each can also run on its own, reading what it needs from the run directory):
    prepare    read train files, split, fit preprocessing, save scaler.joblib and preprocess.json (test is not read)
    train      fit the model on the train part, val_es used only for early stopping (src/train.py)
    diagnose   heavy-tail feature diagnosis on val_thr only; stops before reading test if a single sensor's share exceeds the limit (src/diagnose.py)
    score      score val_thr and test (sensor columns only, src/score.py)
    threshold  threshold = quantile of the val_thr scores (src/threshold.py)
    evaluate   the only stage that reads test labels: point-wise and event-level metrics (src/evaluate.py)

Exit codes: 0 done; 3 diagnosis over the limit, stopped awaiting a decision (status stopped_for_review in run_manifest.json).

Usage:
    python -m src.run_model --model lstm_ae [--resume] [--set ml.window.length=30 ...]
    python -m src.run_model --model lstm_ae --ablation-index 0 --resume     # combination 0 of the ablation grid
    python -m src.run_model --model pca --smoke                               # 1% of the data, 1 epoch, pipeline check only
    python -m src.run_model --list-ablation                                   # print the ablation grid and task indices
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
# Stages allowed in a local diagnostic run: none of them read test files
LOCAL_DIAGNOSTIC_STAGES = ("prepare", "train", "diagnose")


def stage_prepare(ctx: RunContext) -> dict:
    """Read train files and fit preprocessing (train part only); save the scaler and summary. Does not read test files."""
    ctx.data = prepare_train(ctx.cfg, ctx.fraction)
    summary = ctx.data.summary()
    summary["windows"] = {split: len(ctx.dataset(split)) for split in ctx.data.segments}
    if min(summary["windows"].values()) == 0:
        raise RuntimeError(f"some data part yields no windows at all: {summary['windows']} (too little data or window too long)")
    ctx.data.pre.save(ctx.directory / "scaler.joblib")
    (ctx.directory / "preprocess.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    ctx.log.info("data: %s rows, %d features, windows %s", summary["rows"], summary["n_features"], summary["windows"])
    return {"rows": summary["rows"], "n_features": summary["n_features"], "windows": summary["windows"]}


# Stage table in execution order. diagnose must come before score (when over the limit it must stop before test is read).
STAGES = {"prepare": stage_prepare, "train": train.run, "diagnose": diagnose.run, "score": score.run,
          "threshold": threshold.run, "evaluate": evaluate.run}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the complete pipeline of one model (local smoke run or cluster job)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", help="model name, see ml.models in the config")
    p.add_argument("--run-name", help="run directory name (relative to the artifact root), defaults to the model name")
    p.add_argument("--stages", default="all", help=f"comma-separated stages, default all; choices: {','.join(STAGES)}")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config item, e.g. --set ml.window.length=30; repeatable")
    p.add_argument("--ablation-index", type=int, help="task index in the ablation grid (SLURM_ARRAY_TASK_ID); sets the overrides and run name automatically")
    p.add_argument("--list-ablation", action="store_true", help="print the ablation grid and exit")
    p.add_argument("--resume", action="store_true", help="resume from a checkpoint if there is one, otherwise start from scratch")
    p.add_argument("--local-diagnostic", action="store_true",
                   help=f"local diagnostic run: only executes {','.join(LOCAL_DIAGNOSTIC_STAGES)} and does not read test; the manifest is marked "
                        "local_diagnostic=true and cannot be used for the report. Run directory defaults to local_diagnostic/<model>")
    p.add_argument("--smoke", action="store_true", help="smoke test: 1%% of the data, 1 epoch; the numbers must not go into the report")
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
            p.error(f"--ablation-index out of range 0~{len(grid) - 1}")
        combo = grid[args.ablation_index]
        args.model = args.model or cfg["ml"]["ablation"]["model"]
        if args.model != cfg["ml"]["ablation"]["model"]:
            p.error(f"the ablation only applies to {cfg['ml']['ablation']['model']}")
        overrides = combo["overrides"] + overrides
        run_name = run_name or combo["run_name"]
    if not args.model:
        p.error("--model is required")
    if args.model not in cfg["ml"]["models"]:
        p.error(f"unknown model {args.model!r}, choices: {', '.join(cfg['ml']['models'])}")
    stages = list(STAGES) if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    if args.local_diagnostic:
        if args.stages == "all":
            stages = list(LOCAL_DIAGNOSTIC_STAGES)
        elif not set(stages) <= set(LOCAL_DIAGNOSTIC_STAGES):
            p.error(f"--local-diagnostic only allows the stages {','.join(LOCAL_DIAGNOSTIC_STAGES)} (test is not read)")
        if args.smoke:
            p.error("--local-diagnostic uses the full train data and cannot be combined with --smoke")
        run_name = run_name or f"local_diagnostic/{args.model}"
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        p.error(f"unknown stages {unknown}, choices: {', '.join(STAGES)}")

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
        # A follow-up job chained with --dependency=afterany: the previous job already finished with the same config, nothing to do
        log.info("run %s already completed with the same config (%s), nothing to do", run_name or args.model, previous.get("finished_at"))
        return 0
    manifest.start(cfg, model=args.model, run_name=run_name or args.model, smoke=args.smoke, overrides=overrides,
                   argv=argv)
    manifest.update(torch_threads=threads, num_workers=workers, local_diagnostic=bool(args.local_diagnostic))
    d = manifest.data
    log.info("run %s (model %s%s), directory %s", d["run_name"], args.model, ", smoke test" if args.smoke else "", directory)
    log.info("commit %s%s, config hash %s, overrides %s", d["git_commit"], " (worktree has uncommitted changes)" if d["git_dirty"] else "",
             d["config_hash"][:12], overrides or "none")
    log.info("host %s, device %s (%s), %d CPUs, python %s, %s", d["hostname"], d["device"], d["device_name"],
             d["cpus_available"], d["python"], d["versions"])
    if d["slurm_job_id"]:
        log.info("SLURM job %s, array task %s, node %s, GPU %s", d["slurm_job_id"], d["slurm_array_task_id"], d["node"],
                 d["gpu_nvidia_smi"])

    ctx = RunContext(cfg=cfg, model=args.model, run_name=d["run_name"], directory=directory, smoke=args.smoke,
                     resume=args.resume, manifest=manifest, log=log)
    try:
        for name in stages:
            t0 = time.time()
            log.info("stage %s started", name)
            result = STAGES[name](ctx) or {}
            manifest.stage(name, seconds=round(time.time() - t0, 2), **result)
            log.info("stage %s done in %.1f s", name, time.time() - t0)
    except diagnose.DiagnosisStop as e:
        manifest.finish("stopped_for_review")
        log.warning("stopped awaiting a decision: %s", e)
        return EXIT_STOPPED_FOR_REVIEW
    except BaseException as e:  # Including termination by the scheduler (KeyboardInterrupt / SystemExit): record it and re-raise unchanged
        manifest.finish("failed" if isinstance(e, Exception) else "interrupted")
        log.error("run aborted: %r", e)
        raise
    manifest.finish("completed")
    log.info("run completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
