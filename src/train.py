"""Training stage: fit the model on the train part; val_es is used only for early stopping (main-line decision 1). Reads neither val_thr nor test.

With --resume and an already trained model in the run directory, it is loaded instead of retrained; an LSTM model interrupted mid-training continues
from the latest checkpoint (see src/models/torch_base.py). So an interrupted job can simply be resubmitted unchanged.

Usage:
    python -m src.train --model pca [--smoke] [--resume] [--set key=value]
"""

import sys

from src.models import build_model
from src.utils.context import stage_cli


def run(ctx) -> dict:
    model = build_model(ctx.model, ctx.cfg)
    model.bind(directory=ctx.directory, resume=ctx.resume, log=ctx.log, manifest=ctx.manifest)
    if ctx.resume and model.exists(ctx.directory):
        model.load(ctx.directory)
        ctx.state["model"] = model
        ctx.log.info("运行目录里已有训练好的模型，直接读取，不重新训练")
        previous = ctx.manifest.data.get("stages", {}).get("train", {})  # Keep the summary of the previous training
        return {**{k: v for k, v in previous.items() if k not in ("seconds", "finished_at")}, "resumed": True}
    summary = model.fit(ctx.dataset("train"), ctx.dataset("val_es")) or {}
    model.save(ctx.directory)
    ctx.state["model"] = model
    ctx.log.info("训练完成：%s", summary)
    return {"resumed": False, **summary}


if __name__ == "__main__":
    sys.exit(stage_cli("prepare,train", "训练一个模型"))
