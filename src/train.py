"""训练阶段：在 train 部分上拟合模型，val_es 只用于早停（主线补充决策 1）。不读取 val_thr 和 test。

带 --resume 且运行目录里已有训练好的模型时直接读取，不重新训练；LSTM 类模型训练到一半被中断时从最近的
checkpoint 接着训练（见 src/models/torch_base.py）。所以作业被中断后原样重新提交即可。

用法：
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
        previous = ctx.manifest.data.get("stages", {}).get("train", {})  # 保留上次训练的摘要
        return {**{k: v for k, v in previous.items() if k not in ("seconds", "finished_at")}, "resumed": True}
    summary = model.fit(ctx.dataset("train"), ctx.dataset("val_es")) or {}
    model.save(ctx.directory)
    ctx.state["model"] = model
    ctx.log.info("训练完成：%s", summary)
    return {"resumed": False, **summary}


if __name__ == "__main__":
    sys.exit(stage_cli("prepare,train", "训练一个模型"))
