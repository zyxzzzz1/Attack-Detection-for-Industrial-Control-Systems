"""Preprocessing: drop constant columns of the training part, fit StandardScaler on the training part only (constraint 5), output float32.

Fixed order: drop rows overlapping test (load_hai) -> split train | val_es | val_thr -> determine constant columns on the train part -> fit scaler.
Constant columns and the scaler look only at the train part, never at val_es / val_thr / test.

Usage (only prints the data split and preprocessing summary; optionally saves the scaler):
    python -m src.data.preprocess [--fraction 0.01] [--save-dir DIR]
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.load_hai import SPLITS, Segment, load_test_segments, load_train_segments, split_segments
from src.data.windowing import WindowDataset
from src.utils.paths import DEFAULT_CONFIG, load_config


class Preprocessor:
    """Constant-column removal + standardization. fit accepts only segments of the training part."""

    def __init__(self) -> None:
        self.columns: list[str] = []       # Input columns (all sensors)
        self.constant_cols: list[str] = []
        self.features: list[str] = []      # Feature columns kept
        self.scaler: StandardScaler | None = None
        self.n_fit_rows = 0

    def fit(self, train_segments: list[Segment]) -> "Preprocessor":
        if any(s.split != "train" for s in train_segments):
            raise ValueError("Preprocessor 只能在 train 部分上 fit（约束 5）")
        self.columns = train_segments[0].columns
        for s in train_segments:
            self._check(s)
        lo = np.min([s.x.min(axis=0) for s in train_segments], axis=0)
        hi = np.max([s.x.max(axis=0) for s in train_segments], axis=0)
        keep = hi > lo
        self.constant_cols = [c for c, k in zip(self.columns, keep) if not k]
        self.features = [c for c, k in zip(self.columns, keep) if k]
        self._keep_idx = np.flatnonzero(keep)
        self.scaler = StandardScaler()
        for s in train_segments:  # Accumulate segment by segment to avoid building another full copy; compute mean and variance in float64
            self.scaler.partial_fit(s.x[:, self._keep_idx].astype(np.float64))
        self.n_fit_rows = sum(len(s) for s in train_segments)
        return self

    def _check(self, seg: Segment) -> None:
        if seg.columns != self.columns:
            raise ValueError(f"{seg.name} 的列与训练数据不一致")
        if np.isnan(seg.x).any():
            raise ValueError(f"{seg.name} 含缺失值；HAI 21.03 原始数据没有缺失（见 data_summary.md），请检查数据文件")

    def transform(self, seg: Segment) -> Segment:
        self._check(seg)
        x = self.scaler.transform(seg.x[:, self._keep_idx].astype(np.float64)).astype(np.float32)
        return replace(seg, columns=self.features, x=x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Preprocessor":
        return joblib.load(path)


@dataclass
class TrainData:
    pre: Preprocessor
    segments: dict[str, list[Segment]]  # train / val_es / val_thr, standardized

    def dataset(self, split: str, window: int, stride: int) -> WindowDataset:
        return WindowDataset([s.x for s in self.segments[split]], window, stride)

    def summary(self) -> dict:
        return {
            "rows": {k: sum(len(s) for s in v) for k, v in self.segments.items()},
            "segments": {k: [{"file": s.name, "start_row": s.start_row, "start_time": s.start_time, "rows": len(s)}
                             for s in v] for k, v in self.segments.items()},
            "n_features": len(self.pre.features), "features": self.pre.features,
            "constant_cols": self.pre.constant_cols, "scaler_fit_rows": self.pre.n_fit_rows,
        }


def prepare_train(cfg: dict, fraction: float = 1.0) -> TrainData:
    """Read train files -> split -> fit preprocessing on the train part -> transform all three parts. Does not read test files."""
    data_cfg = cfg["ml"]["data"]
    raw = split_segments(load_train_segments(cfg, fraction), data_cfg["val_fraction"], data_cfg["val_es_fraction"])
    pre = Preprocessor().fit(raw["train"])
    return TrainData(pre=pre, segments={k: [pre.transform(s) for s in raw[k]] for k in SPLITS})


def prepare_test(cfg: dict, pre: Preprocessor, fraction: float = 1.0) -> list[Segment]:
    """Transform the test files (one segment per file) with the fitted preprocessing. Must only be called by the scoring / final evaluation pipeline (constraint 2)."""
    return [pre.transform(s) for s in load_test_segments(cfg, fraction)]


def main() -> None:
    p = argparse.ArgumentParser(description="数据划分与预处理摘要（不读取 test 文件）")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--fraction", type=float, default=1.0, help="每个文件只取前这一比例的行（冒烟测试用）")
    p.add_argument("--save-dir", type=Path, help="保存 scaler.joblib 与 preprocess.json 的目录")
    args = p.parse_args()

    cfg = load_config(args.config)
    data = prepare_train(cfg, args.fraction)
    w = cfg["ml"]["window"]
    summary = data.summary()
    summary["windows"] = {k: len(data.dataset(k, w["length"], w["train_stride"] if k == "train" else w["eval_stride"]))
                          for k in SPLITS}
    print(json.dumps({k: v for k, v in summary.items() if k != "features"}, ensure_ascii=False, indent=2))
    if args.save_dir:
        data.pre.save(args.save_dir / "scaler.joblib")
        (args.save_dir / "preprocess.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
        print(f"[保存] {args.save_dir}")


if __name__ == "__main__":
    main()
