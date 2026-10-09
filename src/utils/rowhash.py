"""逐行内容哈希：只看传感器取值，不看时间戳和标签。用于检查文件之间是否含有同一份数据（M0），
以及断言训练窗口中不含 test 的行（M1 测试）。"""

import numpy as np
import pandas as pd


def hash_rows(features: pd.DataFrame) -> np.ndarray:
    """每行一个 uint64 哈希。先统一转成 float64：同一列在不同文件中可能被解析成 int64 或 float64，
    不统一的话取值相同的行会得到不同的哈希。"""
    return pd.util.hash_pandas_object(features.astype(np.float64), index=False).to_numpy()


def matched_runs(hash_a: np.ndarray, hash_b: np.ndarray) -> list[tuple[int, int, int]]:
    """找出 a 中内容也出现在 b 中的行，并合并为连续段。

    返回 [(a 中起始下标, b 中起始下标, 长度)]：段内 a、b 的下标同步递增（即一段被整体复制的数据）。
    不要求两边时间戳对齐，因此能发现错位的部分重叠。b 中内容重复的行取第一次出现的位置。
    """
    first_b = pd.Series(np.arange(len(hash_b)), index=hash_b)
    first_b = first_b[~first_b.index.duplicated()]
    ia = np.flatnonzero(np.isin(hash_a, first_b.index.to_numpy()))
    if len(ia) == 0:
        return []
    ib = first_b.loc[hash_a[ia]].to_numpy()
    breaks = np.flatnonzero((np.diff(ia) != 1) | (np.diff(ib) != 1)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(ia)]))
    return [(int(ia[s]), int(ib[s]), int(e - s)) for s, e in zip(starts, ends)]
