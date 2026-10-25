"""Per-row content hash: looks only at sensor values, not at timestamps or labels. Used to check whether files share the same data (M0)
and to assert that training windows contain no test rows (M1 tests)."""

import numpy as np
import pandas as pd


def hash_rows(features: pd.DataFrame) -> np.ndarray:
    """One uint64 hash per row. Everything is converted to float64 first: the same column may be parsed as int64 or float64 in different files,
    and without unifying, rows with identical values would get different hashes."""
    return pd.util.hash_pandas_object(features.astype(np.float64), index=False).to_numpy()


def matched_runs(hash_a: np.ndarray, hash_b: np.ndarray) -> list[tuple[int, int, int]]:
    """Find rows of a whose content also appears in b, and merge them into contiguous runs.

    Returns [(start index in a, start index in b, length)]: within a run the indices of a and b increase together (i.e. a block of data copied as a whole).
    Timestamps on both sides need not line up, so shifted partial overlaps are found too. For rows duplicated in b, the first occurrence is used.
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
