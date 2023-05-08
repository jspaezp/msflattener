import numpy as np


def _get_breaks(array: np.ndarray, min_diff: float = 0) -> np.ndarray:
    return np.where(np.abs(np.diff(array)) > min_diff)[0] + 1


def _get_breaks_multi(*args: np.ndarray) -> np.ndarray:
    if len({len(x) for x in args}) != 1:
        lens = [len(x) for x in args]
        raise ValueError(
            "All arrays need to be the same length, got lengths: " + str(lens)
        )
    out = np.unique(np.concatenate([_get_breaks(x) for x in args]))
    out = np.concatenate([np.array([0]), out, np.array([len(args[0])])])
    out.sort()
    return out
