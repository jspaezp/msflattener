
import numpy as np

def _get_breaks(array: np.ndarray, min_diff=0):
    return np.where(np.abs(np.diff(array)) > min_diff)[0] + 1


def _get_breaks_multi(*args):
    if not all(len(x) for x in args):
        raise ValueError("All arrays need to be the same length")
    out = np.unique(np.concatenate([_get_breaks(x) for x in args]))
    out = np.concatenate([np.array([0]), out, np.array([len(args[0])])])
    out.sort()
    return out