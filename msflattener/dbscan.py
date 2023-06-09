from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


def _simplify_neighbors(
    neighbors: dict[int : set[int]], order: list[int], expansion_iters: int
):
    used = set()
    out_neighbors = {}
    for o in order:
        if o in neighbors and o not in used:
            for _ in range(expansion_iters):
                new_neighbors = neighbors[o].copy()
                start_len = len(new_neighbors)
                for n in neighbors.pop(o):
                    if n == o:
                        continue
                    elif n in neighbors:
                        new_neighbors.update(neighbors.pop(n))

                new_neighbors = new_neighbors - used
                neighbors[o] = new_neighbors
                if len(new_neighbors) == start_len:
                    break
            if len(neighbors[o]):
                out_neighbors[o] = neighbors.pop(o)
                out_neighbors[o].add(o)
                used.update(out_neighbors[o])

    return out_neighbors, used


def dbscan_1d(
    array: np.array,
    max_distance: float,
    min_neighbors: int,
    order: NDArray[np.int64],
    expansion_iters: int = 10,
) -> dict[int : set[int]]:
    """Find neighbors in a 1D array."""
    # TODO optimize this, there has to be a more efficient way to
    # do this in 1d ... Right now it is not within the scope of
    # the project to do so.
    # maybe something like `np.searchsorted(array, array + max_distance)`.

    assert np.all(np.diff(array) >= 0), "Array not sorted"
    # The KDtree replaces the following code, but is slower
    # neighbors = _find_neighbors_sorted(array=array, max_distance=max_distance)
    # filtered_neighbors = {k:v["neighbors"]
    #  for k,v in neighbors.items() if v['count'] >= min_neighbors}
    tree = KDTree(
        np.expand_dims(array / max_distance, axis=-1), leafsize=2 * min_neighbors
    )
    neighbors = tree.query_pairs(1)
    out = {}
    for k, v in neighbors:
        out.setdefault(k, {k}).add(v)
        out.setdefault(v, {v}).add(k)

    if min_neighbors == 1:
        for i in range(len(array)):
            out.setdefault(i, {i})
        filtered_neighbors = out
    else:
        filtered_neighbors = {k: v for k, v in out.items() if len(v) >= min_neighbors}

    return _simplify_neighbors(
        filtered_neighbors, order=order, expansion_iters=expansion_iters
    )


# @profile
def dbscan_nd(
    values_list: list[np.array],
    value_max_dists: list[float],
    min_neighbors: int,
    order: list[int],
    expansion_iters: int = 50,
) -> dict[int : set[int]]:
    """Find neighbors in n-dimensional space."""
    nleaves = max(2 * min_neighbors, math.ceil(math.log10(len(values_list[0]))))
    obs = np.stack([x / y for x, y in zip(values_list, value_max_dists)]).T
    tree = KDTree(
        obs,
        leafsize=nleaves,
    )
    comb_neighs2 = tree.query_pairs(1)

    out = {}
    for k, v in comb_neighs2:
        out.setdefault(k, {k}).add(v)
        out.setdefault(v, {v}).add(k)

    if min_neighbors == 1:
        for i in range(len(order)):
            out.setdefault(i, {i})
        combined_neighbors = out
    else:
        combined_neighbors = {k: v for k, v in out.items() if len(v) >= min_neighbors}

    s_out, used = _simplify_neighbors(
        combined_neighbors, order=order, expansion_iters=expansion_iters
    )

    return s_out, used


def dbscan_collapse(
    values: np.array,
    intensities: np.array,
    min_neighbors: int,
    value_max_dist: float,
    expansion_iters: int = 10,
) -> tuple[np.array, np.array]:
    """Collapses intensities by clustering them based on values."""
    sorting = np.argsort(values)

    arr = values[sorting]
    intensities = intensities[sorting]

    order = np.argsort(-intensities.astype(np.float64))
    neighbors, _used = dbscan_1d(
        arr,
        value_max_dist,
        min_neighbors=min_neighbors,
        order=order,
        expansion_iters=expansion_iters,
    )
    fin_intensities = np.array([intensities[list(x)].sum() for x in neighbors.values()])
    fin_values = np.array(
        [
            (intensities[list(x)] * arr[list(x)]).sum() / y
            for x, y in zip(neighbors.values(), fin_intensities)
        ]
    )
    return fin_values, fin_intensities


# @profile
def dbscan_collapse_multi(
    values_list: list[np.array],
    value_max_dists: list[float],
    intensities: np.array,
    min_neighbors: int,
    expansion_iters: int = 10,
) -> tuple[list[np.array], np.array]:
    """Collapse a set of values based on a set of distances.

    Parameters
    ----------
    values_list : list[np.array]
        List of arrays to collapse
    value_max_dists : list[float]
        List of maximum distances for each array
    intensities : np.array
        Intensities for each value
    min_neighbors : int
        Minimum number of neighbors to collapse
    expansion_iters : int, optional
        Number of iterations to expand the neighbors, by default 10
    count_only_values_list : None | list[np.array], optional
        List of arrays to count neighbors, by default None
        values in these arrays will vount as neighbors but will
        not be used during the expansion process or count towards
        the final intensity.
    """
    assert len(values_list) == len(value_max_dists)
    assert all(len(values_list[0]) == len(x) for x in values_list)

    order = np.argsort(-intensities.astype(np.float64))
    combined_neighbors, used = dbscan_nd(
        values_list=values_list,
        value_max_dists=value_max_dists,
        min_neighbors=min_neighbors,
        order=order,
        expansion_iters=expansion_iters,
    )

    fin_intensities = np.array(
        [intensities[list(x)].sum() for x in combined_neighbors.values()]
    )
    unused_vals = np.setdiff1d(order, np.array(list(used)))
    fin_values_list = []
    for arr in values_list:
        tmp = np.array(
            [
                (intensities[list(x)] * arr[list(x)]).sum() / y
                for x, y in zip(combined_neighbors.values(), fin_intensities)
            ]
        )
        tmp = np.concatenate([tmp, arr[unused_vals].astype(tmp.dtype)], axis=0)
        fin_values_list.append(tmp)

    fin_intensities = np.concatenate(
        [fin_intensities, intensities[unused_vals].astype(fin_intensities.dtype)],
        axis=0,
    )
    return fin_values_list, fin_intensities
