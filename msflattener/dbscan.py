import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


def _find_neighbors_sorted(array, max_distance):
    assert np.all(np.diff(array) >= 0), "Array not sorted"
    neighbors = {i: {"neighbors": set(), "count": 0} for i in range(len(array))}
    ii = 0
    for ix, x in enumerate(array):
        ii = np.searchsorted(array, x - max_distance, side="left", sorter=None)
        ij = np.searchsorted(array, x + max_distance, side="right", sorter=None)
        neighbors[ix]["neighbors"] = set(range(ii, ij))
        neighbors[ix]["count"] = ij - ii

    return neighbors


def _simplify_neighbors(
    neighbors: dict[int : set[int]], order: list[int], expansion_iters: int
):
    used = set()
    out_neighbors = {}
    for o in order:
        if o in neighbors:
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
                used.update(out_neighbors[o])

    return out_neighbors


def dbscan_1d(
    array: np.array,
    max_distance: float,
    min_neighbors: int,
    order: NDArray[np.int64],
    expansion_iters: int = 10,
) -> dict[int : set[int]]:
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
    count_only_values_list: list[np.array] | None = None,
    expansion_iters: int = 50,
):
    tree = KDTree(
        np.stack([x / y for x, y in zip(values_list, value_max_dists)]).T,
        leafsize=2 * min_neighbors,
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
        if count_only_values_list is not None:
            combined_neighbors = {k: {"count": len(v), "v": v} for k, v in out.items()}
            other_tree = KDTree(
                np.stack(
                    [x / y for x, y in zip(count_only_values_list, value_max_dists)]
                ).T,
                leafsize=10 * min_neighbors,
            )
            comb_neighs2 = tree.query_ball_tree(other_tree, 1)
            for k, v in enumerate(comb_neighs2):
                combined_neighbors.setdefault(k, {"count": 1, "v": {k}})[
                    "count"
                ] += len(v)

            combined_neighbors = {
                k: v["v"]
                for k, v in combined_neighbors.items()
                if v["count"] >= min_neighbors
            }
        else:
            combined_neighbors = {
                k: v for k, v in out.items() if len(v) >= min_neighbors
            }

    return _simplify_neighbors(
        combined_neighbors, order=order, expansion_iters=expansion_iters
    )


def dbscan_collapse(
    values: np.array,
    intensities: np.array,
    min_neighbors: int,
    value_max_dist: float,
    expansion_iters: int = 10,
):
    sorting = np.argsort(values)

    arr = values[sorting]
    intensities = intensities[sorting]

    order = np.argsort(intensities)
    neighbors = dbscan_1d(
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
    count_only_values_list: None | list[np.array] = None,
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

    order = np.argsort(intensities)
    combined_neighbors = dbscan_nd(
        values_list=values_list,
        value_max_dists=value_max_dists,
        min_neighbors=min_neighbors,
        order=order,
        expansion_iters=expansion_iters,
        count_only_values_list=count_only_values_list,
    )

    fin_intensities = np.array(
        [intensities[list(x)].sum() for x in combined_neighbors.values()]
    )
    fin_values_list = []
    for arr in values_list:
        tmp = np.array(
            [
                (intensities[list(x)] * arr[list(x)]).sum() / y
                for x, y in zip(combined_neighbors.values(), fin_intensities)
            ]
        )
        fin_values_list.append(tmp)
    return fin_values_list, fin_intensities
