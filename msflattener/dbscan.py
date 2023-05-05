import numpy as np
from scipy.spatial import KDTree

def _find_neighbors_sorted(array, max_distance):
    assert np.all(np.diff(array) >= 0), "Array not sorted"
    neighbors = {i:{'neighbors': set(), 'count': 0} for i in range(len(array))}
    ii = 0
    for ix, x in enumerate(array):
        ii = np.searchsorted(array, x - max_distance, side='left', sorter=None)
        ij = np.searchsorted(array, x + max_distance, side='right', sorter=None)
        neighbors[ix]['neighbors'] = set(range(ii, ij))
        neighbors[ix]['count'] = ij - ii

    return neighbors

def _simplify_neighbors(neighbors: dict[int: set[int]], order: list[int], expansion_iters: int):
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

def dbscan_1d(array, max_distance, min_neighbors, order, expansion_iters=10) -> dict[int:set[int]]:
    assert np.all(np.diff(array) >= 0), "Array not sorted"
    # neighbors = _find_neighbors_sorted(array=array, max_distance=max_distance)
    # filtered_neighbors = {k:v["neighbors"] for k,v in neighbors.items() if v['count'] >= min_neighbors}
    tree = KDTree(np.expand_dims(array/max_distance, axis=-1))
    neighbors = tree.query_pairs(1)
    out = {}
    for k, v in neighbors:
        out.setdefault(k, set()).add(v)
        out.setdefault(v, set()).add(k)

    filtered_neighbors = {k:v for k, v in out.items() if len(v) >= min_neighbors}

    return _simplify_neighbors(filtered_neighbors, order=order, expansion_iters=expansion_iters)

# @profile
def dbscan_nd(values_list: list[np.array], value_max_dists: list[float], min_neighbors: int, order: list[int], expansion_iters: int=50):
    tree = KDTree(np.stack([x/y for  x,y in zip(values_list, value_max_dists)]).T)
    comb_neighs2 = tree.query_pairs(1)
    out = {}
    for k, v in comb_neighs2:
        out.setdefault(k, set()).add(v)
        out.setdefault(v, set()).add(k)
    
    combined_neighbors = {k:v for k, v in out.items() if len(v) >= min_neighbors}

    return _simplify_neighbors(combined_neighbors, order=order, expansion_iters=expansion_iters)

def dbscan_collapse(values: np.array, intensities: np.array, min_neighbors: int, value_max_dist: float):
    sorting = np.argsort(values)

    arr = values[sorting]
    intensities = intensities[sorting]

    order = np.argsort(intensities)
    neighbors = dbscan_1d(arr, value_max_dist, min_neighbors=min_neighbors, order=order)
    fin_intensities = np.array([intensities[list(x)].sum() for x in neighbors.values()])
    fin_values = arr[list(neighbors.keys())]
    return fin_values, fin_intensities

# @profile
def dbscan_collapse_multi(values_list: list[np.array], value_max_dists: list[float], intensities: np.array, min_neighbors: int, expansion_iters: int=10):
    assert len(values_list) == len(value_max_dists)
    assert all(len(values_list[0]) == len(x) for x in values_list)

    order = np.argsort(intensities)
    combined_neighbors = dbscan_nd(
        values_list=values_list,
        value_max_dists=value_max_dists,
        min_neighbors=min_neighbors,
        order=order,
        expansion_iters=expansion_iters)

    fin_intensities = np.array([intensities[list(x)].sum() for x in combined_neighbors.values()])
    fin_values_list = []
    for arr in values_list:
        tmp = np.array([(intensities[list(x)] * arr[list(x)]).sum() / y for x, y in zip(combined_neighbors.values(), fin_intensities)])
        fin_values_list.append(tmp)
    return fin_values_list, fin_intensities

