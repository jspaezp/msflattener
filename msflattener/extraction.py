from typing import Any

import numpy as np
from scipy.spatial import KDTree


class RTCircularBuffer:
    def __init__(self, max_rt_diff_keep) -> None:
        self.max_rt_diff_keep = max_rt_diff_keep
        self.buffer = []
        self.rts = []
        self.exrtas_buffer = []

    def append(self, element, rt_value, extras=None):
        self.add(element, rt_value, extras)
        self.remove_out_of_range()

    def add(self, element, rt_value, extras=None):
        self.buffer.append(element)
        self.rts.append(rt_value)
        self.exrtas_buffer.append(extras)

    def _out_of_range_indices(self):
        if not self.buffer:
            return False
        if (self.rts[-1] - self.rts[0]) > self.max_rt_diff_keep:
            return True
        else:
            return False

    def _elems_to_pop(self):
        elems = [self.buffer, self.rts, self.exrtas_buffer]

        yield from elems

    def remove_out_of_range(self):
        while self._out_of_range_indices():
            self.on_pop()
            for elem in self._elems_to_pop():
                elem.pop(0)

    def on_pop(self):
        """Abstract method of what needs to happen every time an element will be removed from the buffer."""
        pass


class TracingCircularBuffer(RTCircularBuffer):
    def __init__(self, max_rt, max_distances) -> None:
        super().__init__(2 * max_rt)
        self.center_max_rt = max_rt
        self.max_distances = max_distances

    def add(
        self, element: list[np.ndarray], rt_value: float, extras: dict[str, Any] = None
    ):
        if extras is None:
            extras = {}

        tree = KDTree(np.stack([e / y for e, y in zip(element, self.max_distances)]).T)
        neighboring_trees = self.trees_in_range(rt_value)
        num_elems_before = len(neighboring_trees)
        if len(neighboring_trees) > 0:
            neighbors = tree.query_ball_tree(neighboring_trees[-1], 1)
            out_neighbors = {}
            for i, k in enumerate(neighbors):
                if k:
                    out_neighbors[i] = set(k)
        else:
            out_neighbors = {}

        extras["tree"] = tree
        extras["num_neigh_before"] = num_elems_before
        extras["last_neighbors"] = out_neighbors
        super().add(element, rt_value, extras)

    def trees_in_range(self, rt):
        return [
            t["tree"]
            for t, r in zip(self.extras_buffer, self.rts)
            if rt - r < self.center_max_rt
        ]


# on pop ....
#     if len(self.buffer) > 0:
#         asdasda

# Calculate distances along the window ...
# Keep continous matches in both directions
# For every point, find the closest point in the other direction
# If the closest point is within a certain distance, keep it
# If the closest point is not within a certain distance, remove it
# Then expand using the indices in the next element.
