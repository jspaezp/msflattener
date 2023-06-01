from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import KDTree


@dataclass
class BidirectionalNeighbors:
    left_to_right: dict[int : set[int]]
    right_to_left: dict[int : set[int]]

    @classmethod
    def from_neighbor_list(cls, neighbor_list):
        left_to_right = {}
        right_to_left = {}
        for k, v in neighbor_list.items():
            left_to_right[k] = set(v)
            for n in v:
                right_to_left.setdefault(n, set()).add(k)
        return cls(left_to_right, right_to_left)

    def add_pairs(self, pairs: list[tuple[int, int]]):
        for p in pairs:
            self.add_pair(p)

    def add_pair(self, pair: tuple[int, int]):
        self.left_to_right.setdefault(pair[0], set()).add(pair[1])
        self.right_to_left.setdefault(pair[1], set()).add(pair[0])


class RTCircularBuffer:
    def __init__(self, max_rt_diff_keep) -> None:
        self.max_rt_diff_keep = max_rt_diff_keep
        self.buffer = []
        self.rts = []
        self.exrtas_buffer = []

    def append(self, element, rt_value, extras=None):
        self.add(element, rt_value, extras)
        return self.remove_out_of_range()

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

    def _elems_to_pop(self) -> Iterator:
        """Yields elements to pop.

        The definition of 'element' here is the instance attributes that grow
        when something is appended to the class itself.

        Returns a list of the elements from which the first element should be popped
        when the buffer is out of range.

        This method should be over-written if the child class accumulates more
        elements than the current class.
        """
        elems = [self.buffer, self.rts, self.exrtas_buffer]
        yield from elems

    def remove_out_of_range(self) -> list[list[Any]]:
        removed = []
        while self._out_of_range_indices():
            popped = []
            for elem in self._elems_to_pop():
                popped.append(elem.pop(0))
            removed.append(popped)

        return removed


class TreeCircularBuffer(RTCircularBuffer):
    def __init__(self, max_rt, max_distances) -> None:
        super().__init__(max_rt)
        self.max_distances = max_distances

    def add(
        self, element: list[np.ndarray], rt_value: float, extras: dict[str, Any] = None
    ):
        if extras is None:
            extras = {}

        tree = KDTree(np.stack([e / y for e, y in zip(element, self.max_distances)]).T)
        if self.exrtas_buffer:
            last_tree = self.exrtas_buffer[-1]["tree"]
            neighbors = tree.query_ball_tree(last_tree, 1)
            out_neighbors = BidirectionalNeighbors.from_neighbor_list(neighbors)
        else:
            out_neighbors = BidirectionalNeighbors({}, {})

        extras["tree"] = tree
        extras["last_neighbors"] = out_neighbors
        super().add(element, rt_value, extras)


class BufferWithCenter:
    """Buffer that keeps a center and distances before and after.

    Distance = 2
    -- past - c -future--
    [0, 1, 2] 3 [4, 5, 6] <- 7
    [1, 2, 3] 4 [5, 6, 7]



    """

    def __init__(self, max_rt, max_distances) -> None:
        self.past_buffer = RTCircularBuffer(max_rt)
        self.future_buffer = TreeCircularBuffer(
            max_rt_diff_keep=max_rt, max_distances=max_distances
        )
        self.current_elem = None
        self.current_extras = None
        self.current_rt = None
        self.max_distances = max_distances
        self.trace_queue = []

    def append(self, element, rt_value, extras=None):
        """Adds new elements to the future buffer.

        If the future buffer starts having elements that are out of range,
        They are moved to the center buffer and the center buffer is moved to
        the past buffer.

        """
        removed = self.future_buffer.append(element, rt_value, extras)
        for x in removed:
            element, rt_value, extras = x
            if self.current_elem is not None:
                self.trace_queue.extend(self.get_current_traces())
            self.past_buffer.append(
                self.current_elem, self.current_rt, self.current_extras
            )
            self.current_elem = element
            self.current_rt = rt_value
            self.current_extras = extras

    def get_current_traces(self) -> list[Trace]:
        # get elements that have a trace (defined as neighbor before and after)
        # subset the element arrays to have only those indices
        # make a kdtree with that subset
        # query that kdtree vs all other trees in range
        # integrate intensities.
        # subset for the ones where the apex is the current element (+- some tolerance ...)
        # output would be a list of intensities for every element with a trace.
        # expand the trace

        in_range_before = [
            i
            for i, x in enumerate(self.past_buffer.rts)
            if (self.current_rt - x) < self.max_rt
        ]
        in_range_after = [
            i
            for i, x in enumerate(self.future_buffer.rts)
            if (x - self.current_rt) < self.max_rt
        ]

        if not in_range_before or not in_range_after:
            return []

        has_trace = set.intersection(
            self.current_extras["last_neighbors"].left_to_right.keys(),
            self.future_buffer.exrtas_buffer[in_range_after[0]][
                "last_neighbors"
            ].right_to_left.keys(),
        )
        if not has_trace:
            return []

        temp_tree = KDTree(
            np.stack([e / y for e, y in zip(self.current_elem, self.max_distances)]).T
        )
        neighbors_before = [
            BidirectionalNeighbors.from_neighbor_list(
                temp_tree.query_ball_tree(x["tree"], 1)
            )
            for x in self.past_buffer.exrtas_buffer
        ]
        neighbors_after = [
            BidirectionalNeighbors.from_neighbor_list(
                temp_tree.query_ball_tree(x["tree"], 1)
            )
            for x in self.future_buffer.exrtas_buffer
        ]
        trace_length = len(neighbors_before) + len(neighbors_after) + 1
        center_index = len(neighbors_before)

        for t in has_trace:
            intensities = np.zeros(trace_length, dtype=np.float32)
            intensities[len(neighbors_before)] = self.current_elem[t]
            for i, x in enumerate(neighbors_before):
                if t in x:
                    extract_i = x[t]
                    extract_from = self.past_buffer.exrtas_buffer[i]["intensity"]
                    ci = extract_from[extract_i].sum().astype(np.float32)
                    intensities[len(neighbors_before) - i - 1] = ci
            for i, x in enumerate(neighbors_after):
                if t in x:
                    extract_i = x[t]
                    extract_from = self.past_buffer.exrtas_buffer[i]["intensity"]
                    ci = extract_from[extract_i].sum().astype(np.float32)
                    intensities[len(neighbors_before) + i + 1] = ci

            Trace(
                rts_in_seconds=[],
                intensities=intensities,
                mz=self.current_elem[0][t],
                extras={},
                center=center_index,
            )


@dataclass
class Trace:
    rts_in_seconds: np.ndarray
    intensities: np.ndarray
    mz: float
    extras: dict[str, Any]
    center: int

    def __len__(self) -> int:
        return len(self.intensities)


# on pop ....
#     if len(self.buffer) > 0:
#         asdasda

# Calculate distances along the window ...
# Keep continous matches in both directions
# For every point, find the closest point in the other direction
# If the closest point is within a certain distance, keep it
# If the closest point is not within a certain distance, remove it
# Then expand using the indices in the next element.
