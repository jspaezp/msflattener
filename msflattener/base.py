from collections.abc import Generator

import polars as pl

# TODO should I move all of this to 32 bit?

SCHEMA_DDA = {
    "mz_values": pl.List(pl.Float64),
    "corrected_intensity_values": pl.List(pl.Float64),
    "mobility_values": pl.List(pl.Float64),
    "rt_values": pl.Float64,
    "quad_low_mz_values": pl.Float64,
    "quad_high_mz_values": pl.Float64,
    "precursor_mz_values": pl.Float64,
    "precursor_charge": pl.Int8,
    "precursor_intensity": pl.Int64,
}

SCHEMA_DIA = {
    "mz_values": pl.List(pl.Float64),
    "corrected_intensity_values": pl.List(pl.Float64),
    "mobility_values": pl.List(pl.Float64),
    "rt_values": pl.Float64,
    "quad_low_mz_values": pl.Float64,
    "quad_high_mz_values": pl.Float64,
}

YIELDING_FIELDS = [
    "rt_values",
    "quad_low_mz_values",
    "quad_high_mz_values",
    "precursor_mz_values",
    "precursor_charge",
    "precursor_intensity",
]


def yield_scans(df: pl.DataFrame) -> Generator[tuple[dict, list[dict]], None, None]:
    """Yield scans from a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to yield scans from.
        It needs to contain the following columns:
            mz_values: np.ndarray, nested
            corrected_intensity_values: np.ndarray, nested
            rt_values: np.ndarray
            quad_low_mz_values: np.ndarray
            quad_high_mz_values: np.ndarray

    Yields
    ------
    tuple[dict, list[dict]]
        A tuple with the parent scan and the children scans.
        Each scan is a dict with the following keys:
            mz_values: np.ndarray
            corrected_intensity_values: np.ndarray
            rt_values: float
            quad_low_mz_values: float
            quad_high_mz_values: float
            id: int

    """
    curr_parent = None
    curr_children = []
    for id, row in enumerate(
        df.sort(["rt_values", "precursor_mz_values"]).iter_rows(named=True)
    ):
        row["id"] = id

        # u, inv = np.unique(np.array(row["mz_values"]).round(2), return_inverse=True)
        # sums = np.zeros(len(u), dtype=np.float64)
        # np.add.at(sums, inv, np.array(row["corrected_intensity_values"], dtype=np.float64))
        # row["mz_values"] = u
        # row["corrected_intensity_values"] = sums

        # If the current row is a parent, and there are already children, yield
        if row["quad_low_mz_values"] < 0:
            if curr_parent is not None:
                yield curr_parent, curr_children
                curr_parent = row
                curr_children = []
            else:
                curr_parent = row

        else:
            curr_children.append(row)

    if curr_parent is not None:
        yield curr_parent, curr_children
