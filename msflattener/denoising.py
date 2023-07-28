import math

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import KDTree
from tqdm.auto import tqdm

from .base import SCHEMA_DIA
from .dbscan import dbscan_collapse_multi


def simple_denoise(df: pl.DataFrame, mz_tol: float = 0.01, mobility_tol: float = 0.005):
    """Extremely simple denoising of data.

    This function will remove peaks that do not have aneighbor in a contiguous scan
    within the provided tolerance.
    """
    logger.debug("Starting simple denoising")
    assert set(df.columns) == set(
        SCHEMA_DIA.keys()
    ), f"Invalid DataFrame, required columns: {set(SCHEMA_DIA.keys())}"
    SORTING_COLS = ["rt_values"]  # noqa
    GROUPING_COLS = ["quad_low_mz_values", "quad_high_mz_values"]  # noqa
    compressions = []

    def _make_kdtree(mzs, mobilities):
        leaves = 2 * math.ceil(math.log10(len(mzs) + 1))
        kdtree_arr = np.stack(
            [
                mzs / mz_tol,
                mobilities / mobility_tol,
            ]
        ).T

        # I made a lot of testing and the scipy implementation
        # is a lot faster in query times when the tree is large.
        kdt = KDTree(kdtree_arr, balanced_tree=False, leafsize=leaves)
        return kdt, kdtree_arr

    progbar = tqdm(total=len(df), desc="Simple Denoising")
    out_df_chunks = []

    # Iterate over each group
    for group, sub_df in df.groupby(GROUPING_COLS):
        sub_df = sub_df.sort(SORTING_COLS)

        # For each scan generate a kdtree,
        # then keep only points that have a neighbor within
        # the given tolerance

        last_kdt = None
        curr_kdt = None

        curr_kdt_arr = None

        curr_mzs = np.array(sub_df["mz_values"][0])
        curr_mobilities = np.array(sub_df["mobility_values"][0])
        curr_intensities = np.array(sub_df["corrected_intensity_values"][0])

        next_kdt, next_kdt_arr = _make_kdtree(curr_mzs, curr_mobilities)

        keep_vals = {
            "mz_values": [],
            "mobility_values": [],
            "corrected_intensity_values": [],
        }
        for i in range(len(sub_df)):
            progbar.update(1)
            is_last = i == (len(sub_df) - 1)
            is_first = i == 0
            if (is_last) or (is_first):
                keep_vals["mz_values"].append(curr_mzs)
                keep_vals["mobility_values"].append(curr_mobilities)
                keep_vals["corrected_intensity_values"].append(curr_intensities)

                compressions.append(1)
                if is_last:
                    continue

            next_mzs = np.array(sub_df["mz_values"][i + 1])
            next_mobilities = np.array(sub_df["mobility_values"][i + 1])
            next_intensities = np.array(sub_df["corrected_intensity_values"][i + 1])

            kdt, kdt_arr = _make_kdtree(next_mzs, next_mobilities)

            last_kdt, curr_kdt, next_kdt = curr_kdt, next_kdt, kdt
            _, curr_kdt_arr, next_kdt_arr = (
                curr_kdt_arr,
                next_kdt_arr,
                kdt_arr,
            )

            if not is_first:
                bundles = curr_kdt.query_ball_tree(last_kdt, r=1)
                # Check if the lists are non-empty
                # Chacking len(x) > 0 could be more readable but
                # it might also be slower ...
                indices_keep_last = [i for i, x in enumerate(bundles) if x]

            if next_kdt is not None:
                # Time both approaches and see which is faster
                bundles = curr_kdt.query_ball_tree(next_kdt, r=1)
                indices_keep_next = [i for i, x in enumerate(bundles) if x]

            else:
                logger.error("I am here!!")
                indices_keep_next = []

            if not is_first:
                indices_keep = list(set(indices_keep_last + indices_keep_next))
                if len(indices_keep) == 0:
                    logger.warning(
                        f"No points were kept (out of {len(curr_mzs)})"
                        f" on {i}th/{len(sub_df)} iteration"
                    )
                    logger.debug(f"RT = {sub_df['rt_values'][i]} and group = {group}")

                keep_vals["mz_values"].append(curr_mzs[indices_keep])
                keep_vals["mobility_values"].append(curr_mobilities[indices_keep])
                keep_vals["corrected_intensity_values"].append(
                    curr_intensities[indices_keep]
                )

                compressions.append(len(indices_keep) / len(curr_mzs))

            curr_mzs = next_mzs
            curr_mobilities = next_mobilities
            curr_intensities = next_intensities

        df_chunk = sub_df.with_columns(
            mz_values=pl.Series("mz_values", keep_vals["mz_values"]),
            mobility_values=pl.Series("mobility_values", keep_vals["mobility_values"]),
            corrected_intensity_values=pl.Series(
                "corrected_intensity_values", keep_vals["corrected_intensity_values"]
            ),
        )
        out_df_chunks.append(df_chunk)

        # Log the mean compression
        logger.debug(
            f"Mean compression for {group} (peaks after/peaks before):"
            f" {np.mean(compressions):.2f}"
        )

    return pl.concat(out_df_chunks)


def dbscan_df(
    df: pl.DataFrame,
    min_neighbors: int = 1,
    expansion_iters: int = 10,
    mz_tol: float = 0.01,
    mobility_tol: float = 0.01,
):
    """Denoise a DataFrame using DBSCAN.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to denoise.
        It needs to contain the colums specified in
        `msflattener.base.SCHEMA_DIA`.
    min_neighbors : int, optional
        The minimum number of neighbors to keep a point, by default 1
    expansion_iters : int, optional
        The number of iterations to expand the neighbors, by default 10
    mz_tol : float, optional
        The m/z tolerance to use, by default 0.01
    mobility_tol : float, optional
        The mobility tolerance to use, by default 0.01

    Returns
    -------
    pl.DataFrame
        The denoised DataFrame.
    """
    assert set(df.columns) == set(
        SCHEMA_DIA.keys()
    ), f"Invalid DataFrame, required columns: {set(SCHEMA_DIA.keys())}"

    keep_vals = {
        "mz_values": [],
        "mobility_values": [],
        "corrected_intensity_values": [],
    }
    compressions = []

    for row in tqdm(df.iter_rows(named=True), desc="DBSCAN Denoising", miniters=1):
        if len(row["mz_values"]) < min_neighbors:
            keep_vals["mz_values"].append(np.array(row["mz_values"]))
            keep_vals["mobility_values"].append(np.array(row["mobility_values"]))
            keep_vals["corrected_intensity_values"].append(
                np.array(row["corrected_intensity_values"])
            )
            continue
        (mzs, imss), intensities = dbscan_collapse_multi(
            values_list=[np.array(row["mz_values"]), np.array(row["mobility_values"])],
            value_max_dists=[mz_tol, mobility_tol],
            intensities=np.array(row["corrected_intensity_values"]),
            expansion_iters=expansion_iters,
            min_neighbors=min_neighbors,
        )
        compressions.append(len(mzs) / len(row["mz_values"]))
        keep_vals["mz_values"].append(mzs)
        keep_vals["mobility_values"].append(imss)
        keep_vals["corrected_intensity_values"].append(intensities)

    out = df.with_columns(
        mz_values=pl.Series("mz_values", keep_vals["mz_values"]),
        mobility_values=pl.Series("mobility_values", keep_vals["mobility_values"]),
        corrected_intensity_values=pl.Series(
            "corrected_intensity_values", keep_vals["corrected_intensity_values"]
        ),
    )
    logger.debug(
        f"Mean compression (peaks after/peaks before): {np.mean(compressions):.2f}"
    )
    return out


if __name__ == "__main__":
    from pathlib import Path

    from msflattener.bruker import get_timstof_data

    # datafile = "/Users/sebastianpaez/git/msflattener2/tests/data/230711_idleflow_400-1000mz_25mz_diaPasef_10sec.d"
    datafile = "/Users/sebastianpaez/git/msflattener2/tests/raw_data/7min/7min/20220302_tims1_nElute_8cm_DOl_Phospho_7min_rep1_Slot1-94_1_1811.d"
    datafile = Path(datafile)
    out = get_timstof_data(datafile, progbar=False, centroid=False, safe=True)
    print(out)
    denoised = simple_denoise(out)
    print(denoised)

    dbscanned = dbscan_df(denoised)
    print(dbscanned)
