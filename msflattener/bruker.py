from __future__ import annotations

import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import polars as pl
from alphatims import bruker
from loguru import logger
from numpy.typing import DTypeLike
from tqdm.auto import tqdm

from .base import SCHEMA_DDA, SCHEMA_DIA, YIELDING_FIELDS
from .dbscan import dbscan_collapse_multi


def __get_nesting_level(x):
    """Get the nesting level of a nested list or array.

    This is meant to be an internal function.

    Notes
    -----
        This function is recursive and does not check all the
        elements of the list or array. It only checks the first
        element of each sub-array.

    Examples
    --------
        >>> x = [[1, 2, 3], [4, 5, 6]]
        >>> __get_nesting_level(x)
        2
        >>> x = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        >>> __get_nesting_level(x)
        3
    """
    if hasattr(x, "__len__"):
        return 1 + __get_nesting_level(x[0])
    return 0


def __iter_dict_arrays(x: dict[str, np.ndarray]):
    """Iterate over a dictionary of arrays, yielding dictionaries of values.

    This is meant to be an internal function.

    Examples
    --------
        >>> x = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
        >>> for y in __iter_dict_arrays(x):
        ...     print(y)
        {'a': 1, 'b': 4}
        {'a': 2, 'b': 5}
        {'a': 3, 'b': 6}
    """
    assert isinstance(x, dict)
    lengths = {k: len(v) for k, v in x.items()}
    assert (
        len({len(y) for y in x.values()}) == 1
    ), f"All elements should be the same length, got: {lengths}"

    for i in range(len(x[list(x.keys())[0]])):
        yield {k: v[i] for k, v in x.items()}


def __count_chunks(x: dict[str, np.ndarray], breaking_names):
    # Counts the number of times the values in breaking_names change
    # in the dictionary of arrays x.

    diffs = {k: np.abs(np.diff(v)) for k, v in x.items() if k in breaking_names}
    # Since changes are not only positive, we make them abs values
    # then get the positions where they are not zero

    non_zeros = {k: np.where(v != 0)[0] for k, v in diffs.items()}
    # then remove the duplicates

    non_zeros = np.unique(np.concatenate(list(non_zeros.values())))
    # then count the number of changes

    return len(non_zeros)


def _iter_timstof_data(timstof_file: bruker.TimsTOF, progbar=True, safe=False):
    boundaries = [
        (start, end)
        for start, end in zip(
            timstof_file.push_indptr[:-1], timstof_file.push_indptr[1:]
        )
        if end - start >= 1
    ]
    # Note: I used to have a check here to see if end - start >= min_peaks
    # But it would check that the minimum number of peaks are present
    # per IMS push, not per scan. Which can be problematic.
    # remember ... in alphatims lingo, a scan is a full IMS ramp,
    # whilst a push is a single acquisition.
    contexts = timstof_file.convert_from_indices(
        raw_indices=[x[0] for x in boundaries],
        return_rt_values=True,
        return_quad_mz_values=True,
        return_mobility_values=True,
        return_precursor_indices=True,
        raw_indices_sorted=True,
    )
    # >>> pl.DataFrame(contexts)
    # shape: (7_447_578, 5)
    # ┌───────────────┬─────────────┬──────────────┬──────────────┬──────────────┐
    # │ precursor_ind ┆ rt_values   ┆ mobility_val ┆ quad_low_mz_ ┆ quad_high_mz │
    # │ ices          ┆ ---         ┆ ues          ┆ values       ┆ _values      │
    # │ ---           ┆ f32         ┆ ---          ┆ ---          ┆ ---          │
    # │ i64           ┆             ┆ f32          ┆ f32          ┆ f32          │
    # ╞═══════════════╪═════════════╪══════════════╪══════════════╪══════════════╡
    # │ 0             ┆ 0.640523    ┆ 1.371029     ┆ -1.0         ┆ -1.0         │
    # │ 0             ┆ 0.640523    ┆ 1.36996      ┆ -1.0         ┆ -1.0         │
    # │ 0             ┆ 0.640523    ┆ 1.36889      ┆ -1.0         ┆ -1.0         │
    # │ 0             ┆ 0.640523    ┆ 1.367821     ┆ -1.0         ┆ -1.0         │
    # │ …             ┆ …           ┆ …            ┆ …            ┆ …            │
    # │ 2             ┆ 1259.964782 ┆ 0.930313     ┆ 625.0        ┆ 650.0        │
    # │ 2             ┆ 1259.964782 ┆ 0.925969     ┆ 625.0        ┆ 650.0        │
    # │ 2             ┆ 1259.964782 ┆ 0.804063     ┆ 425.0        ┆ 450.0        │
    # │ 2             ┆ 1259.964782 ┆ 0.784422     ┆ 425.0        ┆ 450.0        │
    # └───────────────┴─────────────┴──────────────┴──────────────┴──────────────┘

    if safe:
        context2 = timstof_file.convert_from_indices(
            raw_indices=[x[1] - 1 for x in boundaries],
            return_rt_values=True,
            return_quad_mz_values=True,
            return_mobility_values=True,
            return_precursor_indices=True,
            raw_indices_sorted=True,
        )
        for k, v in context2.items():
            if not np.all(v == contexts[k]):
                raise ValueError(
                    f"Not all fields in the timstof context share values for {k} "
                )

    num_pushes = len(boundaries)
    num_chunks = __count_chunks(contexts, ["precursor_indices"])
    logger.info(f"Found {num_chunks} chunks in {num_pushes} pushes")

    my_iter = zip(boundaries, __iter_dict_arrays(contexts))
    my_progbar = tqdm(
        disable=not progbar,
        desc="Quad/rt groups",
        total=num_chunks,
        mininterval=0.2,
        maxinterval=5,
    )
    chunk_out = {}
    last_rt = None
    last_quad_high = None
    last_quad_low = None

    for (start, end), context in my_iter:
        query_range = range(start, end)

        out = timstof_file.convert_from_indices(
            raw_indices=query_range,
            return_corrected_intensity_values=True,
            return_mz_values=True,
            raw_indices_sorted=True,
        )
        out = {k: v.astype(np.float32) for k, v in out.items()}
        out["query_range"] = query_range
        for k, v in context.items():
            out[k] = v

        if timstof_file.acquisition_mode in {"ddaPASEF", "noPASEF"}:
            if context["precursor_indices"] == 0:
                out["precursor_mz_values"] = -1.0
                out["precursor_charge"] = -1
                out["precursor_intensity"] = -1
            else:
                context_precursor = timstof_file._precursors.iloc[
                    context["precursor_indices"] - 1
                ]
                prec_mz = context_precursor.MonoisotopicMz
                charge = context_precursor.Charge
                prec_intensity = context_precursor.Intensity
                out["precursor_intensity"] = prec_intensity
                if np.isnan(prec_mz):
                    out["precursor_mz_values"] = context_precursor.LargestPeakMz
                    # TODO: handle better missing charge states
                    out["precursor_charge"] = 2
                else:
                    out["precursor_mz_values"] = prec_mz
                    out["precursor_charge"] = charge

        if last_rt is not None:
            any_change = (
                last_rt != out["rt_values"]
                or last_quad_high != out["quad_high_mz_values"]
                or last_quad_low != out["quad_low_mz_values"]
            )
            if any_change:
                my_progbar.update(1)
                yield chunk_out
                chunk_out = {}

        for k, v in out.items():
            if k in YIELDING_FIELDS:
                if k in chunk_out:
                    assert np.all(chunk_out[k] == v)
                chunk_out[k] = v
            else:
                chunk_out.setdefault(k, []).append(v)

        last_rt = out["rt_values"]
        last_quad_high = out["quad_high_mz_values"]
        last_quad_low = out["quad_low_mz_values"]

    if chunk_out:
        my_progbar.update(1)
        yield chunk_out


def __iter_timstof_data_fractioned(
    timstof_file: bruker.TimsTOF,
    progbar=True,
    safe=False,
    pushes_per_fraction=20_000,
):
    pass


def __expand_like(arr: np.ndarray, like_arr: np.ndarray, dtype: DTypeLike = np.float32):
    """Expand an array to be the same length as another array.

    Expands an array to be the same length as another array,
    repeating each value in the first array.

    Parameters
    ----------
    arr : np.ndarray
        The array to expand
    like_arr : np.ndarray
        The array to match the length of
    dtype : np.dtype, optional
        The dtype of the output array, by default np.float32

    Returns
    -------
    np.ndarray
        The expanded array

    Examples
    --------
    >>> __expand_like([1, 2], [[1, 2, 3], [4, 5]])
    array([1., 1., 1., 2., 2.], dtype=float32)
    """
    out = np.concatenate(
        [np.full_like(y, fill_value=ai, dtype=dtype) for ai, y in zip(arr, like_arr)]
    )
    return out


def __unnest_chunk(chunk_dict: dict) -> dict | None:
    """Unnest a chunk of data from the TimsTOF file.

    Unnests a chunk of data from the TimsTOF file, expanding
    the arrays to be the same length as the m/z array.

    Meant to be an internal function.
    """
    mzs = np.concatenate(chunk_dict["mz_values"])
    if len(mzs) < 1:
        return
    intensities = np.concatenate(chunk_dict["corrected_intensity_values"])
    imss = __expand_like(
        chunk_dict["mobility_values"], chunk_dict["mz_values"], dtype=np.float32
    )
    if "precursor_charge" in chunk_dict:
        if isinstance(chunk_dict["precursor_charge"], int):
            pass
        elif isinstance(chunk_dict["precursor_charge"], float):
            pass
        else:
            chunk_dict["precursor_charge"] = __expand_like(
                chunk_dict["precursor_charge"], chunk_dict["mz_values"], np.int32
            )
            chunk_dict["precursor_intensity"] = __expand_like(
                chunk_dict["precursor_intensity"], chunk_dict["mz_values"], np.float32
            )

    chunk_dict["mz_values"] = mzs
    chunk_dict["mobility_values"] = imss
    chunk_dict["corrected_intensity_values"] = intensities

    return chunk_dict


def __clean_dangling_precursor_scans(df: pl.DataFrame):
    """Clean dangling precursor scans.

    I stilld o not understand why BUT there are many instances where consequent MS1
    Scans in DIA data have significantly less points than the other scans. This
    function removes those scans.

    In my experience this is noticed as scans with 10-50 peaks after a scan that
    might have 200,000.

    I believe it is caused when a dangling section of the IMS dimension is not scanned
    and the instrument does not use the quad filter for it.

    It does not seem to occur in data from out TimsTOF SCP, but it might
    be a method rather than an instrument issue.
    """
    logger.debug("Cleaning dangling precursor scans")

    non_ms1_df = df.filter(pl.col("quad_low_mz_values") > 0)
    ms1_df = df.filter(pl.col("quad_low_mz_values") < 0)
    initial_len = len(ms1_df)

    # For ms1 scans, sort by rt_values
    # Count number of mz_values in each scan
    # and calculate the ratio vs the next, then remove the ones
    # that are less than 0.1 of the next scan.
    # This is done twice in case there are 2 consecutive dangling scans.
    ms1_df = ms1_df.sort("rt_values").with_columns(
        n_mz_values=pl.col("mz_values").apply(lambda x: len(x))
    )
    ms1_df = ms1_df.with_columns(
        n_mz_values_next=pl.col("n_mz_values").shift(-1),
    )

    def __shift_and_filter(df):
        df = df.with_columns(
            n_mz_values_next=pl.col("n_mz_values").shift(-1),
        )
        df = df.with_columns(
            n_mz_values_ratio=pl.col("n_mz_values") / pl.col("n_mz_values_next"),
        )
        df = df.filter(
            (pl.col("n_mz_values_ratio") > 0.1)
            | (pl.col("n_mz_values_ratio").is_null())
        )
        return df

    curr_len = initial_len
    last_len = len(ms1_df)
    max_iter = 10
    num_iter = 0
    while (curr_len != last_len) or (num_iter == 0):
        num_iter += 1
        if num_iter > max_iter:
            raise RuntimeError(
                "Could not clean dangling scans. This is likely an issue with the data."
            )
        curr_len = last_len
        ms1_df = __shift_and_filter(ms1_df)
        last_len = len(ms1_df)

    ms1_df = ms1_df.drop("n_mz_values", "n_mz_values_next", "n_mz_values_ratio")

    final_len = len(ms1_df)
    logger.debug(
        f"Removed {initial_len - final_len} dangling scans, out of {initial_len} total"
        " scans."
    )

    # Finally concatenate the two dataframes back together
    return pl.concat((ms1_df, non_ms1_df), rechunk=False)


def __centroid_chunk(
    chunk_dict,
    mz_distance: float,
    ims_distance: float,
    min_neighbors: int = 1,
    max_peaks: int = 5_000,
    min_intensity: float | int = 10,
):
    mzs = np.concatenate(chunk_dict["mz_values"])
    prior_len = len(mzs)
    if len(mzs) < 1:
        return
    intensities = np.concatenate(chunk_dict["corrected_intensity_values"])
    # For some reason the dbscan section needs to be run on 64 bit precision
    # othersise, it has a bug where the final total intensity is higher than
    # the prior total intensity. (more intensity after cluster collapse)
    intensities = intensities.astype(np.float64)
    imss = chunk_dict["mobility_values"]
    imss = np.concatenate(
        [
            np.full_like(y, x)
            for x, y in zip(chunk_dict["mobility_values"], chunk_dict["mz_values"])
        ]
    )
    prior_intensity = intensities.sum()
    (mzs, imss), intensities = dbscan_collapse_multi(
        [mzs, imss],
        value_max_dists=[mz_distance, ims_distance],
        intensities=intensities,
        min_neighbors=min_neighbors,
        expansion_iters=3,
    )
    new_intensities = intensities.sum()
    assert new_intensities <= prior_intensity, (
        "New total intensity is higher than the prior"
        " total intensity "
        f"(New {new_intensities}, Prior {prior_intensity})"
        f" Diff = {new_intensities - prior_intensity}"
    )

    keep = np.argsort(intensities)[-max_peaks:]
    mzs = mzs[keep]
    imss = imss[keep]
    intensities = intensities[keep]

    if (min_intensity > 0) and len(intensities):
        keep = np.where(intensities > min_intensity)[0]
        mzs = mzs[keep]
        imss = imss[keep]
        intensities = intensities[keep]

    if len(mzs) < 1:
        return

    chunk_dict["mz_values"] = mzs
    chunk_dict["corrected_intensity_values"] = intensities.astype(np.float32)
    chunk_dict["mobility_values"] = imss

    compression = prior_len / len(mzs)
    signal_representation = new_intensities / prior_intensity
    return chunk_dict, {
        "compression": compression,
        "signal_representation": signal_representation,
    }


def get_timstof_data(
    path: os.PathLike,
    progbar: bool = True,
    safe: bool = False,
    centroid: bool = False,
    max_peaks_per_spectrum: int = 50_000,
    min_intensity: int | float = 10,
) -> pl.DataFrame:
    """Reads timsTOF data from a file and returns a DataFrame with the data.

    Parameters
    ----------
    path : os.PathLike
        The path to the timsTOF file (.d directory or hdf converted file).
    min_peaks : int, optional
        The minimum number of peaks to keep a spectrum, by default 15
    progbar : bool, optional
        Whether to show a progress bar, by default True
    safe : bool, optional
        Whether to use the safe method of reading the data, by default False
        Will be marginally faster if you disable it.
    centroid : bool, optional
        Whether to centroid the data, by default False
    max_peaks_per_spectrum : int, optional
        The maximum number of peaks to keep per spectrum, by default 5_000
    min_intensity : int, optional
        The minimum intensity to keep a peak, by default 10
    """
    # The TimsTOF object requires a string as input
    timstof_file = bruker.TimsTOF(str(path), mmap_detector_events=True)
    if timstof_file.acquisition_mode in {"ddaPASEF", "noPASEF"}:
        schema = SCHEMA_DDA
        logger.info("DDA data detected, using DDA schema")
    elif timstof_file.acquisition_mode == "diaPASEF":
        schema = SCHEMA_DIA
        logger.info("DIA data detected, using DIA schema")
    else:
        raise ValueError("Unknown acquisition mode")

    final_out = {k: [] for k in schema}

    if centroid:
        compressions = []
        fun = partial(
            __centroid_chunk,
            mz_distance=0.01,
            ims_distance=0.02,
            min_neighbors=1,
            max_peaks=max_peaks_per_spectrum,
            min_intensity=min_intensity,
        )
        iter = _iter_timstof_data(timstof_file, progbar=progbar, safe=safe)
        with Pool(processes=cpu_count()) as pool:
            for chunk_dict in pool.imap_unordered(
                fun,
                iter,
                chunksize=5,
            ):
                ## Dead code, only left here for debugging purposes
                ## Or when profiling some of the code.
                # for x in _iter_timstof_data(
                #     timstof_file, min_peaks=min_peaks, progbar=progbar, safe=safe
                # ):
                #     chunk_dict = fun(x)

                if chunk_dict is not None:
                    chunk_dict, compression = chunk_dict
                    chunk_dict["mz_values"] = chunk_dict["mz_values"].astype(np.float32)
                    chunk_dict["mobility_values"] = chunk_dict[
                        "mobility_values"
                    ].astype(np.float32)
                    chunk_dict["corrected_intensity_values"] = chunk_dict[
                        "corrected_intensity_values"
                    ].astype(np.float32)
                    compressions.append(compression)
                    for k in schema:
                        final_out[k].append(chunk_dict[k])

        compression = np.array([x["compression"] for x in compressions])
        signal_representation = np.array(
            [x["signal_representation"] for x in compressions]
        )
        logger.info(
            f"Compression: {np.mean(compression):.2f} +/- {np.std(compression):.2f}"
        )
        logger.info(
            f"Signal representation: {np.mean(signal_representation):.2f} +/-"
            f" {np.std(signal_representation):.2f}"
        )

    else:
        for chunk_dict in _iter_timstof_data(timstof_file, progbar=progbar, safe=safe):
            unnested = __unnest_chunk(chunk_dict)
            if unnested is not None:
                if len(unnested["mz_values"]) < 20:
                    # logger.info("Skipping chunk with less than 20 peaks")
                    # logger.debug(f"Ranges skipped: {unnested['query_range']} rt: {unnested['rt_values']}")
                    continue
                for k, v in unnested.items():
                    if k in schema:
                        final_out[k].append(v)

    del timstof_file

    logger.debug("Converting to arrays")
    final_out_f = {}
    for k, v in final_out.items():
        if k in schema:
            if not isinstance(v[0], np.ndarray):
                final_out_f[k] = np.array(v)

            else:
                final_out_f[k] = v

    logger.debug("Converting to DataFrame")
    final_out_f["corrected_intensity_values"] = [
        x.astype(np.float32) for x in final_out_f["corrected_intensity_values"]
    ]
    final_out = pl.DataFrame(final_out_f, schema=schema)
    final_out = __clean_dangling_precursor_scans(final_out)

    return final_out


if __name__ == "__main__":
    from pathlib import Path

    BASE_PATH = Path("/Users/sebastianpaez/tmp")
    DATA_FILE = "230612_SKNAS_DMSO_06_CHRO_1_DIA_S1-B1_1_1678.d"

    datafile = BASE_PATH / DATA_FILE
    out = get_timstof_data(datafile, progbar=False, centroid=False, safe=True)
