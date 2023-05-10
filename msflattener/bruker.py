import os
from multiprocessing import Pool, cpu_count

import numpy as np
import polars as pl
from alphatims import bruker
from loguru import logger
from tqdm.auto import tqdm

from .dbscan import dbscan_collapse, dbscan_collapse_multi
from .utils import _get_breaks_multi

SCHEMA = {
    "mz_values": pl.List(pl.Float64),
    "corrected_intensity_values": pl.List(pl.UInt32),
    "rt_values": pl.Float64,
    "mobility_values": pl.Float64,
    "quad_low_mz_values": pl.Float64,
    "quad_high_mz_values": pl.Float64,
}

NON_NESTED = [
    "rt_values",
    "mobility_values",
    "quad_low_mz_values",
    "quad_high_mz_values",
]

YIELDING_FIELDS = [
    "rt_values",
    "quad_low_mz_values",
    "quad_high_mz_values",
]


def _iter_timstof_data(
    timstof_file: bruker.TimsTOF, min_peaks=15, progbar=True, safe=False
):
    boundaries = [
        (start, end)
        for start, end in zip(
            timstof_file.push_indptr[:-1], timstof_file.push_indptr[1:]
        )
        if end - start >= min_peaks
    ]
    contexts = timstof_file.convert_from_indices(
        raw_indices=[x[0] for x in boundaries],
        return_rt_values=True,
        return_quad_mz_values=True,
        return_mobility_values=True,
        raw_indices_sorted=True,
    )
    if safe:
        context2 = timstof_file.convert_from_indices(
            raw_indices=[x[1] - 1 for x in boundaries],
            return_rt_values=True,
            return_quad_mz_values=True,
            return_mobility_values=True,
            raw_indices_sorted=True,
        )
        for k, v in context2.items():
            if not np.all(v == contexts[k]):
                raise ValueError(
                    f"Not all fields in the timstof context share valuesfor {k} "
                )

    my_iter = tqdm(
        boundaries,
        disable=not progbar,
        desc="Tims Pushes",
        total=len(boundaries),
        mininterval=0.2,
        maxinterval=5,
    )
    chunk_out = {}
    last_rt = None
    last_quad_high = None
    last_quad_low = None

    for i, (start, end) in enumerate(my_iter):
        context = {k: v[i] for k, v in contexts.items()}
        query_range = range(start, end)

        out = timstof_file.convert_from_indices(
            raw_indices=query_range,
            return_corrected_intensity_values=True,
            return_mz_values=True,
            raw_indices_sorted=True,
        )
        for k, v in context.items():
            out[k] = v

        if last_rt is not None:
            any_change = (
                last_rt != out["rt_values"]
                or last_quad_high != out["quad_high_mz_values"]
                or last_quad_low != out["quad_low_mz_values"]
            )
            if any_change:
                yield chunk_out
                chunk_out = {}

        for k, v in out.items():
            if k in YIELDING_FIELDS:
                chunk_out[k] = v
            else:
                chunk_out.setdefault(k, []).append(v)

        last_rt = out["rt_values"]
        last_quad_high = out["quad_high_mz_values"]
        last_quad_low = out["quad_low_mz_values"]

    if chunk_out:
        yield chunk_out


def __centroid_chunk(chunk_dict):
    mzs = np.concatenate(chunk_dict["mz_values"])
    if len(mzs) < 1:
        return
    intensities = np.concatenate(chunk_dict["corrected_intensity_values"])
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
        value_max_dists=[0.01, 0.01],
        intensities=intensities,
        min_neighbors=3,
        expansion_iters=10,
    )
    new_intensities = intensities.sum()
    assert new_intensities <= prior_intensity

    if len(mzs) < 5:
        return

    chunk_dict["mz_values"] = mzs
    chunk_dict["corrected_intensity_values"] = intensities
    chunk_dict["mobility_values"] = imss
    return chunk_dict


def get_timstof_data(
    path: os.PathLike, min_peaks=15, progbar=True, safe=False, centroid=False
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
    """
    timstof_file = bruker.TimsTOF(path, mmap_detector_events=True)

    final_out = {k: [] for k in SCHEMA}

    if centroid:
        with Pool(processes=cpu_count()) as pool:
            for chunk_dict in pool.imap_unordered(
                __centroid_chunk,
                _iter_timstof_data(
                    timstof_file, min_peaks=min_peaks, progbar=progbar, safe=safe
                ),
            ):
                if chunk_dict is None:
                    continue
                for k, v in chunk_dict.items():
                    final_out[k].append(v)

    else:
        for chunk_dict in _iter_timstof_data(
            timstof_file, min_peaks=min_peaks, progbar=progbar, safe=safe
        ):
            for k, v in chunk_dict.items():
                final_out[k].append(v)

    del timstof_file

    final_out = {
        k: np.array(v) if not isinstance(v[0], np.ndarray) else v
        for k, v in final_out.items()
    }
    if centroid:
        SCHEMA["mobility_values"] = SCHEMA["mz_values"]
    final_out = pl.DataFrame(final_out, schema=SCHEMA)

    return final_out


def _merge_ims_simple_chunk(chunk, min_neighbors=3, mz_distance=0.01):
    mzs = np.concatenate(chunk["mz_values"])
    intensities = np.concatenate(chunk["corrected_intensity_values"])
    in_len = len(mzs)

    mzs, intensities = dbscan_collapse(
        mzs,
        intensities=intensities,
        min_neighbors=min_neighbors,
        value_max_dist=mz_distance,
    )
    out_vals = {}
    out_vals["mz_values"] = mzs
    out_vals["corrected_intensity_values"] = intensities
    out_vals["rt_values"] = chunk["rt_values"][0]
    out_vals["quad_low_mz_values"] = chunk["quad_low_mz_values"][0]
    out_vals["quad_high_mz_values"] = chunk["quad_high_mz_values"][0]
    out_vals["mobility_low_values"] = chunk["mobility_values"].min()
    out_vals["mobility_high_values"] = chunk["mobility_values"].max()
    return pl.DataFrame(out_vals), in_len


def merge_ims_simple(df: pl.DataFrame, min_neighbors=3, mz_distance=0.01, progbar=True):
    df = df.sort(["rt_values", "quad_low_mz_values"])
    breaks = _get_breaks_multi(
        df["rt_values"].to_numpy(), df["quad_low_mz_values"].to_numpy()
    )
    my_iter = tqdm(
        zip(breaks[:-1], breaks[1:]),
        total=len(breaks) - 1,
        disable=not progbar,
        desc="Merging in 1d",
    )
    out_vals = []
    skipped = 0
    total = len(breaks) - 1
    for start, end in my_iter:
        chunk = df[start:end]
        out_chunk, in_len = _merge_ims_simple_chunk(
            chunk=chunk, min_neighbors=min_neighbors, mz_distance=mz_distance
        )
        if len(out_chunk):
            my_iter.set_postfix({"out_len": len(out_chunk), "in_len": in_len})
        else:
            out_vals.append(out_chunk)
            skipped += 1

    logger.info(
        f"Finished simple ims merge, skipped {skipped}/{total} spectra for not having"
        " any peaks"
    )

    return pl.DataFrame(out_vals)


def _centroid_ims_chunk(chunk, mz_distance, ims_distance, min_neighbors):
    mzs = np.concatenate(chunk["mz_values"])
    imss = np.concatenate(
        [[x] * len(y) for x, y in zip(chunk["mobility_values"], chunk["mz_values"])]
    )
    intensities = np.concatenate(chunk["corrected_intensity_values"])
    prior_intensity = intensities.sum()
    in_len = len(mzs)

    (mzs, imss), intensities = dbscan_collapse_multi(
        [mzs, imss],
        value_max_dists=[mz_distance, ims_distance],
        intensities=intensities,
        min_neighbors=min_neighbors,
        expansion_iters=10,
    )
    new_intensities = intensities.sum()
    assert new_intensities <= prior_intensity

    out_vals = {}
    out_vals["mz_values"] = mzs
    out_vals["corrected_intensity_values"] = intensities
    out_vals["mobility_values"] = imss
    out_vals["rt_values"] = chunk["rt_values"][0]
    out_vals["quad_low_mz_values"] = chunk["quad_low_mz_values"][0]
    out_vals["quad_high_mz_values"] = chunk["quad_high_mz_values"][0]

    return pl.DataFrame(out_vals), in_len


def centroid_ims(
    df: pl.DataFrame, min_neighbors=3, mz_distance=0.01, ims_distance=0.01, progbar=True
):
    df = df.sort(["rt_values", "quad_low_mz_values"])
    breaks = _get_breaks_multi(
        df["rt_values"].to_numpy(),
        df["quad_low_mz_values"].to_numpy(),
    )
    total = len(breaks) - 1
    my_iter = tqdm(
        zip(breaks[:-1], breaks[1:]),
        total=total,
        disable=not progbar,
        desc="Merging in 2d",
    )
    out_vals = []
    skipped = 0
    for start, end in my_iter:
        chunk = df[start:end]
        out_chunk, in_len = _centroid_ims_chunk(
            chunk=chunk,
            mz_distance=mz_distance,
            ims_distance=ims_distance,
            min_neighbors=min_neighbors,
        )

        if len(out_chunk):
            my_iter.set_postfix({"out_len": len(out_chunk), "in_len": in_len})
            out_vals.append(out_chunk)
        else:
            skipped += 1

    logger.info(
        f"Finished simple ims merge, skipped {skipped}/{total} spectra for not having"
        " any peaks"
    )

    return pl.concat(out_vals, axis=0)
