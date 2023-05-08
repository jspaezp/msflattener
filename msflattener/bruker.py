import numpy as np
import polars as pl
from alphatims import bruker
from loguru import logger
from tqdm.auto import tqdm

from .dbscan import dbscan_collapse, dbscan_collapse_multi
from .utils import _get_breaks, _get_breaks_multi


def get_timstof_data(path, min_peaks=15, progbar=True, safe=False):
    FIELDS = [
        "mz_values",
        "corrected_intensity_values",
        "rt_values",
        "mobility_values",
        "quad_low_mz_values",
        "quad_high_mz_values",
    ]
    timstof_file = bruker.TimsTOF(path)

    final_out = {}
    for x in FIELDS:
        final_out[x] = []

    my_iter = tqdm(
        zip(timstof_file.push_indptr[:-1], timstof_file.push_indptr[1:]),
        disable=not progbar,
        desc="Tims Pushes",
        total=len(timstof_file.push_indptr),
    )
    for start, end in my_iter:
        if end - start < min_peaks:
            continue

        context = timstof_file.convert_from_indices(
            raw_indices=[start],
            return_rt_values=True,
            return_quad_mz_values=True,
            return_mobility_values=True,
            raw_indices_sorted=True,
        )
        query_range = range(start, end)
        if safe:
            context2 = timstof_file.convert_from_indices(
                raw_indices=[end],
                return_rt_values=True,
                return_quad_mz_values=True,
                return_mobility_values=True,
                raw_indices_sorted=True,
            )
            if not all(context[x] == context2[x] for x in context):
                raise ValueError(
                    "Not all fields in the timstof context share values"
                    f" {context} {context2}"
                )

        for k, v in context.items():
            final_out[k].append(v[0])
        out = timstof_file.convert_from_indices(
            raw_indices=query_range,
            return_corrected_intensity_values=True,
            return_mz_values=True,
            raw_indices_sorted=True,
        )
        for k, v in out.items():
            final_out[k].append(v)

    return pl.DataFrame(final_out)


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
    out_vals = {
        "mz_values": [],
        "corrected_intensity_values": [],
        "quad_low_mz_values": [],
        "quad_high_mz_values": [],
        "rt_values": [],
        "mobility_low_values": [],
        "mobility_high_values": [],
    }
    skipped = 0
    total = len(breaks) - 1
    for start, end in my_iter:
        chunk = df[start:end]

        mzs = np.concatenate(chunk["mz_values"])
        intensities = np.concatenate(chunk["corrected_intensity_values"])
        in_len = len(mzs)

        mzs, intensities = dbscan_collapse(
            mzs,
            intensities=intensities,
            min_neighbors=min_neighbors,
            value_max_dist=mz_distance,
        )

        if len(mzs):
            my_iter.set_postfix({"out_len": len(mzs), "in_len": in_len})
            out_vals["mz_values"].append(mzs)
            out_vals["corrected_intensity_values"].append(intensities)
            out_vals["rt_values"].append(chunk["rt_values"][0])
            out_vals["quad_low_mz_values"].append(chunk["quad_low_mz_values"][0])
            out_vals["quad_high_mz_values"].append(chunk["quad_high_mz_values"][0])
            out_vals["mobility_low_values"].append(chunk["mobility_values"].min())
            out_vals["mobility_high_values"].append(chunk["mobility_values"].max())
        else:
            skipped += 1

    logger.info(
        f"Finished simple ims merge, skipped {skipped}/{total} spectra for not having"
        " any peaks"
    )

    return pl.DataFrame(out_vals)


def centroid_ims(
    df: pl.DataFrame, min_neighbors=3, mz_distance=0.01, ims_distance=0.01, progbar=True
):
    df = df.sort(["rt_values", "quad_low_mz_values"])
    breaks = _get_breaks_multi(
        df["rt_values"].to_numpy(),
        df["quad_low_mz_values"].to_numpy(),
    )
    my_iter = tqdm(
        zip(breaks[:-1], breaks[1:]),
        total=len(breaks) - 1,
        disable=not progbar,
        desc="Merging in 2d",
    )
    out_vals = {
        "mz_values": [],
        "corrected_intensity_values": [],
        "mobility_values": [],
        "quad_low_mz_values": [],
        "quad_high_mz_values": [],
        "rt_values": [],
    }
    skipped = 0
    total = len(breaks) - 1
    for start, end in my_iter:
        chunk = df[start:end]

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

        if len(mzs):
            my_iter.set_postfix({"out_len": len(mzs), "in_len": in_len})
            out_vals["mz_values"].append(mzs)
            out_vals["corrected_intensity_values"].append(intensities)
            out_vals["mobility_values"].append(imss)
            out_vals["rt_values"].append(chunk["rt_values"][0])
            out_vals["quad_low_mz_values"].append(chunk["quad_low_mz_values"][0])
            out_vals["quad_high_mz_values"].append(chunk["quad_high_mz_values"][0])
        else:
            skipped += 1

    logger.info(
        f"Finished simple ims merge, skipped {skipped}/{total} spectra for not having"
        " any peaks"
    )

    return pl.DataFrame(out_vals)


def centroid_ims_slidingwindow(
    df: pl.DataFrame,
    min_neighbors=3,
    mz_distance=0.01,
    ims_distance=0.01,
    rt_distance=2,
    progbar=True,
):
    out_vals = {
        "mz_values": [],
        "corrected_intensity_values": [],
        "mobility_values": [],
        "quad_low_mz_values": [],
        "quad_high_mz_values": [],
        "rt_values": [],
    }
    skipped = 0
    total = 0
    my_iter = tqdm(
        _iter_with_window(df, rt_window_max=rt_distance),
        disable=not progbar,
        desc="Merging in 2d with sliding window",
    )
    for chunk, before, after in my_iter:
        total += 1
        mzs = np.concatenate(chunk["mz_values"])
        imss = np.concatenate(
            [[x] * len(y) for x, y in zip(chunk["mobility_values"], chunk["mz_values"])]
        )
        intensities = np.concatenate(chunk["corrected_intensity_values"])
        if len(mzs) == 0:
            logger.error("No peaks in chunk")
        if len(before + after) > 0:
            n_imss = []
            n_mzs = []
            for z in before + after:
                c_mzs = np.concatenate(z["mz_values"])
                n_mzs.append(c_mzs)
                c_imss = [
                    [x] * len(y) for x, y in zip(z["mobility_values"], z["mz_values"])
                ]
                n_imss.append(np.concatenate(c_imss))

            n_imss = np.concatenate(n_imss)
            n_mzs = np.concatenate(n_mzs)
            count_only_values_list = [n_mzs, n_imss]
        else:
            count_only_values_list = None

        prior_intensity = intensities.sum()
        in_len = len(mzs)

        (mzs, imss), intensities = dbscan_collapse_multi(
            [mzs, imss],
            value_max_dists=[mz_distance, ims_distance],
            intensities=intensities,
            min_neighbors=min_neighbors,
            expansion_iters=10,
            count_only_values_list=count_only_values_list,
        )
        new_intensities = intensities.sum()
        assert new_intensities <= prior_intensity

        if len(mzs):
            my_iter.set_postfix({"out_len": len(mzs), "in_len": in_len})
            out_vals["mz_values"].append(mzs)
            out_vals["corrected_intensity_values"].append(intensities)
            out_vals["mobility_values"].append(imss)
            out_vals["rt_values"].append(chunk["rt_values"][0])
            out_vals["quad_low_mz_values"].append(chunk["quad_low_mz_values"][0])
            out_vals["quad_high_mz_values"].append(chunk["quad_high_mz_values"][0])
        else:
            skipped += 1

    logger.info(
        f"Finished Sliding, skipped {skipped}/{total} spectra for not having any peaks"
    )

    return pl.DataFrame(out_vals)
