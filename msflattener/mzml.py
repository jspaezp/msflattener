"""Implements reading and writting mzml file and its interface with polars.

This module was written by Will Fondrie.
"""

from __future__ import annotations

import base64
import os
import zlib

import numpy as np
import polars as pl
from loguru import logger
from lxml import etree
from psims.mzml.writer import MzMLWriter
from tqdm.auto import tqdm

from msflattener.base import yield_scans

ACC_TO_TYPE = {
    "MS:1000519": np.float32,
    "MS:1000520": np.float16,
    "MS:1000521": np.float32,
    "MS:1000522": np.int32,
    "MS:1000523": np.float64,
}


def read(
    mzml_file: os.PathLike, progbar: bool = True
) -> dict[str, tuple[np.ndarray, np.ndarray, float, float, float]]:
    """Parse the mzML file.

    Parameters
    ----------
    mzml_file : Path-like
        The mzML file to parse.
    progbar : bool
        Whether to show progress.

    Returns
    -------
    Dict of Tuple of (np.ndarray, np.ndarray, float, float, float)
        The mass spectra, indexed by the scan identifier.
        Companion float values are the retention time (seconds),
        lower isolation window, and upper isolation window.

    Note:
    This function was originally written by Will Fondrie
    """
    spectra = {}
    for _, elem in tqdm(
        etree.iterparse(str(mzml_file), tag="{*}spectrum"), disable=not progbar
    ):
        spec_id, arrays = _parse_spectrum(elem)
        spectra[spec_id] = arrays

    return spectra


def _desc_acc_val(elem: etree.Element, accesion: str) -> str:
    """Get the value of an element with a given accession."""
    elems = elem.xpath(f"descendant::*[@accession='{accesion}']")
    out = elems[0].get("value")
    return out


def _parse_spectrum(  # noqa: C901
    elem: etree.Element,
) -> tuple[str, np.ndarray, np.ndarray, float, float, float]:
    """Parse a mass spectrum.

    Parameters
    ----------
    elem : lxml.etree.Element
        The element with the Spectrum tag.

    Returns
    -------
    spec_id : str
        The spectrum identifier.
    mz_array : np.ndarray
    intensity_array : np.ndarray
        The mass spectrum.
    rt : float
        The retention time in seconds
    low_isolation : float
        Lower limit set to the isolation window
        Will be set to -1 if no isolation is detected
    high_isolation : float
        Higher limit set to the isolation window
        Will be set to -1 if no isolation is detected

    Note:
    This function was originally written by Will Fondrie and JSPP added
    functionality to extract the isolation windows, retention times
    and different compresison schemas.
    """
    spec_id = elem.get("id")

    # Handle retention time
    rt_info = elem.xpath("descendant::*[@accession='MS:1000016']")[0]
    unit_accession = rt_info.get("unitAccession")
    rt = float(rt_info.get("value"))
    if unit_accession == "UO:0000031":
        # minutes
        rt *= 60
    elif unit_accession == "UO:0000010":
        pass
    else:
        raise ValueError("Unable to find retention time")

    # Handle isolation windows
    try:
        iso_window = next(elem.iter("{*}isolationWindow"))
        window_center = float(_desc_acc_val(iso_window, "MS:1000827"))
        low_offset = float(_desc_acc_val(iso_window, "MS:1000828"))
        high_offset = float(_desc_acc_val(iso_window, "MS:1000829"))

        low_iso, high_iso = window_center - low_offset, window_center + high_offset
    except StopIteration:
        low_iso, high_iso = -1.0, -1.0

    bin_list = next(elem.iter("{*}binaryDataArrayList"))
    # spectrum is m/z array and intensity arrays and rtinseconds.
    spec = [None, None, rt, low_iso, high_iso]
    for bin_array in bin_list:
        compressed = False
        np_type = None

        for child in bin_array:
            if child.tag.endswith("cvParam"):
                accession = child.get("accession")
                if accession == "MS:1000514":
                    idx = 0
                elif accession == "MS:1000515":
                    idx = 1
                elif accession == "MS:1000574":
                    compressed = True
                elif accession in list(ACC_TO_TYPE):
                    np_type = ACC_TO_TYPE[accession]
                    # TODO check if finding the compression and type
                    # schema can be done once per file and not once per
                    # spectrum

            if child.tag.endswith("binary"):
                decoded = base64.b64decode(child.text.encode("ascii"))
                if compressed:
                    decoded = zlib.decompress(decoded)

                try:
                    array = np.frombuffer(bytearray(decoded), dtype=np_type)
                except ValueError as e:
                    if "buffer size must be a multiple of element size" in str(e):
                        err = ValueError(
                            f"Unable to decompress array to type {np_type}"
                        )
                        raise err
                    else:
                        raise

                if np.any(array < 0):
                    error_msg = str(bin_array)
                    raise ValueError(f"Array failed to decompress:\n{error_msg}")

        spec[idx] = array

    return spec_id, *spec


def get_mzml_data(
    path: os.PathLike, min_peaks: int, progbar: bool = True
) -> pl.DataFrame:
    """Reads a mzML file and returns a DataFrame with the spectra.

    Parameters
    ----------
    path : os.PathLike
        The path to the mzML file.
    min_peaks : int
        The minimum number of peaks to keep a spectrum.
    progbar : bool
        Whether to show progress.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the spectra.
        It has the following columns:
            mz_values: np.ndarray, nested
            corrected_intensity_values: np.ndarray, nested
            rt_values: np.ndarray
            quad_low_mz_values: np.ndarray
            quad_high_mz_values: np.ndarray
    """
    out = {
        "mz_values": [],
        "corrected_intensity_values": [],
        "rt_values": [],
        "quad_low_mz_values": [],
        "quad_high_mz_values": [],
    }

    for _, elem in tqdm(
        etree.iterparse(str(path), tag="{*}spectrum"),
        desc="Spectra",
        disable=not progbar,
    ):
        _spec_id, *arrays = _parse_spectrum(elem)
        if len(arrays[0]) > min_peaks:
            for k, v in zip(out, arrays):
                out[k].append(v)

    return pl.DataFrame(out)


def write_mzml(df: pl.DataFrame, out_path: os.PathLike) -> None:
    """Write a parquet df to mzml.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to write. It needs to contain the following columns:
        mz_values, corrected_intensity_values, rt_values, quad_low_mz_values,
        quad_high_mz_values.
    out_path : os.PathLike
        The path to write the mzml file to.

    This function has ben greatly taken from the readme of psims.
    """
    # Load the data to write
    scans = yield_scans(df)
    ms1_scans = 0
    ms2_scans = 0
    with MzMLWriter(open(out_path, "wb"), close=True) as out:
        # Add default controlled vocabularies
        out.controlled_vocabularies()
        # Open the run and spectrum list sections
        with out.run(id="my_analysis"):
            spectrum_count = len(df)
            with out.spectrum_list(count=spectrum_count):
                for scan, products in scans:
                    ms1_scans += 1
                    # Write Precursor scan
                    out.write_spectrum(
                        scan["mz_values"],
                        scan["corrected_intensity_values"],
                        scan_start_time=scan["rt_values"] / 60,
                        id=scan["id"],
                        params=[
                            "MS1 Spectrum",
                            {"ms level": 1},
                            {
                                "total ion current": sum(
                                    scan["corrected_intensity_values"]
                                )
                            },
                        ],
                    )
                    # Write MSn scans
                    for prod in products:
                        ms2_scans += 1
                        prec_mz = prod["precursor_mz_values"]
                        offset_low = prec_mz - prod["quad_low_mz_values"]
                        offset_high = prod["quad_high_mz_values"] - prec_mz

                        prec_info = {
                            "mz": prec_mz,
                            # "intensity": prod.precursor_intensity,
                            "scan_id": scan["id"],
                            "activation": [
                                "beam-type collisional dissociation",
                                {"collision energy": 25},
                            ],
                            "isolation_window": [offset_low, prec_mz, offset_high],
                        }

                        if "precursor_charge" in prod:
                            prec_info["charge"] = prod["precursor_charge"]

                        out.write_spectrum(
                            prod["mz_values"],
                            prod["corrected_intensity_values"],
                            scan_start_time=prod["rt_values"] / 60,
                            id=prod["id"],
                            params=[
                                "MSn Spectrum",
                                {"ms level": 2},
                                {
                                    "total ion current": sum(
                                        prod["corrected_intensity_values"]
                                    )
                                },
                            ],
                            # Include precursor information
                            precursor_information=prec_info,
                        )
        logger.info(
            f"Written {ms1_scans} MS1 scans and {ms2_scans} MS2 scans to {out_path}"
        )
