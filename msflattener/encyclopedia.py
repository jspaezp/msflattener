from __future__ import annotations

import sqlite3
import struct
import zlib
import os

import numpy as np
import polars as pl
from numpy.typing import DTypeLike
from tqdm.auto import tqdm

from msflattener.base import yield_scans

PRECURSOR_SCHEMA = """
CREATE TABLE precursor (
    Fraction int not null,
    SpectrumName string not null,
    SpectrumIndex int not null,
    ScanStartTime float not null,
    IonInjectionTime float,
    IsolationWindowLower float not null,
    IsolationWindowUpper float not null,
    MassEncodedLength int not null,
    MassArray blob not null,
    IntensityEncodedLength int not null,
    IntensityArray blob not null,
    MobilityEncodedLength int,
    MobilityArray blob,
    TIC float,
    primary key (SpectrumIndex)
);
"""

# TODO write a check to make sure the mobilities are either all null or none null
RANGES_SCHEMA = """
CREATE TABLE ranges (
    Start float not null,
    Stop float not null,
    MobilityStart float, -- Added here
    MobilityStop float, -- Added here
    DutyCycle float not null,
    NumWindows int
);
"""


SPECTRA_SCHEMA = """
CREATE TABLE spectra (
    Fraction int not null,
    SpectrumName string not null,
    PrecursorName string,
    SpectrumIndex int not null,
    ScanStartTime float not null,
    IonInjectionTime float,
    IsolationWindowLower float not null,
    IsolationWindowCenter float not null,
    IsolationWindowUpper float not null,
    PrecursorCharge int not null,
    MassEncodedLength int not null,
    MassArray blob not null,
    IntensityEncodedLength int not null,
    IntensityArray blob not null,
    MobilityEncodedLength int, -- Added here
    MobilityArray blob, -- Added here
    MobilityIsolationWindowLower float, -- Added here
    MobilityIsolationWindowCenter float, -- Added here
    MobilityIsolationWindowUpper float, -- Added here
    primary key (SpectrumIndex)
);
"""

METADATA_SCHEMA = """
CREATE TABLE metadata (
    Key string not null,
    Value string not null,
    primary key (Key)
);
"""

# TODO add some indexing to the mobility information
INDEX_SCHEMA = """
CREATE INDEX
    "precursor_index_isolation_window_lower" on
    "precursor" ("IsolationWindowLower" ASC);
CREATE INDEX
    "precursor_index_isolation_window_upper" on
    "precursor" ("IsolationWindowUpper" ASC);
CREATE INDEX
    "precursor_index_scan_start_time" on
    "precursor" ("ScanStartTime" ASC);
CREATE INDEX
    "spectra_index_isolation_window_lower" on
    "spectra" ("IsolationWindowLower" ASC);
CREATE INDEX
    "spectra_index_isolation_window_upper" on
    "spectra" ("IsolationWindowUpper" ASC);
CREATE INDEX
    "spectra_index_scan_start_time_and_windows" on
    "spectra" ("ScanStartTime","IsolationWindowLower","IsolationWindowUpper" ASC);
"""

DIA_SCHEMA = (PRECURSOR_SCHEMA, RANGES_SCHEMA, SPECTRA_SCHEMA, METADATA_SCHEMA)


def _compress_array(array: np.ndarray, dtype: str) -> bytes:
    r"""Compress the array to the EncyclopeDIA format.

    Compresses an array into a zlib-compressed byte array. The array is first
    packed into a byte array using the struct module. The byte array is then
    compressed using zlib.

    Examples
    --------
        >>> _compress_array(np.array([1, 2, 3, 4, 5]), "d")
        b'x\xda\xb3\xff\xc0\x00\x06\x0e\x0cP\x9a\x03J\x0b@i\x11\x08\r\x00D\xc4\x02\\'
        >>> _compress_array(np.array([1, 2, 3, 4, 5]), "i")
        b'x\xdac```d```\x02bf f\x01bV\x00\x00s\x00\x10'
    """
    packed = struct.pack(">" + (dtype * len(array)), *array.astype(dtype))
    compressed = zlib.compress(packed, 9)
    return compressed


def _extract_array(byte_array: bytes, type_str: DTypeLike = "d") -> np.ndarray:
    r"""Extract the array from the byte array.

    Extracts an array from a zlib-compressed byte array. The byte array is
    first decompressed using zlib. The decompressed byte array is then unpacked
    using the struct module. The unpacked array is then converted to a numpy
    array.
    The type for masses is double and the rest if floats.

    Examples
    --------
        >>> samp_mass = b"x\xda\xb3\xff\xc0\x00\x06\x0e\x0cP\x9a\x03J\x0b@i\x11\x08\r\x00D\xc4\x02\\"
        >>> _extract_array(samp_mass, "d")
        array([1., 2., 3., 4., 5.])
        >>> _extract_array(b"x\xda\xb3o``p`\x00b \xe1\x00b/``\x00\x00 \xa0\x03 ", "f")
        array([1., 2., 3., 4., 5.], dtype=float32)
    """  # noqa
    dtype = np.dtype(type_str)
    decompressed = zlib.decompress(byte_array, 32)
    decompressed_length = len(decompressed) // dtype.itemsize
    unpacked = struct.unpack(">" + (type_str * decompressed_length), decompressed)
    return np.array(unpacked, dtype=dtype)


def _write_scan(
    scan_dict: dict, curr: sqlite3.Cursor, precursor_id: int | None = None
) -> None:
    # mz_values: np.ndarray
    # corrected_intensity_values: np.ndarray
    # mobility_values: np.ndarray
    # rt_values: float
    # quad_low_mz_values: float
    # quad_high_mz_values: float
    # id: int

    mz_values = scan_dict["mz_values"]
    corrected_intensity_values = scan_dict["corrected_intensity_values"]
    mobility_values = scan_dict["mobility_values"]

    rt_values = scan_dict["rt_values"]
    quad_low_mz_values = scan_dict["quad_low_mz_values"]
    quad_high_mz_values = scan_dict["quad_high_mz_values"]
    id = scan_dict["id"]

    scan_name = f"scan={id}"
    tic = corrected_intensity_values.sum()

    mz_values = _compress_array(mz_values, "f")
    corrected_intensity_values = _compress_array(corrected_intensity_values, "f")
    mobility_values = _compress_array(mobility_values, "f")

    if quad_low_mz_values < 0:
        assert precursor_id is None
        input_query = " ".join(
            "INSERT INTO precursor (",
            ", ".join(
                "SpectrumName ",
                "SpectrumIndex",
                "ScanStartTime",
                "MassEncodedLength",
                "MassArray",
                "IntensityEncodedLength",
                "IntensityArray",
                "TIC",
                "IonMobilityArrayEncodedLength",
                "IonMobilityArray",
            ),
            ")",
            "VALUES",
            f" ({','.join(['?']*10)})",
        )

        values = (
            scan_name,
            id,
            rt_values,
            len(mz_values),
            mz_values,
            len(corrected_intensity_values),
            corrected_intensity_values,
            tic,
            len(mobility_values),
            mobility_values,
        )
    else:
        assert precursor_id is not None

        precursor_name = f"scan={precursor_id}"
        input_query = " ".join(
            "INSERT INTO precursor (",
            ", ".join(
                "SpectrumName ",
                "SpectrumIndex",
                "ScanStartTime",
                "MassEncodedLength",
                "MassArray",
                "IntensityEncodedLength",
                "IntensityArray",
                "TIC",
                "IonMobilityArrayEncodedLength",
                "IonMobilityArray",
                "IsolationWindowLower",
                "IsolationWindowUpper",
                "PrecursorName",
            ),
            ")",
            "VALUES",
            f" ({','.join(['?']*13)})",
        )
        values = (
            scan_name,
            id,
            rt_values,
            len(mz_values),
            mz_values,
            len(corrected_intensity_values),
            corrected_intensity_values,
            tic,
            len(mobility_values),
            mobility_values,
            quad_low_mz_values,
            quad_high_mz_values,
            precursor_name,
        )

    curr.execute(
        input_query,
        values,
    )


def write_scans(df: pl.DataFrame, conn: sqlite3.Connection) -> None:
    """Writes the spectra and precursor table to the database.

    The scans table is a table that contains the mass, intensity, and
    ion mobility values for each scan. It also contains the retention time
    for each scan.

    """
    iter = yield_scans(df)
    curr = conn.cursor()

    for precursor, spectra in tqdm(iter):
        _write_scan(precursor, curr)
        for scan in spectra:
            _write_scan(scan, curr, precursor_id=precursor["id"])

    conn.commit()


def write_ranges(df: pl.DataFrame, conn: sqlite3.Connection) -> None:
    """Writes the ranges table to the database.

    The ranges table is a table that contains the quad isolation ranges and
    the median RT difference between scans in that range.

    This table contains only the information for ms2 scans.

    Example of a ranges table:
    ```
    sqlite> select * from ranges;
    602.523803710938|604.524719238281|2.49313354492187|3127
    636.539184570313|638.540100097656|2.4931333065033|3127
    638.540100097656|640.541015625|2.49313306808472|3127
    ...
    ```

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the scans.
    conn : sqlite3.Connection
        The connection to the database.

    Notes
    -----
    The ranges table is a table that contains the quad isolation ranges and
    the median RT difference between scans in that range.

    """
    g_df = df.groupby(["quad_high_mz_values", "quad_low_mz_values"])
    ranges_dict = {}

    for i, x in g_df:
        diff = np.diff(x["rt_values"].to_numpy(zero_copy_only=True))
        min_ims = x["mobility_values"].min()
        max_ims = x["mobility_values"].max()
        assert diff.min() >= 0

        med_diff = np.median(diff)
        ranges_dict[i] = {
            "med_diff": med_diff,
            "num_scans": len(x),
            "min_ims": min_ims,
            "max_ims": max_ims,
        }

    curr = conn.cursor()

    # CREATE TABLE ranges (
    # Start float not null,
    # Stop float not null,
    # DutyCycle float not null,
    # NumWindows int
    for (quad_high_mz_values, quad_low_mz_values), rt_range in tqdm(
        ranges_dict.items()
    ):
        curr.execute(
            (
                "INSERT INTO ranges (Start, Stop, DutyCycle, NumWindows, MobilityStart, MobilityStop)"
                " VALUES (?, ?, ?, ?)"
            ),
            (
                quad_low_mz_values,
                quad_high_mz_values,
                rt_range["med_diff"],
                rt_range["num_scans"],
                rt_range["min_ims"],
                rt_range["max_ims"],
            ),
        )

    conn.commit()


def write_encyclopedia(df: pl.DataFrame, output_filepath: os.PathLike) -> None:
    conn = sqlite3.connect(output_filepath)
    conn.executescript(PRECURSOR_SCHEMA)
    conn.executescript(RANGES_SCHEMA)
    conn.executescript(SPECTRA_SCHEMA)
    conn.executescript(METADATA_SCHEMA)
    write_scans(df, conn)
    conn.executescript(INDEX_SCHEMA)
    conn.commit()
    conn.close()
