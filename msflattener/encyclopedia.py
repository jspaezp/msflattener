from __future__ import annotations

import os
import sqlite3
import zlib
from pathlib import Path

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
INSERT INTO "main"."metadata" ("Key", "Value") VALUES ('version', '0.5.0');
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


def _sort_by_first(*args: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sorts the arrays by the first array."""
    idx = np.argsort(args[0])
    return tuple(x[idx] for x in args)


def _compress_array(array: np.ndarray, dtype: str) -> bytes:
    r"""Compress the array to the EncyclopeDIA format.

    Compresses an array into a zlib-compressed byte array. The array is first
    packed into a byte array using the struct module. The byte array is then
    compressed using zlib.

    Examples
    --------
        >>> _compress_array(np.array([1, 2, 3, 4, 5]), "d")
        b'x^\xb3\xff\xc0\x00\x06\x0e\x10\x8a\xc1\x81\x03J\x0b@i\x11\x08\r\x00D\xc4\x02\\'
        >>> _extract_array(_compress_array(np.array([1, 2, 3, 4, 5]), "d"), "d")
        array([1., 2., 3., 4., 5.])
        >>> _compress_array(np.array([1, 2, 3, 4, 5]), "i")
        b'x^c```d```\x02bf f\x01bV\x00\x00s\x00\x10'
    """
    # packed = struct.pack(">" + (dtype * len(array)), *array.astype(dtype))
    # This version seems to be faster
    packed = array.astype(f">{dtype}").tobytes()
    # compressed = zlib.compress(packed, 9)
    compressed = zlib.compress(packed, 2)
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
        >>> _extract_array(b"x\xda\xb3o``p`\x00b \xe1\x00b/``\x00\x00 \xa0\x03 ", "f")
        array([1., 2., 3., 4., 5.], dtype=float32)
        >>> samp_mass = b"x\xda\xb3\xff\xc0\x00\x06\x0e\x0cP\x9a\x03J\x0b@i\x11\x08\r\x00D\xc4\x02\\"
        >>> _extract_array(samp_mass, "d")
        array([1., 2., 3., 4., 5.])
    """  # noqa
    dtype = np.dtype(f">{type_str}")
    decompressed = zlib.decompress(byte_array, 32)
    # decompressed_length = len(decompressed) // dtype.itemsize
    # unpacked = struct.unpack(">" + (type_str * decompressed_length), decompressed)
    # unpacked = np.array(unpacked, dtype=dtype)
    unpacked = np.frombuffer(decompressed, dtype=dtype)
    return unpacked


def _write_scan(
    scan_dict: dict, curr: sqlite3.Cursor, precursor_id: int | None = None
) -> None:
    # mz_values: list | np.ndarray
    # corrected_intensity_values: list | np.ndarray
    # mobility_values: list | np.ndarray
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
    tic = sum(corrected_intensity_values)

    mz_values = np.array(mz_values, dtype="f")
    corrected_intensity_values = np.array(corrected_intensity_values, dtype="f")
    mobility_values = np.array(mobility_values, dtype="f")

    # I am not sure if this is needed
    mz_values, corrected_intensity_values, mobility_values = _sort_by_first(
        mz_values,
        corrected_intensity_values,
        mobility_values,
    )

    mz_values = _compress_array(mz_values, "d")
    corrected_intensity_values = _compress_array(corrected_intensity_values, "f")
    mobility_values = _compress_array(mobility_values, "f")

    if quad_low_mz_values < 0:
        assert precursor_id is None
        input_query = " ".join(
            [
                "INSERT INTO precursor (",
                ", ".join(
                    [
                        "Fraction",
                        "SpectrumName ",
                        "SpectrumIndex",
                        "ScanStartTime",
                        "MassEncodedLength",
                        "MassArray",
                        "IntensityEncodedLength",
                        "IntensityArray",
                        "TIC",
                        "MobilityEncodedLength",
                        "MobilityArray",
                        "IsolationWindowLower",
                        "IsolationWindowUpper",
                    ]
                ),
                ")",
                "VALUES",
                f" ({','.join(['?']*13)})",
            ]
        )

        values = (
            0,  # Fraction
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
            min(400.0, min(mz_values)),  # TODO replace with actual values ...
            max(1600.0, max(mz_values)),  # TODO replace with actual values ...
        )
    else:
        assert precursor_id is not None

        precursor_name = f"scan={precursor_id}"
        input_query = " ".join(
            [
                "INSERT INTO spectra (",
                ", ".join(
                    [
                        "Fraction",
                        "SpectrumName ",
                        "SpectrumIndex",
                        "ScanStartTime",
                        "MassEncodedLength",
                        "MassArray",
                        "IntensityEncodedLength",
                        "IntensityArray",
                        # "TIC",
                        "MobilityEncodedLength",
                        "MobilityArray",
                        "IsolationWindowLower",
                        "IsolationWindowUpper",
                        "IsolationWindowCenter",
                        "PrecursorName",
                        "PrecursorCharge",
                    ]
                ),
                ")",
                "VALUES",
                f" ({','.join(['?']*15)})",
            ]
        )
        values = (
            0,  # Fraction
            scan_name,
            id,
            rt_values,
            len(mz_values),
            mz_values,
            len(corrected_intensity_values),
            corrected_intensity_values,
            # tic, # Oddly enought here is no TIC in the spectra table
            len(mobility_values),
            mobility_values,
            quad_low_mz_values,
            quad_high_mz_values,
            (quad_high_mz_values + quad_low_mz_values) / 2,
            precursor_name,
            0,  # PrecursorCharge
        )

    try:
        curr.execute(
            input_query,
            values,
        )
    except sqlite3.IntegrityError as e:
        print(e)
        raise


def write_scans(df: pl.DataFrame, conn: sqlite3.Connection) -> None:
    """Writes the spectra and precursor table to the database.

    The scans table is a table that contains the mass, intensity, and
    ion mobility values for each scan. It also contains the retention time
    for each scan.

    """
    iter = yield_scans(df)
    curr = conn.cursor()
    tic = 0

    with tqdm(
        desc="Writing precursor/scans", total=len(df), miniters=1, smoothing=0.1
    ) as pbar:
        for precursor, spectra in iter:
            tic += sum(precursor["corrected_intensity_values"])
            _write_scan(precursor, curr)
            pbar.update(1)
            for scan in spectra:
                _write_scan(scan, curr, precursor_id=precursor["id"])
                pbar.update(1)

    conn.commit()
    return tic


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
                "INSERT INTO ranges (Start, Stop, DutyCycle, NumWindows, MobilityStart,"
                " MobilityStop) VALUES (?, ?, ?, ?, ?, ?)"
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


# TODO implement these metadata values
"""
INSERT INTO "main"."metadata" ("Key", "Value") VALUES ('SoftwareVersion_MS:1000532', '4.2-4.2.319.10-SP1/4.2.362.21');
INSERT INTO "main"."metadata" ("Key", "Value") VALUES ('SoftwareVersion_MS:1000615', '3.0.23052');
INSERT INTO "main"."metadata" ("Key", "Value") VALUES ('InstrumentConfigurations', 'configurationId:IC1,accession:null,name:null[INSTRUMENT-ID-COMPONENT-DELIMITER]order:1,cvRef:MS,accessionId:MS:1000485,name:nanospray inlet,type:source;order:2,cvRef:MS,accessionId:MS:1000081,name:quadrupole,type:analyzer;order:3,cvRef:MS,accessionId:MS:1000484,name:orbitrap,type:analyzer;order:4,cvRef:MS,accessionId:MS:1000624,name:inductive detector,type:detector');
INSERT INTO "main"."metadata" ("Key", "Value") VALUES ('runStartTime', '2023-04-08T08:38:13Z');
"""


def write_encyclopedia(
    df: pl.DataFrame,
    output_filepath: os.PathLike,
    orig_filepath: os.PathLike,
) -> None:
    conn = sqlite3.connect(output_filepath)
    try:
        conn.executescript(PRECURSOR_SCHEMA)
        conn.executescript(RANGES_SCHEMA)
        conn.executescript(SPECTRA_SCHEMA)
        conn.executescript(METADATA_SCHEMA)

        # For performance reasons
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        conn.execute("PRAGMA cache_size = 1000000;")
        conn.execute("PRAGMA temp_store = MEMORY")
        tic = write_scans(df, conn)
        conn.executemany(
            "INSERT INTO main.metadata (Key, Value) VALUES (?, ?)",
            [
                ("totalPrecursorTIC", tic),
                ("sourcename", Path(orig_filepath).stem),
                ("filelocation", str(orig_filepath)),
                ("filename", Path(orig_filepath).name),
            ],
        )
        write_ranges(df, conn)
        conn.executescript(INDEX_SCHEMA)
        conn.commit()
    finally:
        conn.close()
