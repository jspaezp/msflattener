import polars as pl
import pytest

from msflattener.base import SCHEMA_DDA, SCHEMA_DIA
from msflattener.bruker import get_timstof_data
from msflattener.mzml import get_mzml_data


def _check_schema_verbose(real_schema, target_schema):
    real_keys = set(real_schema.keys())
    target_keys = set(target_schema.keys())
    missing_keys = target_keys - real_keys
    extra_keys = real_keys - target_keys

    assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
    assert len(extra_keys) == 0, f"Extra keys: {extra_keys}"

    for k in target_schema:
        assert k in real_schema, f"{k} not in schema"
        assert (
            real_schema[k] == target_schema[k]
        ), f"{k} does not match; {real_schema[k]} != {target_schema[k]}"


def _check_polars_table(df, acquisition):
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] > 0

    if acquisition == "DDA":
        _check_schema_verbose(df.schema, SCHEMA_DDA)
        assert df.schema == SCHEMA_DDA

    elif acquisition == "DIA":
        assert df.schema == SCHEMA_DIA

    else:
        raise ValueError(f"Unknown acquisition type {acquisition}")


@pytest.mark.parametrize(
    "centroiding",
    [True, False],
)
def test_bruker_dda(shared_datadir, centroiding):
    bruker_dda_path = shared_datadir / "DDPASEF_10seconds.d"
    if not bruker_dda_path.exists():
        pytest.skip("Bruker DDA data not available")

    out = get_timstof_data(
        bruker_dda_path,
        min_peaks=5,
        progbar=False,
        centroid=centroiding,
        safe=True,
    )

    assert out is not None
    _check_polars_table(out, "DDA")


def test_bruker_dia(shared_datadir):
    bruker_dia_path = (
        shared_datadir / "230711_idleflow_400-1000mz_25mz_diaPasef_10sec.d"
    )
    if not bruker_dia_path.exists():
        pytest.skip("Bruker DIA data not available")

    out = get_timstof_data(
        bruker_dia_path,
        min_peaks=5,
        progbar=False,
        centroid=False,
        safe=True,
    )

    assert out is not None
    _check_polars_table(out, "DIA")


def test_mzml_dda(shared_datadir):
    mzml_dda_path = shared_datadir / "DDPASEF_10seconds.mzml"

    out = get_mzml_data(
        mzml_dda_path,
        min_peaks=5,
        progbar=False,
    )

    assert out is not None
    _check_polars_table(out, "DDA")


def test_mzml_dia(shared_datadir):
    mzml_dia_path = (
        shared_datadir / "230711_idleflow_400-1000mz_25mz_diaPasef_10sec.mzml"
    )

    out = get_mzml_data(
        mzml_dia_path,
        min_peaks=5,
        progbar=False,
    )

    assert out is not None
    _check_polars_table(out, "DIA")

