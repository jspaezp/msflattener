
from msflattener.bruker import get_timstof_data
from msflattener.mzml import get_mzml_data
from msflattener.base import SCHEMA_DDA, SCHEMA_DIA
import pytest

def test_bruker_dda(shared_datadir):
    bruker_dda_path = shared_datadir / "test.baf"
    if not bruker_dda_path.exists():
        pytest.skip("Bruker DDA data not available")

    out = get_timstof_data(
        bruker_dda_path,
        min_peaks=5,
        progbar=False,
        centroid=False,
        safe=True,
    )

    assert out is not None

def test_bruker_dia(shared_datadir):
    bruker_dia_path = shared_datadir / "test.baf"
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

def test_mzml_dda(shared_datadir):
    mzml_dda_path = shared_datadir / "test.mzML"

    out = get_mzml_data(
        mzml_dda_path,
        min_peaks=5,
        progbar=False,
    )

    assert out is not None

def test_mzml_dia(shared_datadir):
    mzml_dia_path = shared_datadir / "test.mzML"

    out = get_mzml_data(
        mzml_dia_path,
        min_peaks=5,
        progbar=False,
    )

    assert out is not None
