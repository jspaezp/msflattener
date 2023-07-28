import pytest

from msflattener.bruker import get_timstof_data
from msflattener.encyclopedia import write_encyclopedia


@pytest.mark.parametrize(
    "centroiding",
    [True, False],
)
def test_bruker_dia(shared_datadir, centroiding, tmp_path):
    bruker_dia_path = (
        shared_datadir / "230711_idleflow_400-1000mz_25mz_diaPasef_10sec.d"
    )
    if not bruker_dia_path.exists():
        pytest.skip("Bruker DIA data not available")

    out = get_timstof_data(
        bruker_dia_path,
        progbar=False,
        centroid=centroiding,
        safe=True,
    )

    outpath = tmp_path / "test_encyclopedia_dia.sqlite.dia"
    write_encyclopedia(out, outpath, "asdad/foo.d")
