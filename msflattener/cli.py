import rich_click as click

from .bruker import get_timstof_data, centroid_ims, merge_ims_simple, merge_ims_twostep
from .mzml import get_mzml_data, write_mzml

# msflattener
# msflattener --to_mzml
# msflattener --no_collapse_ims/collapse_ims
# msflattener --min_peaks 10


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--file", type=click.Path(exists=True), help="File to use as input", required=True
)
@click.option(
    "--output",
    type=click.Path(exists=False),
    help="Name of the output file",
    required=True,
)
@click.option("--to_mzml", "out_format", flag_value="mzml", default=False)
@click.option("--to_parquet", "out_format", flag_value="parquet", default=True)
@click.option(
    "--min_peaks", default=10, help="Minimum number of peaks to keep a spectrum"
)
@click.option(
    "--progbar/--no-progbar",
    " /-S",
    default=True,
    help="Whether to show progress bars.",
)
def bruker(file, output, out_format, min_peaks, progbar):
    dat = get_timstof_data(file, min_peaks=min_peaks, progbar=progbar)
    # dat = merge_ims_simple(dat,min_neighbors=15, mz_distance=0.01, progbar=progbar)
    # dat = merge_ims_simple(dat,min_neighbors=15, mz_distance=0.01, ims_distance=0.01, progbar=progbar)
    dat = centroid_ims(
        dat, min_neighbors=2, mz_distance=0.02, ims_distance=0.02, progbar=progbar
    )
    if out_format == "parquet":
        dat.to_parquet(output)
    elif out_format == "mzml":
        write_mzml(dat, output)
    else:
        raise RuntimeError


@cli.command()
@click.option(
    "--file",
    prompt="File Name",
    type=click.Path(exists=True),
    help="The person to greet.",
)
@click.option("--output", type=click.Path(exists=False), help="Name of the output file")
@click.option("--min_peaks", default=10, help="Number of greetings.")
@click.option(
    "--progbar/--no-progbar",
    " /-S",
    default=False,
    help="Whether to show progress bars.",
)
def mzml(file, output, min_peaks, progbar):
    dat = get_mzml_data(str(file), min_peaks=min_peaks, progbar=progbar)
    dat.write_parquet(output)


if __name__ == "__main__":
    cli()
