import rich_click as click
from loguru import logger

from .bruker import get_timstof_data
from .encyclopedia import write_encyclopedia
from .mzml import get_mzml_data, write_mzml


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
@click.option("--to_dia", "out_format", flag_value="dia", default=True)
@click.option(
    "--progbar/--no-progbar",
    " /-S",
    default=True,
    help="Whether to show progress bars.",
)
@click.option(
    "--centroid/--no-centroid",
    default=True,
    help="Whether to centroid the IMS dimension.",
)
def bruker(file, output, out_format, progbar, centroid):
    dat = get_timstof_data(file, progbar=progbar, centroid=centroid)
    if out_format == "parquet":
        logger.info("Writing parquet")
        dat.write_parquet(output)
    elif out_format == "mzml":
        logger.info("Writing mzML")
        write_mzml(dat, output)
    elif out_format == "dia":
        logger.info("Writing encyclopedia dia file")
        write_encyclopedia(dat, output, orig_filepath=file)
    else:
        raise RuntimeError


@cli.command()
@click.option(
    "--file",
    prompt="File Name",
    type=click.Path(exists=True),
    help="The file to read data from!",
)
@click.option("--output", type=click.Path(exists=False), help="Name of the output file")
@click.option(
    "--progbar/--no-progbar",
    " /-S",
    default=False,
    help="Whether to show progress bars.",
)
def mzml(file, output, progbar):
    dat = get_mzml_data(str(file), progbar=progbar)
    dat.write_parquet(output)


if __name__ == "__main__":
    cli()
