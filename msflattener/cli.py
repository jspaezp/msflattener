from rich_click import click as click

# msflattener
# msflattener --to_mzml
# msflattener --no_collapse_ims/collapse_ims
# msflattener --min_peaks 10


@click.group()
def cli():
    pass

@cli.command()
@click.option('--file', type=click.Path(exists=True), help="File to use as input")
@click.option('--output', type=click.Path(exists=False), help="Name of the output file")
@click.option('--to_mzml', 'out_format', flag_value='upper',
              default=True)
@click.option('--to_parquet', 'out_format', flag_value='lower')
@click.option('--min_peaks', default=10, help='Minimum number of peaks to keep a spectrum')
def bruker(file, out_format, min_peaks):
    pass

@cli.command()
@click.option('--file', prompt='Your name',
              help='The person to greet.')
@click.option('--min_peaks', default=10, help='Number of greetings.')
def mzml():
    pass

