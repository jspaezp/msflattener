from rich_click import click as click

# msflattener
# msflattener --to_mzml
# msflattener --no_collapse_ims/collapse_ims
# msflattener --min_peaks 10

@click.command()
@click.option('--min_peaks', default=10, help='Minimum number of peaks to keep a spectrum')
@click.option('--file', type=click.Path(exists=True), help="File to use as input")
def main(min_peaks, file):
    pass

@click.command()
@click.option('--min_peaks', default=10, help='Number of greetings.')
@click.option('--file', prompt='Your name',
              help='The person to greet.')
def convert_to_mzml():
    pass

