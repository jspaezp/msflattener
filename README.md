
# MSflattener

It flattens ms ...

# Current state

Right now the project is in a very early stage!
Feel free to let me know if you see utility in it and request features.

## Scope

The idea of this project is to provide a way to convert and export
timsTOF and mzML data to parquet. The output format is not meant to
contain all of the data in the original file, but to provide an
intermediate representation that is fast to read by other tools.

This approach is burrowed from the genomics world, where it is common
to have intermediate representatinos of your data that do not have all
information from the progenitor data (trimmed fastq files for example);
but instead focus on just being usable for the next tool in your analysis
pipeline.

## Rationale

timsTOF data is internally organized in a peak-wise fashion with
indexing that points to the conditions in which such peak was acquired.
I would like to have a representation where contiguous peaks that share
acquisition are stored together.

In contrast, an mzML file is organized in a scan-wise fashion, where
each scan contains a list of peaks. But suffers from the fact that it
needs to support every possible acquisition mode and thus has a lot of
redundant information in each scan.

Therefore this format attempts to provide a fast and easy to read format
that will work for most of the modern experimental workflows (DIA, DDA, PRM)

In addition a lot of the project I have seen that make similar
"decompressions" of the data, do so in a way that optimizes for random
access ("I want all peaks in x-y intensity range, in x-y mass range,
in x-y ims range, in this retention time"); and whilst this is useful,
it can make reading the data slower if you want to access it in a more
sequential fashion (making XICs of A LOT of species at once).

On top of the formerly mentioned benefits, this project makes use
of apache arrow and the parquet data format pretty extensively. These
specifications and formats are widely accepted in a broad range
of applications, therefore have been very heavily optimized and give
us many benefits out of the box (readability in any programming language,
compression, strict schema within columns, speed of read/write).

## Assumptions

The file format is meant to sacrifice some of the flexibility of the
mass spectrometer in exchange for speed. In particular it assumes that
the retention time will be monotonically increasing and that all scans
that have enabled quad isolation are being fragmented with the same
collision energy.

It also assumes there will only be ms1 and ms2 scans.

## TODO
- Add sliding window neighborhood in RT.
    - This means that neighbors are counted but not integrated from neighboring scans.
- Add parallel compute.

## Additional Utility

# Installation

```
pipx install git+https://github.com/talusbio/msflattener.git
```

# Usage

The idea is to make the project usable via docker or directly as a
python package/cli.

```shell
$ msflattener bruker --to_mzml --file somefile.d --output collapsed.mzML
$ msflattener bruker --to_parquet --file somefile.d --output collapsed.parquet
$ msflattener mzml --file myfile2.mzML --output collapsed2.parquet
```

```python
import polars as pl
pl.scan_parquet("collapsedhela_mp.parquet").head().collect()
# shape: (5, 6)
# ┌─────────────────┬─────────────────┬───────────┬────────────────┬────────────────┬────────────────┐
# │ mz_values       ┆ corrected_inten ┆ rt_values ┆ mobility_value ┆ quad_low_mz_va ┆ quad_high_mz_v │
# │ ---             ┆ sity_values     ┆ ---       ┆ s              ┆ lues           ┆ alues          │
# │ list[f64]       ┆ ---             ┆ f64       ┆ ---            ┆ ---            ┆ ---            │
# │                 ┆ list[u64]       ┆           ┆ list[f64]      ┆ f64            ┆ f64            │
# ╞═════════════════╪═════════════════╪═══════════╪════════════════╪════════════════╪════════════════╡
# │ [377.167734,    ┆ [612, 299, …    ┆ 1.331647  ┆ [0.847556,     ┆ -1.0           ┆ -1.0           │
# │ 835.493336, …   ┆ 564]            ┆           ┆ 0.79407, …     ┆                ┆                │
# │ 301.0…          ┆                 ┆           ┆ 0.791722]      ┆                ┆                │
# │ [283.36197,     ┆ [42, 154, …     ┆ 1.861262  ┆ [0.868832,     ┆ -1.0           ┆ -1.0           │
# │ 283.737158, …   ┆ 507]            ┆           ┆ 0.868445, …    ┆                ┆                │
# │ 934.26…         ┆                 ┆           ┆ 0.921157]      ┆                ┆                │
# │ [564.03779,     ┆ [2687, 147, …   ┆ 2.39146   ┆ [0.943754,     ┆ -1.0           ┆ -1.0           │
# │ 1270.632453, …  ┆ 514]            ┆           ┆ 0.883083, …    ┆                ┆                │
# │ 698.7…          ┆                 ┆           ┆ 0.795804]      ┆                ┆                │
# │ [283.760379,    ┆ [244, 260, …    ┆ 2.815393  ┆ [0.867842,     ┆ -1.0           ┆ -1.0           │
# │ 1089.13072, …   ┆ 682]            ┆           ┆ 0.998711, …    ┆                ┆                │
# │ 283.2…          ┆                 ┆           ┆ 0.863129]      ┆                ┆                │
# │ [283.158069,    ┆ [85, 282, …     ┆ 3.177816  ┆ [0.86572,      ┆ -1.0           ┆ -1.0           │
# │ 283.76007, …    ┆ 620]            ┆           ┆ 0.868674, …    ┆                ┆                │
# │ 313.23…         ┆                 ┆           ┆ 0.844221]      ┆                ┆                │
# └─────────────────┴─────────────────┴───────────┴────────────────┴────────────────┴────────────────┘
```

## Exporting to mzML

The mzML will not be compliant with the standard but should be good enough to be searched with a search engine.
If it does not, feel free to write an issue so we can figure out trogether why that is the case.

```python
from msflattener.mzml import write_mzml
write_mzml(two_d_merge, "foo.mzml")
```

## Acknowledgements

Open source projects that I would like to acknowledge:
- Alphatims
- PSIMS
- scipy
- polars
- numpy
