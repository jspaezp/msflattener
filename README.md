
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

## Additional Utility

# Usage

The idea is to make the project usable via docker or directly as a
python package/cli.

```shell
$ msflattener myfile.d
$ msflattener myfile.mzML
# Outputs myfile.flatms.parquet
```

```python
import polars as pl
pl.scan_parquet("myfile.flatms.parquet").head().collect()
```

## Collapsing Modes

I currently offer two collapsing modes, a simple one that will merge only using clusters
of peaks in the mz dimension and a 'centroid' mode that will merge peaks in both the
mz and the ims dimension.

```python
# Data from the SLICEpasef paper
df = get_timstof_data(
    "20221016_PRO1_LSVD_00_30-0051_100ng_Regular_P1-C6_1_11234.d",
    safe=True,
)
out.write_parquet("test.parquet")
out = pl.read_parquet("test.parquet")
print(out)


one_d_merge = merge_ims_simple(out, min_neighbors=5, mz_distance=0.01)
print(two_d_merge)
#┌───────────────────────────────────┬────────────────────────────┬────────────────────┬─────────────────────┬────────────┬─────────────────────┬──────────────────────┐
#│ mz_values                         ┆ corrected_intensity_values ┆ quad_low_mz_values ┆ quad_high_mz_values ┆ rt_values  ┆ mobility_low_values ┆ mobility_high_values │
#│ ---                               ┆ ---                        ┆ ---                ┆ ---                 ┆ ---        ┆ ---                 ┆ ---                  │
#│ list[f64]                         ┆ list[u64]                  ┆ f64                ┆ f64                 ┆ f64        ┆ f64                 ┆ f64                  │
#╞═══════════════════════════════════╪════════════════════════════╪════════════════════╪═════════════════════╪════════════╪═════════════════════╪══════════════════════╡
#│ [301.125211, 371.085488, … 932.6… ┆ [2934, 2534, … 588]        ┆ -1.0               ┆ -1.0                ┆ 0.554721   ┆ 0.767873            ┆ 1.270136             │
#│ [429.078869, 667.157179, … 830.6… ┆ [1552, 932, … 421]         ┆ -1.0               ┆ -1.0                ┆ 1.256356   ┆ 0.766063            ┆ 1.233032             │
#│ …                                 ┆ …                          ┆ …                  ┆ …                   ┆ …          ┆ …                   ┆ …                    │
#│ [863.685532, 762.636295, … 745.7… ┆ [1203, 483, … 333]         ┆ -1.0               ┆ -1.0                ┆ 467.120844 ┆ 0.751584            ┆ 1.236652             │
#│ [684.181934, 759.209893, … 1175.… ┆ [17265, 7369, … 518]       ┆ -1.0               ┆ -1.0                ┆ 467.832057 ┆ 0.747059            ┆ 1.235747             │
#└───────────────────────────────────┴────────────────────────────┴────────────────────┴─────────────────────┴────────────┴─────────────────────┴──────────────────────┘

two_d_merge = centroid_ims(out, min_neighbors=5, mz_distance=0.01, ims_distance=0.01)
print(two_d_merge)
# ┌───────────────────────────────────┬────────────────────────────┬──────────────────────────────────┬────────────────────┬─────────────────────┬────────────┐
# │ mz_values                         ┆ corrected_intensity_values ┆ mobility_values                  ┆ quad_low_mz_values ┆ quad_high_mz_values ┆ rt_values  │
# │ ---                               ┆ ---                        ┆ ---                              ┆ ---                ┆ ---                 ┆ ---        │
# │ list[f64]                         ┆ list[u64]                  ┆ list[f64]                        ┆ f64                ┆ f64                 ┆ f64        │
# ╞═══════════════════════════════════╪════════════════════════════╪══════════════════════════════════╪════════════════════╪═════════════════════╪════════════╡
# │ [684.188895, 371.089286, … 413.2… ┆ [12110, 1799, … 632]       ┆ [1.147135, 0.837198, … 1.018001] ┆ -1.0               ┆ -1.0                ┆ 0.554721   │
# │ [413.251921, 667.163663, … 366.9… ┆ [1497, 806, … 703]         ┆ [1.016177, 1.122811, … 0.812]    ┆ -1.0               ┆ -1.0                ┆ 1.256356   │
# │ …                                 ┆ …                          ┆ …                                ┆ …                  ┆ …                   ┆ …          │
# │ [409.258342, 762.203255, … 378.3… ┆ [2267, 1987, … 613]        ┆ [1.026119, 1.200416, … 0.89553]  ┆ -1.0               ┆ -1.0                ┆ 467.120844 │
# │ [759.207114, 684.190582, … 651.1… ┆ [6361, 17265, … 453]       ┆ [1.197485, 1.149633, … 1.11794]  ┆ -1.0               ┆ -1.0                ┆ 467.832057 │
# └───────────────────────────────────┴────────────────────────────┴──────────────────────────────────┴────────────────────┴─────────────────────┴────────────┘
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
