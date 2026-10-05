# Datasets (not included in this repository)

Raw dataset binaries are **not redistributed** here. Please download from the
official sources and convert with the scripts in `scripts/`.

## SKAB (industrial testbed multivariate time series)

- Official repository: https://github.com/TIM-Rub/skab
  (historical mirror often cited: SKAB by Zolotov & Kim)
- Alternative citation/source page: see the SKAB paper
  *SKAB — Benchmark for Anomaly Detection in Multivariate Time Series*
- Download: clone the repository, place CSV files under `data/raw/SKAB/`
- Convert: `python scripts/convert_skab.py --src data/raw/SKAB --out data/skab/`

## SMD (Server Machine Dataset)

- Source: OmniAnomaly artifact
  https://github.com/NetManAIOps/OmniAnomaly
- Folder: `ServerMachineDataset/` (`train/`, `test/`, `test_label/`)
- Download: clone the repository and copy `ServerMachineDataset/` to
  `data/raw/SMD/`
- Convert: `python scripts/convert_smd.py --src data/raw/SMD --out data/smd/`

## MSL (Mars Science Laboratory / spacecraft telemetry)

- Original release (NASA / Telemanom):
  https://github.com/khundman/telemanom
  and the SMAP/MSL data package described in the OmniAnomaly README
  (`https://s3-us-west-2.amazonaws.com/telemanom/data.zip`)
- Practical copy used by several benchmarks (train/test CSV with labels):
  e.g. https://github.com/d-ailin/GDN (`data/msl/`) or equivalent
  SMAP/MSL CSV dumps from Telemanom-based pipelines
- Place `train.csv` / `test.csv` (and labels) under `data/raw/MSL/`
- Convert: `python scripts/convert_msl.py --src data/raw/MSL --out data/msl/`

## License note

Each dataset has its own license/terms of use. This repository does not grant
any additional rights to the data. Users must comply with the original licenses
before using the data in publications.
