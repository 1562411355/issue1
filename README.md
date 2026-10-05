# Lightweight Score Distillation for Multivariate Time-Series Anomaly Detection

Reproducible evaluation of anomaly detectors under a **locked protocol**, with
score-distillation student models for edge-oriented deployment. All table
numbers are script-injected from `results/results.csv` and machine-audited.

## One-shot reproduction

```bash
git clone <this-repo-url>
cd <repo>
pip install -r requirements.txt
# optional deep baselines (TranAD / USAD):
pip install -r requirements_deep.txt

# 1) Obtain datasets (see data/README.md), then convert:
python scripts/convert_skab.py --src data/raw/SKAB --out data/skab/
python scripts/convert_smd.py  --src data/raw/SMD  --out data/smd/
python scripts/convert_msl.py  --src data/raw/MSL  --out data/msl/

# 2) Full grid (locked protocol; skip-resume via results/results.csv)
python run_all.py --workers 4

# 3) Analysis + verification
python analysis/analyze.py
python verify/verify.py
```

`verify.py` must report **FAIL 0** before any paper tables are trusted.

## Protocol summary (locked, `configs/protocol.md`)

| Item | Value |
|------|--------|
| Window / stride | 120 / 10 |
| Seeds | 0–6 (7 independent runs) |
| Normalization | z-score, train-segment stats only |
| Threshold | validation quantile grid, PA-F1 |
| Metrics | PA + non-PA P/R/F1, AUC-ROC, AUC-PR, params, CPU latency |
| SKAB split | chronological 60/20/20 (train / val / test) |
| SMD / MSL split | official train boundary; official test **stratified** val/test |
| Statistical tests | paired Wilcoxon vs `pca_recon`, Holm, α=0.05 |

Ablations (`ae_l_teacher`, `ae_s_nodistill`, λ∈{0.1,0.3,0.5}) and deep baselines
(`tranad`, `usad`) are registered in `configs/protocol.md` `hyperparams`.

## Repository layout

```
src/                  models, data loading, metrics
run_all.py            batch training / evaluation
analysis/analyze.py   leaderboard, stats, figures
verify/verify.py      number↔log audit gate
draft/inject_tables.py
configs/protocol.md   machine-readable locked protocol
results/results.csv   raw experiment rows (traceable)
scripts/              dataset conversion
data/README.md        official dataset download instructions
```

## Citation

```bibtex
@article{tbd2026,
  title   = {A Locked-Protocol Evaluation of Lightweight Score Distillation
             for Multivariate Time-Series Anomaly Detection},
  author  = {[AUTHOR-TO-FILL]},
  journal = {Electronics},
  year    = {2026},
  note    = {Code: [GITHUB-URL-TO-FILL]}
}
```

Paper DOI will be added after acceptance.

## License

Code: MIT (see `LICENSE`). Datasets: see `data/README.md` (not redistributed).
