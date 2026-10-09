# Reproducibility notes

This document records exactly what was run to produce the numbers in the paper,
where the archived raw outputs live, and the points of care when comparing a
fresh run against the published tables.

## Environment

| Item | Value |
|---|---|
| Python | 3.12.2 |
| PyTorch | 2.3.1+cu121 |
| NumPy / pandas | >= 1.23 / >= 2.0 |
| GPU | single NVIDIA GPU (reference runs on an RTX 3090; a 4 GB RTX 3050 laptop GPU also completes the station01 runs) |
| Seeds | `torch`, `numpy` and `random` all seeded from `TrainConfig.seed` (default 42); DataLoader shuffling is seeded |

## Data pipeline

Order of operations in `src/preprocess/common_preprocess.py`:

1. Parse timestamps, de-duplicate, resample onto a uniform 15-minute grid.
2. Physical night mask: `GHI < 10 W/m^2` forces power to `0`.
3. Linear interpolation of short gaps, `limit=4` consecutive steps.
4. Cyclical features: `hour_sin/cos`, `month_sin/cos`.
5. Hemisphere-aware season label (`season_cluster`).
6. `K=4` K-Means over `(GHI, temperature, humidity)` -> `weather_cluster`.
7. Chronological split **7:1:2** — `train_end = int(n * 0.7)`,
   `val_end = int(n * 0.8)`.
8. Z-score standardisation. `--scale-scope` selects what the statistics are
   fitted on; see the section below.

Step 7 is stated explicitly because the repository's original `Readme.md`
described the split as "7:2:1". That description is wrong; the code and the
paper's Section 3.1 both use 70% / 10% / 20%. The split is chronological, not
shuffled, so the test set is the most recent 20% of each record.

#### Standardisation scope (`--scale-scope`)

The Z-score statistics are fitted on the **whole series, before splitting** by
default (`--scale-scope full`). This is the convention under which the published
results were produced, and it is a common one in time-series forecasting.

`--scale-scope train` fits on the train split only, which is the strictly
leakage-free alternative. Measured on station01:

| Quantity (raw power, kW) | Value |
|---|---|
| mean / std over the full series | 3.677611 / 5.552965 |
| mean / std over the train split | 3.307305 / 5.228447 |
| scale ratio (full / train) | 1.062068 |

So a test MAE of 0.0877 on the full-series scale corresponds to roughly 0.0825 on
the train-only scale — about a 5.8% difference. **Absolute MAE, MSE, RMSE, MAPE
and nMAE/nRMSE are therefore not comparable across the two modes.** `R^2` and
Pearson correlation are invariant to a global affine rescaling and are
unaffected.

This is a scale convention, not a difference in the models: every model in every
comparison table was trained and evaluated under one and the same convention, so
the comparative conclusions of the paper hold either way.

**Reproducing the published data.** Running the pipeline with the default
`--scale-scope full` reproduces the archived station01 split exactly in row
count and row order (23,385 / 3,341 / 6,682 = the 7:1:2 split of 33,408) and in
its standardisation statistics. Values agree to within 0.01 kW for 99.4% of
rows; the residual (209 of 33,408 rows, max 0.585 kW, i.e. ~0.1σ) comes from
minor gap-interpolation differences on a handful of long missing stretches.

## NWP feature selection

Six input features are used, chosen by XGBoost gain importance out of the raw
weather columns: `GHI`, `DHI`, `temperature`, `humidity`, `wind_speed`,
`wind_direction`.

**Point of care.** `dataset.infer_feature_cols` falls back to *every*
non-timestamp, non-power column when no explicit list is supplied — which would
silently pull in `hour_sin`, `hour_cos`, `month_sin`, `month_cos`,
`season_cluster` and `weather_cluster` as model inputs. Always pass the six
columns explicitly. `TrainConfig.nwp_features` does this by default, so the CLI
is safe; hand-written scripts must do the same.

## Model configuration

The released defaults in `TrainConfig` match the configuration used for the
paper:

```
seq_len = 96            pred_len = 1 | 4 | 24 | 96
cnnrs_channels = 64     rgbf_hidden = 64     attn_dim = 64
dropout = 0.1           grids (K) = 8
batch_size = 32         lr = 1e-3            epochs = 50
mse_pretrain_epochs = 5 nll_warmup_epochs = 10
nll_weight_max = 0.1    sharpness_alpha = 1e-3
```

Parameter count at `seq_len=96`, 6 NWP features:

```
total     109,258
trainable 109,250
frozen          8      <- rgbflayer.rbf.grid (the K=8 reflectional grid)
```

The 8 grid nodes are registered as an `nn.Parameter` with
`requires_grad=False`, so they appear in `model.parameters()` but not in
`parameters()` filtered on `requires_grad`. A report that prints **109,250** is
counting trainable parameters only; one that prints **109,258** is counting all
parameters. The difference is exactly the grid, and the two figures describe the
same model.

Optimiser: Adam, `ReduceLROnPlateau(mode="min", patience=5, factor=0.5)`,
gradient-norm clipping at 5.0. The loss is

```
Loss = MSE(mu, y) + lambda * NLL(mu, sigma^2, y) + alpha * mean(sigma)
```

with `lambda` = 0 for the first 5 epochs and then ramping linearly to 0.1 over
the next 10. Variance is clamped to `[1e-2, 1e3]` before entering the NLL, which
is why the training loss can go negative once `lambda > 0` — the NLL term is a
log-likelihood and is unbounded below. This is expected and affects no reported
metric, all of which are computed on the model outputs, not on the loss.

## Metric definitions

`src/upvnet/metrics.py`:

* MAE, MSE, RMSE, MAPE, R^2, Pearson correlation.
* nMAE and nRMSE are expressed as a percentage of the mean of the target.
* CRPS uses the Gaussian closed form
  `sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))`.
* PICP and MPIW are evaluated at the 95% level (`z = 1.96`).
* Winkler Score uses `penalty_factor = 2 / alpha` with `alpha = 0.05`.
* `SS` is the skill score against `ref_nrmse`.

These formulas were re-implemented independently in NumPy and reproduce the
archived CRPS / PICP / MPIW values to six decimal places.

## Value scale by table

Two scales appear in the paper and they must not be mixed up:

* **Standardised (Z-score) power** — the deterministic comparison tables
  (MAE / RMSE / R^2 / MAPE) are on this scale. This is what `TrainConfig`
  produces by default; the archived station01 H=1 run reports MAE 0.0877.
* **Physical power (kW)** — the probabilistic table is on this scale. The same
  model at the same horizon reports MAE 0.4425 there.

Both are correct and describe the same predictions; the probabilistic table
reports in the original units so that interval widths are interpretable in kW.

## Archived raw outputs

The full set of per-run artefacts behind every table and figure is preserved
under `paper/已中稿/paperdata/`, indexed by its own `README.md`:

```
01_datasets/                  sample counts and periods
02_deterministic_comparison/  test_results.json per baseline, per horizon
03_model_complexity/          parameter counts and per-epoch timings
04_probabilistic/             CRPS / PICP / MPIW / WS / NLL
05_ablation/                  the five ablation variants
06_weather_season/            per-weather and per-season breakdowns
07_robustness/                input-perturbation study
08_figures/                   figure source data
09_paper_source/              model definition and generation scripts
10_raw_prediction_csvs/       raw prediction dumps
```

This directory is intentionally excluded from the public repository (see
`.gitignore`), because it contains the manuscript source.

## Reproducing a single run

```bash
python scripts/train_upvnet.py --dataset station01 --pred-len 1 --epochs 50
```

Writes `results/station01_seq96_pred1/{summary.json,history.json,test_predictions.npz}`.
Compare `test_metrics` in `summary.json` against the archived values.

## Expected agreement

A fresh 50-epoch run of station01 H=1 lands within run-to-run noise of the
archived result. As a reference point, a shortened run (2 of 50 epochs) already
reaches test MAE 0.0907 / RMSE 0.1923 / R^2 0.9679 against the archived
0.0877 / 0.1982 / 0.9698. Expect agreement to about the third decimal for
MAE/RMSE and the fourth for R^2. MAPE is the least stable of these metrics
because near-zero night-time targets dominate its denominator; it moves by
several points between runs and should be read as indicative only.

## Scope of this release

* **The baselines are not bundled.** The ten comparison forecasters in the
  deterministic table (LSTM, GRU, BiLSTM, BiGRU, CNN, CNN-LSTM, TCN,
  Transformer, Informer, Autoformer) were trained in a separate codebase that is
  not part of this release. Obtain them from their original repositories; the
  archived `02_deterministic_comparison/` contains their raw test outputs for
  comparison.

* **Ablation folder names are the authority.** Variant identifiers in the
  manuscript and in the archived LaTeX table source differ from the folder names
  under `05_ablation/`. When tracing a reported ablation value back to a run,
  use the archived folder names.

* **Figures are not regenerated here.** Source data for the figures is under
  `08_figures/`; the plotting scripts are in the archived bundle rather than in
  this repository.
