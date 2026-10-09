# UPVNet: Uncertainty-Aware Probabilistic Photovoltaic Power Forecasting Network

Reference implementation of **UPVNet**, a probabilistic photovoltaic (PV) power
forecasting network that treats numerical weather prediction (NWP) inputs as
Gaussian distributions, attends over them in distribution space, and fuses them
with the historical power series through an asymmetric bidirectional
cross-attention module.

Published at **PRICAI 2026** (see [Citation](#citation)).

## Method

UPVNet has five components, all in [`src/upvnet/model.py`](src/upvnet/model.py):

| Component | Class | Role |
|---|---|---|
| Probabilistic NWP encoder | `NWPEngine` | Maps each NWP time step to a diagonal Gaussian `N(mu, diag(sigma^2))`; `sigma` is kept positive by a `Softplus`-style ELU offset (`CovPositive`). |
| Wasserstein probabilistic self-attention | `ProbAttention` | Scores attention by the 2-Wasserstein distance between diagonal Gaussians, `D_W(i,j) = ||mu_i - mu_j||^2 + ||sigma_i - sigma_j||^2`, so both mean deviation and uncertainty discrepancy drive the weights. Aggregated variance follows `sigma^2 = sum_j alpha_ij^2 sigma_j^2`. |
| Convolutional residual power encoder | `CNNRsBlock` | Two 1-D convolutions (kernel 3, same padding, ReLU) plus a residual connection, LayerNorm, and a 1x1 projection for channel alignment. |
| Asymmetric bidirectional cross-attention | `CrossAttention` x2 | Two independent directional pathways: Power-queries-NWP and NWP-queries-Power. Outputs are concatenated, LayerNormed and linearly projected. |
| Reflected grid basis function decoder | `RGBFLayer` | Expands the fused representation over `K = 8` uniformly spaced grid nodes using `phi_k(x) = 1 - tanh^2((x - g_k)/delta)`. Two heads then produce the mean `mu` and the variance `sigma^2` (via `Softplus`). |

### Training objective

```
Loss = MSE(mu, y) + lambda * NLL(mu, sigma^2, y) + alpha * mean(sigma)
```

`lambda` is warmed up: the first `mse_pretrain_epochs` epochs are MSE-only, then
`lambda` ramps linearly to `nll_weight_max`. The `alpha * mean(sigma)` term is a
sharpness penalty that prevents the predicted interval from collapsing into a
degenerate wide band. Defaults (used for all reported results):
`mse_pretrain_epochs=5`, `nll_warmup_epochs=10`, `nll_weight_max=0.1`,
`sharpness_alpha=1e-3`, variance clamped to `[1e-2, 1e3]`, gradient norm
clipped at `5.0`.

## Datasets

Both datasets are public; this repository does **not** redistribute them.

| Dataset | Source | Hemisphere | Samples | Period | Resolution |
|---|---|---|---|---|---|
| `station01` | [PVOD](https://doi.org/10.1016/j.solener.2021.09.050) | Northern (Hebei, China) | 33,408 | 2018-07 to 2019-06 | 15 min |
| `23-Calyxo` | [DKASC](https://dkasolarcentre.com.au/) | Southern (Alice Springs, Australia) | 544,983 | 2010-02 to 2025-09 | 5 min -> 15 min |

Six NWP features are used, selected by XGBoost feature importance: GHI, DHI,
temperature, humidity, wind speed and wind direction.

Preprocessing (`src/preprocess/common_preprocess.py`) applies, in order:
timestamp de-duplication and alignment to a 15-minute grid; a physical
night-time mask (GHI < 10 W/m^2 forces power to 0); linear interpolation with
`limit=4`; cyclical hour/month encodings; a hemisphere-aware season label; a
`K=4` K-Means weather-condition label over (GHI, temperature, humidity); and a
chronological **7:1:2** train/val/test split (70% / 10% / 20%).

The final Z-score standardisation has two modes, selected with `--scale-scope`:

| Mode | Statistics fitted on | Notes |
|---|---|---|
| `full` (default) | the whole series, before splitting | Reproduces the published results. A standard convention in time-series forecasting. |
| `train` | the train split only | Strict anti-leakage. Produces a ~6% smaller standardised scale, so absolute MAE/RMSE are not comparable with the paper. |

`R^2` and the correlation coefficient are unaffected by this choice, and all
comparisons reported in the paper are between models trained on identical data,
so they remain valid under either mode. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the measured difference.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch 2.0+ is required. The reference runs used a single NVIDIA GPU (CUDA);
CPU inference works but training will be slow.

## Reproducing the results

1. **Get the raw data.** Download `station01` from PVOD and `23-Calyxo` from
   DKASC (links above) and place the CSV files in `data/`. The expected source
   column names are documented in `src/preprocess/common_preprocess.py`
   (`build_common_from_pvod`, `build_common_from_dka`).

2. **Preprocess.** This writes `preprocessed/<dataset>/{train,val,test}.csv`.

   ```bash
   python -m src.preprocess.common_preprocess
   ```

3. **Train and test.** For example, station01 at a 15-minute horizon:

   ```bash
   python scripts/train_upvnet.py --dataset station01 --pred-len 1
   ```

   Results land in `results/<dataset>_seq96_pred<len>/`: `summary.json`
   (configuration, parameter counts, test metrics), `history.json` (per-epoch
   curves) and `test_predictions.npz` (true / predicted mean / predicted
   variance, for recomputing probabilistic metrics or drawing fan charts).

   Horizons used in the paper: `--pred-len 1` (15 min), `4` (1 h), `24` (6 h),
   `96` (24 h). You can also pass `--horizon 15min|1h|6h|24h`.

   `--dataset` must match the folder name under `preprocessed/`. The two
   datasets used in the paper are `station01` and `23-Calyxodataset`.

Everything the model needs is pinned by default in
[`src/upvnet/train.py`](src/upvnet/train.py) (`TrainConfig`), so no extra flags
are required to match the released configuration.

## Evaluation metrics

`src/upvnet/metrics.py` computes everything reported in the paper.

* **Deterministic:** MAE, MSE, RMSE, MAPE, R^2, Pearson correlation.
* **Normalised:** nMAE, nRMSE (as a percentage of the mean of the target).
* **Probabilistic:** CRPS (Gaussian closed form), PICP and MPIW at the 95%
  level, and Winkler Score, with the penalty coefficient `2/alpha`.

Note: deterministic metrics are reported on Z-score standardised power, whereas
the probabilistic comparison is computed in the original physical power scale.
See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the value scale of each table.

## Repository layout

```
src/upvnet/model.py       UPVNet architecture
src/upvnet/train.py       Trainer (composite loss, warmup, early stopping) + run entry point
src/upvnet/dataset.py     Sliding-window sequence construction
src/upvnet/metrics.py     Deterministic + probabilistic metrics
src/preprocess/           Raw-data cleaning, feature engineering, split, standardisation
scripts/train_upvnet.py   Command-line entry point
configs/                  Preprocessing configuration
```

## Citation

```bibtex
@inproceedings{tang2026upvnet,
  title     = {UPVNet: Uncertainty-Aware Probabilistic Photovoltaic Power Forecasting Network},
  author    = {Tang, Shouyuan and Sun, Lingcui and Xiang, Jinbao and Wang, Mengjun and
               Liu, Yichen and Xu, Haoran and Chen, Hao and Zhu, Shiyu},
  booktitle = {Pacific Rim International Conference on Artificial Intelligence (PRICAI)},
  year      = {2026}
}
```

## Acknowledgements

The reflected grid basis function layer (`SplineLinear`,
`ReflectionalSwitchFunction`, `RGBFLayer`) follows the FasterKAN formulation by
A. Delis. The baseline forecasters compared against in the paper (LSTM, GRU,
BiLSTM, BiGRU, CNN, CNN-LSTM, TCN, Transformer, Informer, Autoformer) are not
bundled here; please obtain them from their original repositories.

This work was supported by the Science and Technology Research Program of the
Chongqing Municipal Education Commission, the Higher Education Scientific
Research Project of the Chongqing Association of Higher Education, and the
Intramural Scientific Research Project of Chongqing Institute of Engineering.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
