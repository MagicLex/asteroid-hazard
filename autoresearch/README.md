# Model optimization run: Gaia spectrum → albedo

Autonomous model search over the joined feature view `asteroid_albedo_fv` (Gaia
DR3 reflectance × NEOWISE albedo), in the
[hopsworks-autoresearch](https://github.com/MagicLex/hopsworks-autoresearch) style.
The loop edits one `train.py`, runs a fixed-budget experiment, keeps the change if
the metric clears a margin and reverts otherwise, recording every run in two
Hopsworks surfaces:

- **Leaderboard**: feature group `autoresearch_experiments_albedo`, one row per experiment.
- **Versions**: model registry `autoresearch_albedo`, one version per experiment
  (keep AND discard), each with `val_metric` so the registry charts the run.

Metric: **5-fold CV R²** for predicting `log10(albedo)` from the 16-band spectrum,
same folds (seed 42) every run, on the 21,046 joined asteroids. Keep margin +0.003.

## The run

| step | CV R² | decision |
|---|---:|---|
| baseline HistGradientBoosting, raw 16 bands | 0.5932 | keep |
| **XGBoost, raw 16 bands** (n700, depth5) | **0.5971** | keep (best) |
| XGBoost + spectral features (slopes, ratios, 1 µm band depth) | 0.5989 | discard |
| XGBoost bigger (n1400, depth6) | 0.5948 | discard |

Best model: **XGBoost on the raw 16 bands, CV R² 0.597**, which halves the size
error of the blind constant-albedo guess (diameter error ×1.34 → ×1.13).

## The honest finding: the win was the data, not the tuning

Two discards tell the real story:

- **Spelling out the spectral physics did not help** (+0.0018, below margin).
  Slopes (S-types are red, C-types flat) and the 0.9 µm silicate band depth are
  genuine taxonomic signal, but a gradient-boosted tree already reconstructs them
  from the raw bands through its splits; the raw spectrum already carries everything.
- **More capacity overfit** (0.5948). The ceiling is ~0.60, set by albedo
  measurement noise (Gaia and NEOWISE both) and the intrinsic albedo spread within
  a taxonomic type, not by model power.

So the entire lift came from the **data**: Gaia reflectance spectra joined to
NEOWISE albedo in the feature view. Model choice and feature engineering moved the
metric by hundredths; the spectrum itself is what reaches ×1.13 on diameter. When a
target is signal-thin, change the signal, not the model.

## Files

- `train.py`: the single file the loop edits (`EXP_DESC`, `engineer_features`,
  `build_model`). Reads `asteroid_albedo_fv`, caches it to `data_cache.parquet`.
- `log_row.py` / `progression.py` / `log_exp.sh`: leaderboard insert, run chart,
  and per-experiment model registration.

## Reproduce

```bash
python autoresearch/train.py > autoresearch/run.log 2>&1   # one experiment
bash autoresearch/log_exp.sh keep "my change"              # record it
```
