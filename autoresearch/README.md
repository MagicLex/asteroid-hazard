# Model optimization run — asteroid PHA

Autonomous model-search over the NEO catalogue, in the
[hopsworks-autoresearch](https://github.com/MagicLex/hopsworks-autoresearch) style
(adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)).
The loop edits a single `train.py`, runs a fixed-budget experiment, keeps the
change if the metric clears a margin and reverts it otherwise, and records every
run in two Hopsworks surfaces:

- **Leaderboard**: feature group `autoresearch_experiments_astjun29`, one row per
  experiment (`commit`, `val_metric`, `peak_memory_gb`, `status`, `description`, `ts`).
- **Versions**: model registry `autoresearch_astjun29`, one version per experiment
  (keep AND discard), each with `val_metric` so the registry charts the run and the
  card carries ROC / PR / confusion + a run-progression image.

Metric: **5-fold stratified CV mean ROC-AUC**, same folds (seed 42) every run, on
the full 42,153-row catalogue. At that size CV is stable (fold std ~0.006), so the
keep margin is a modest **+0.002**.

Intent held throughout: predict PHA from orbit **geometry alone**. `moid`, `h`,
`diameter`, `albedo` (the PHA definition plus size proxies) are never features.
Deriving Earth-approach geometry *from the orbital elements the model already has*
is allowed — that is the task, not leakage.

## The run

| step | val_metric (CV) | decision |
|---|---:|---|
| baseline HistGradientBoosting (champion config) | 0.8668 | keep |
| XGBoost (n600, depth4, scale_pos_weight) | 0.8761 | keep |
| **+ orbital-mechanics geometry** | **0.9221** | keep |
| + deeper XGBoost (depth5, n900, lr0.025) | **0.9245** | keep (best) |
| + encounter velocity, sin(i), log node-distance | 0.9250 | discard |
| soft-vote XGB + HistGradientBoosting ensemble | 0.9243 | discard |

## The lever: geometry beats the raw elements

Swapping HGB for XGBoost bought +0.009. The real jump (+0.046, from 0.876 to
**0.922**) came from engineering Earth-approach geometry out of the orbital
elements:

- **`node_dist_min`** — the heliocentric distance where the asteroid's orbit
  crosses Earth's orbital plane (ascending/descending node), compared to 1 AU. This
  is the classic **MOID lower bound**: `r_node = a(1-e²)/(1 ± e·cos ω)`, then
  `min(|r_asc − 1|, |r_desc − 1|)`. The model was being asked to predict a flag
  defined by MOID; handing it a geometric MOID proxy (computed, never read from the
  `moid` column) is the honest way to give it the "comes close to Earth" signal.
- **`tisserand_earth`** — Tisserand parameter w.r.t. Earth, the encounter
  invariant `1/a + 2√(a(1−e²))·cos i`.
- **Earth-crossing band** — `q ≤ 1.017`, `ad ≥ 0.983`, and the apsis-to-1-AU gap.

The raw `a, e, i, q, ad` were already present; the model just could not derive the
node-crossing distance on its own through tree splits. Spelling it out as a feature
is the same lesson as #001 (raw README text beat the structural counts): the signal
was reachable, but only once the right transform exposed it.

## Why the last two were discarded

- **Encounter velocity / sin(i) / log node-distance** scored 0.9250 — numerically
  above 0.9245, but inside the fold noise band (std 0.006) and well under the
  +0.002 margin, while adding three features. Noise, not signal: discard.
- **XGB + HGB ensemble** scored 0.9243, no better than the lone XGBoost and more
  complex. Equal result, more code: discard (simplicity rule).

Best model: **XGBoost on orbital geometry, CV ROC-AUC 0.9245**, up from the 0.8668
baseline measured the same way — the project's served model is the simpler 0.86
geometry-only HGB; this run shows the orbit-geometry ceiling is ~0.92 once the
node-distance proxy is exposed, without ever touching the real MOID.

## Files

- `train.py`: the single file the loop edits. Only `EXP_DESC`,
  `engineer_features()`, and `build_model()` change between experiments.
- `log_row.py`: SDK insert for the leaderboard (the CLI cannot write a timestamp
  column from JSON).
- `progression.py`: builds the run-progression chart from the leaderboard FG.
- `log_exp.sh`: insert the row, build the chart, register the model version.

## Reproduce

From a Hopsworks terminal (first run caches the FG to `data_cache.parquet`):

```bash
python autoresearch/train.py > autoresearch/run.log 2>&1   # one experiment
bash autoresearch/log_exp.sh keep "my change"              # record it
```

Edit the experiment section of `train.py`, rerun, keep or revert.
