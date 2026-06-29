"""autoresearch train.py — the single file the loop edits.

Intent: MAXIMIZE 5-fold CV R² for predicting log10(albedo) from the Gaia DR3
16-band reflectance spectrum. Direction: max. The lever here is spectral physics:
slopes (S-types are red, C-types flat) and band depths (silicate absorption near
0.9 µm) — engineer them from the raw bands, never from the albedo itself.

Metric: 5-fold CV R² on the 21k joined asteroids, same folds (seed 42) every run.
Margin to "keep": +0.003. Also reports the diameter error factor (D ∝ 1/√albedo),
the honest headline: beat the blind constant-albedo guess (×1.34).

Edit only EXP_DESC, engineer_features(), and build_model(). Below the marker stays.
"""
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ============================ EXPERIMENT (edit me) ============================
EXP_DESC = "XGBoost on raw 16 bands (n700, depth5, lr0.03)"


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise transform of the 16 reflectance bands. Add spectral features here
    (slopes, ratios, band depths) — pure arithmetic on the bands, never the label."""
    return X


def build_model():
    """Unfitted sklearn regressor with predict()."""
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        random_state=42, n_jobs=-1)
# ========================== end experiment section ===========================


HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache.parquet"
MODEL_DIR = HERE / "model"
LABEL = "albedo"
BANDS = [374, 418, 462, 506, 550, 594, 638, 682, 726, 770, 814, 858, 902,
         946, 990, 1034]
BAND_COLS = [f"r{b}" for b in BANDS]


def load_data():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import hopsworks
    fs = hopsworks.login().get_feature_store()
    X, y = fs.get_feature_view("asteroid_albedo_fv", version=1).training_data()
    df = X[BAND_COLS].copy()
    df[LABEL] = y[LABEL].astype(float).values
    df = df.dropna(); df = df[df[LABEL] > 0]
    df.to_parquet(CACHE)
    return df


def diam_error_factor(log_pred, log_true):
    return float(np.median(10 ** (0.5 * np.abs(log_pred - log_true))))


def main():
    t0 = time.time()
    df = load_data()
    X = engineer_features(df[BAND_COLS].copy())
    y = np.log10(df[LABEL].values)

    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import r2_score
    cv = KFold(5, shuffle=True, random_state=42)
    pred = cross_val_predict(build_model(), X, y, cv=cv, n_jobs=-1)
    val_metric = r2_score(y, pred)
    dfac = diam_error_factor(pred, y)
    base = diam_error_factor(np.full_like(y, y.mean()), y)
    print(f"cv_r2: {val_metric:.4f}  diam_error: x{dfac:.3f} (baseline x{base:.3f})")

    MODEL_DIR.mkdir(exist_ok=True)
    model = build_model().fit(X, y)
    import joblib
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "exp": EXP_DESC, "val_metric": float(val_metric),
        "diam_error_factor": dfac, "n_features_in": X.shape[1]}, indent=2))
    _benchmark_plot(y, pred)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"val_metric: {val_metric:.4f}")
    print(f"peak_memory_gb: {peak_gb:.3f}")
    print(f"training_seconds: {time.time() - t0:.2f}")


def _benchmark_plot(y, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = diam_error_factor(np.full_like(y, y.mean()), y)
    mdl = diam_error_factor(pred, y)
    plt.figure(figsize=(5.5, 4))
    bars = plt.bar(["blind formula", "Gaia spectrum"], [base, mdl],
                   color=["#6b7280", "#f59e0b"])
    for b, v in zip(bars, [base, mdl]):
        plt.text(b.get_x() + b.get_width() / 2, v, f"x{v:.2f}", ha="center",
                 va="bottom", fontweight="bold")
    plt.ylabel("median diameter error (factor)")
    plt.title("Size error vs the blind formula")
    plt.ylim(1.0, base * 1.15)
    plt.savefig(MODEL_DIR / "diameter_error_benchmark.png", bbox_inches="tight", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
