"""Training pipeline (T stage) - runs as a Hopsworks job.

Builds the feature view that JOINS Gaia reflectance (features) to NEOWISE albedo
(label) on the asteroid number -- the join lives in the feature store, not in a
pre-baked table -- then trains a regressor that predicts albedo from the 16-band
Gaia spectrum. The headline result is the honest benchmark: does the spectrum beat
the blind constant-albedo assumption everyone falls back to when albedo is
unmeasured (97% of asteroids)? It does, and by a real margin.

albedo -> diameter is the formula D = 1329 * 10^(-H/5) / sqrt(albedo), so halving
the albedo error directly tightens every size (and impact-energy) estimate.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hopsworks

FG_REFL = "asteroid_reflectance"
FG_ALB = "asteroid_albedo"
FV_NAME = "asteroid_albedo_fv"
MODEL_NAME = "asteroid_albedo"
LABEL = "albedo"
BANDS = [374, 418, 462, 506, 550, 594, 638, 682, 726, 770, 814, 858, 902,
         946, 990, 1034]
BAND_COLS = [f"r{b}" for b in BANDS]
OUT = Path("artifact").resolve()


def get_feature_view(fs):
    refl = fs.get_feature_group(FG_REFL, version=1)
    alb = fs.get_feature_group(FG_ALB, version=1)
    # The JOIN: reflectance bands + albedo label, matched on `number`.
    query = refl.select(BAND_COLS).join(alb.select([LABEL]), on=["number"])
    fv = fs.get_or_create_feature_view(
        name=FV_NAME, version=1,
        description="Gaia 16-band reflectance (features) joined to NEOWISE albedo "
                    "(label) on asteroid number, for spectrum->albedo prediction.",
        query=query, labels=[LABEL])
    return fv


def build_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=5,
        l2_regularization=1.0, random_state=42)


def diam_error_factor(log_pred, log_true):
    """Median factor error on diameter. D ∝ 1/sqrt(albedo), so a dex error e on
    log-albedo is a 10^(e/2) factor on diameter."""
    return float(np.median(10 ** (0.5 * np.abs(log_pred - log_true))))


def make_plots(y, pred_cv, const_cv, imp_bands):
    OUT.mkdir(parents=True, exist_ok=True)
    alb_t, alb_p = 10 ** y, 10 ** pred_cv

    # 1) predicted vs measured albedo
    plt.figure(figsize=(5, 5))
    plt.scatter(alb_t, alb_p, s=4, alpha=0.25, color="#f59e0b")
    lim = [0.01, 1.0]
    plt.plot(lim, lim, color="#3b82f6", lw=1)
    plt.xscale("log"); plt.yscale("log"); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("measured albedo (NEOWISE)"); plt.ylabel("predicted albedo (Gaia spectrum)")
    plt.title("Albedo: predicted vs measured")
    plt.savefig(OUT / "albedo_pred_vs_true.png", bbox_inches="tight", dpi=120); plt.close()

    # 2) THE headline: diameter error, blind formula vs spectrum model
    base = diam_error_factor(const_cv, y)
    mdl = diam_error_factor(pred_cv, y)
    plt.figure(figsize=(5.5, 4))
    bars = plt.bar(["blind formula\n(constant albedo)", "Gaia spectrum\n(this model)"],
                   [base, mdl], color=["#6b7280", "#f59e0b"])
    for b, v in zip(bars, [base, mdl]):
        plt.text(b.get_x() + b.get_width() / 2, v, f"×{v:.2f}", ha="center",
                 va="bottom", fontsize=12, fontweight="bold")
    plt.ylabel("median diameter error (factor)")
    plt.title("Size error: spectrum beats the blind formula")
    plt.ylim(1.0, base * 1.15)
    plt.savefig(OUT / "diameter_error_benchmark.png", bbox_inches="tight", dpi=120); plt.close()

    # 3) residual histogram (log-albedo dex)
    plt.figure(figsize=(5.5, 4))
    plt.hist(pred_cv - y, bins=50, color="#f59e0b", alpha=0.85)
    plt.axvline(0, color="#3b82f6", lw=1)
    plt.xlabel("log10(albedo) residual (dex)"); plt.ylabel("asteroids")
    plt.title("Residuals (CV)")
    plt.savefig(OUT / "residuals.png", bbox_inches="tight", dpi=120); plt.close()

    # 4) which wavelengths carry the signal
    order = np.argsort(imp_bands)
    plt.figure(figsize=(6, 5))
    plt.barh([f"{BANDS[i]} nm" for i in order], imp_bands[order], color="#f59e0b")
    plt.xlabel("drop in R² when shuffled"); plt.title("Reflectance bands driving albedo")
    plt.savefig(OUT / "band_importance.png", bbox_inches="tight", dpi=120); plt.close()
    return base, mdl


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fv = get_feature_view(fs)

    X_raw, y_df = fv.training_data()
    df = X_raw[BAND_COLS].copy()
    df[LABEL] = y_df[LABEL].astype(float).values
    n0 = len(df)
    df = df.dropna()
    df = df[df[LABEL] > 0]
    X = df[BAND_COLS]
    y = np.log10(df[LABEL].values)
    print(f"training rows={len(X)} (dropped {n0 - len(X)} with no/zero albedo "
          "label after the join)", flush=True)

    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_absolute_error
    cv = KFold(5, shuffle=True, random_state=42)
    pred_cv = cross_val_predict(build_model(), X, y, cv=cv, n_jobs=-1)
    const_cv = np.full_like(y, y.mean())

    metrics = {
        "r2_cv": round(float(r2_score(y, pred_cv)), 4),
        "albedo_mae_dex_cv": round(float(mean_absolute_error(y, pred_cv)), 4),
        "diam_error_factor_model": round(diam_error_factor(pred_cv, y), 4),
        "diam_error_factor_baseline": round(diam_error_factor(const_cv, y), 4),
        "n_train": int(len(X)),
    }
    print("metrics:", json.dumps(metrics), flush=True)

    model = build_model().fit(X, y)
    imp = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="r2")
    base, mdl = make_plots(y, pred_cv, const_cv, imp.importances_mean)

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    mr = project.get_model_registry()
    model_schema = ModelSchema(Schema(X), Schema(df[[LABEL]]))
    m = mr.python.create_model(
        name=MODEL_NAME, metrics=metrics,
        description="Predict asteroid visible albedo from the Gaia DR3 16-band "
                    "reflectance spectrum. Beats the blind constant-albedo size "
                    f"estimate (diameter error ×{base:.2f} -> ×{mdl:.2f}).",
        input_example=X.head(1), model_schema=model_schema, feature_view=fv)
    m.save(str(OUT))
    print(f"registered {MODEL_NAME} v{m.version}", flush=True)


if __name__ == "__main__":
    main()
