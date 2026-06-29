"""autoresearch train.py — the single file the loop edits.

Intent: MAXIMIZE ROC-AUC for PHA (potentially-hazardous asteroid) from orbit
GEOMETRY alone. Direction: max. Leakage rule held throughout: moid, h, diameter,
albedo are the PHA definition + size proxies and are never features. Engineering
geometry from the orbital elements the model already has (Tisserand, Earth-cross
distances) is allowed — that IS the task.

Metric: 5-fold stratified CV mean ROC-AUC, same folds (seed 42) every run, on the
full 42k-row catalogue. 42k rows make CV stable; the margin to "keep" is +0.002.

Edit only EXP_DESC, engineer_features(), and build_model(). Everything below the
marker stays put.
"""
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ============================ EXPERIMENT (edit me) ============================
EXP_DESC = "baseline HistGradientBoosting (champion config)"


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, stateless transform of the orbit-element frame. Keep 'class'
    as a string column; build_model() one-hot encodes it per fold."""
    return X


def build_model():
    """Unfitted sklearn estimator with predict_proba. Any fit-stateful step
    (scaling, encoding, selection) lives here so it fits per CV fold."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["class"])],
        remainder="passthrough")
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, class_weight="balanced", random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])
# ========================== end experiment section ===========================


HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache.parquet"
MODEL_DIR = HERE / "model"
FG_NAME = "neo_features"
LABEL = "pha_label"
# Excluded: ids, raw flags, the PHA definition (moid, h) + size proxies
# (diameter, albedo), and tp (an epoch). All lowercase — Hopsworks lowercases
# feature names, so an uppercase entry silently misses and leaks.
NON_FEATURES = {"spkid", "full_name", "neo", "pha", "pha_label",
                "moid", "h", "diameter", "albedo", "tp"}


def load_data():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import hopsworks
    fs = hopsworks.login().get_feature_store()
    df = fs.get_feature_group(FG_NAME, version=1).read(dataframe_type="pandas")
    df.to_parquet(CACHE)
    return df


def make_card_images(model, X, y):
    """ROC / PR / confusion from a single stratified holdout, for the model
    card. Cheap (one fit); the CV mean above is the metric of record."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay,
                                 ConfusionMatrixDisplay, confusion_matrix)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=42)
    m = build_model().fit(Xtr, ytr)
    prob = m.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    RocCurveDisplay.from_predictions(yte, prob); plt.title("ROC — PHA")
    plt.savefig(MODEL_DIR / "roc_curve.png", bbox_inches="tight", dpi=110); plt.close()
    PrecisionRecallDisplay.from_predictions(yte, prob); plt.title("PR — PHA (6% positive)")
    plt.savefig(MODEL_DIR / "pr_curve.png", bbox_inches="tight", dpi=110); plt.close()
    ConfusionMatrixDisplay(confusion_matrix(yte, pred),
                           display_labels=["safe", "hazardous"]).plot()
    plt.title("Confusion"); plt.savefig(MODEL_DIR / "confusion_matrix.png",
                                        bbox_inches="tight", dpi=110); plt.close()


def main():
    t0 = time.time()
    df = load_data()
    feat_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = engineer_features(df[feat_cols].copy())
    y = df[LABEL].astype(int)

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(build_model(), X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    val_metric = scores.mean()
    print(f"cv_folds: {' '.join(f'{s:.4f}' for s in scores)} (std {scores.std():.4f})")

    MODEL_DIR.mkdir(exist_ok=True)
    model = build_model().fit(X, y)
    import joblib
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "exp": EXP_DESC, "val_metric": float(val_metric),
        "features": list(X.columns), "n_features_in": X.shape[1]}, indent=2))
    make_card_images(model, X, y)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"val_metric: {val_metric:.4f}")
    print(f"peak_memory_gb: {peak_gb:.3f}")
    print(f"training_seconds: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
