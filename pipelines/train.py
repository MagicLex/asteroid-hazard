"""Training pipeline (T stage) - runs as a Hopsworks job.

Feature view over the orbit-geometry features (leakage excluded) -> PHA
classifier handling the 6% class imbalance -> evaluate -> register with plots.

Honesty: the feature view excludes moid, H, diameter, albedo (the PHA definition
plus size proxies) and tp/ids. The model sees orbital geometry + orbit class.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hopsworks

FG_NAME = "neo_features"
FV_NAME = "neo_pha_fv"
MODEL_NAME = "asteroid_pha"
LABEL = "pha_label"
OUT = Path("artifact").resolve()

# Excluded from the model: ids, the raw flags, the PHA definition (moid, H) and
# size proxies (diameter, albedo), and tp (an epoch, not predictive).
# NOTE: Hopsworks lowercases feature names, so these MUST be lowercase or the
# exclusion silently misses (e.g. "H" -> stored as "h" -> leaks the size half).
NON_FEATURES = {
    "spkid", "full_name", "neo", "pha", "pha_label",
    "moid", "h", "diameter", "albedo", "tp",
}
CATEGORICAL = ["class"]


def get_feature_view(fs):
    fg = fs.get_feature_group(FG_NAME, version=1)
    feature_cols = [f.name for f in fg.features if f.name.lower() not in NON_FEATURES]
    query = fg.select(feature_cols + [LABEL])
    fv = fs.get_or_create_feature_view(
        name=FV_NAME, version=1,
        description="Orbit-geometry-only features for PHA prediction (no MOID/H/size)",
        query=query, labels=[LABEL],
    )
    return fv, feature_cols


def build_model(num_cols, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",  # numerics pass through; HGB handles NaN natively
    )
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_plots(model, X_te, y_te, prob, pred, feature_cols):
    from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay,
                                 ConfusionMatrixDisplay, confusion_matrix)
    from sklearn.inspection import permutation_importance
    OUT.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_te, prob); plt.title("ROC - asteroid PHA")
    plt.savefig(OUT / "roc_curve.png", bbox_inches="tight", dpi=120); plt.close()
    PrecisionRecallDisplay.from_predictions(y_te, prob); plt.title("Precision-Recall - PHA (6% positive)")
    plt.savefig(OUT / "pr_curve.png", bbox_inches="tight", dpi=120); plt.close()
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred),
                           display_labels=["safe", "hazardous"]).plot()
    plt.title("Confusion matrix"); plt.savefig(OUT / "confusion_matrix.png", bbox_inches="tight", dpi=120); plt.close()
    imp = permutation_importance(model, X_te, y_te, n_repeats=8, random_state=42, scoring="average_precision")
    order = np.argsort(imp.importances_mean)[::-1]
    names = [feature_cols[i] for i in order]; vals = imp.importances_mean[order]
    plt.figure(figsize=(8, 6)); plt.barh(names[::-1], vals[::-1])
    plt.xlabel("drop in average precision when shuffled"); plt.title("Orbit features driving PHA")
    plt.savefig(OUT / "feature_importance.png", bbox_inches="tight", dpi=120); plt.close()


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fv, feature_cols = get_feature_view(fs)
    num_cols = [c for c in feature_cols if c not in CATEGORICAL]

    X_tr, X_te, y_tr, y_te = fv.train_test_split(test_size=0.2)
    X_tr, X_te = X_tr[feature_cols], X_te[feature_cols]
    y_tr = y_tr[LABEL].astype(int); y_te = y_te[LABEL].astype(int)
    print(f"train={len(X_tr)} test={len(X_te)} pos_rate={y_tr.mean():.3f} feats={len(feature_cols)}", flush=True)

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    Xall = X_tr  # CV on train split only, to keep the test set clean for plots
    auc = cross_val_score(build_model(num_cols, CATEGORICAL), Xall, y_tr, cv=skf, scoring="roc_auc").mean()
    ap = cross_val_score(build_model(num_cols, CATEGORICAL), Xall, y_tr, cv=skf, scoring="average_precision").mean()

    model = build_model(num_cols, CATEGORICAL); model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]; pred = (prob >= 0.5).astype(int)
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_score, recall_score, f1_score)
    metrics = {
        "roc_auc_cv": round(float(auc), 4),
        "average_precision_cv": round(float(ap), 4),
        "roc_auc_holdout": round(float(roc_auc_score(y_te, prob)), 4),
        "average_precision_holdout": round(float(average_precision_score(y_te, prob)), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "pos_rate": round(float(y_tr.mean()), 4),
    }
    print("metrics:", json.dumps(metrics), flush=True)

    make_plots(model, X_te, y_te, prob, pred, feature_cols)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    mr = project.get_model_registry()
    model_schema = ModelSchema(Schema(X_tr), Schema(y_tr.to_frame()))
    m = mr.sklearn.create_model(
        name=MODEL_NAME, metrics=metrics,
        description="PHA classification from orbit geometry alone (MOID/H/size excluded).",
        input_example=X_tr.head(1), model_schema=model_schema, feature_view=fv,
    )
    m.save(str(OUT))
    print(f"registered {MODEL_NAME} v{m.version}", flush=True)


if __name__ == "__main__":
    main()
