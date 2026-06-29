"""Feature pipeline (F stage) - runs as a Hopsworks job.

Loads the NEO catalogue into an offline feature group. The FG keeps every pulled
field (orbital, physical, label) for the record and for EDA; the honest
orbit-geometry-only subset is selected later in the feature view.

Self-contained single file. Reads the dataset locally when run in the terminal,
otherwise downloads it from the project's Resources dataset (for the job pod).
"""
import json
from pathlib import Path

import pandas as pd
import hopsworks
from hsfs.feature import Feature

FG_NAME = "neo_features"
FG_VERSION = 1
LOCAL_DATA = Path(__file__).resolve().parent.parent / "data" / "neos.jsonl"
RESOURCES_DATA = "Resources/asteroid/neos.jsonl"

STRING_COLS = ["full_name", "neo", "pha", "class"]
INT_COLS = ["spkid", "pha_label"]
# Everything else numeric (orbital + physical), parsed to float; None -> NaN.
FLOAT_COLS = ["a", "e", "i", "om", "w", "q", "ad", "per", "n", "ma", "tp",
              "rot_per", "H", "diameter", "albedo", "moid"]

DESCRIPTIONS = {
    "spkid": "JPL SPK-ID (primary key)",
    "full_name": "Object designation / name",
    "neo": "Near-Earth object flag (always Y here)",
    "pha": "Potentially-hazardous flag (Y/N) - source of the label",
    "pha_label": "Label: 1 = potentially hazardous, 0 = not",
    "class": "Orbit class (AMO/APO/ATE/IEO/...)",
    "a": "Semi-major axis (AU)",
    "e": "Eccentricity",
    "i": "Inclination (deg)",
    "om": "Longitude of ascending node (deg)",
    "w": "Argument of perihelion (deg)",
    "q": "Perihelion distance (AU)",
    "ad": "Aphelion distance (AU)",
    "per": "Orbital period (days)",
    "n": "Mean motion (deg/day)",
    "ma": "Mean anomaly (deg)",
    "tp": "Time of perihelion passage (JD)",
    "rot_per": "Rotation period (h)",
    "H": "Absolute magnitude (size proxy; PHA definition - excluded from model)",
    "diameter": "Diameter (km; size proxy - excluded from model)",
    "albedo": "Geometric albedo (size proxy - excluded from model)",
    "moid": "Earth MOID (AU; PHA definition - excluded from model)",
}


def resolve_data(project) -> Path:
    if LOCAL_DATA.exists():
        return LOCAL_DATA
    print(f"local data absent, downloading {RESOURCES_DATA}", flush=True)
    return Path(project.get_dataset_api().download(RESOURCES_DATA, local_path=".", overwrite=True))


def load_df(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["spkid"]).reset_index(drop=True)
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    for c in STRING_COLS:
        df[c] = df[c].astype(str)
    return df


def feature_list(df):
    feats = []
    for col in df.columns:
        if col in INT_COLS:
            ftype = "bigint"
        elif col in STRING_COLS:
            ftype = "string"
        else:
            ftype = "double"
        feats.append(Feature(col, ftype, description=DESCRIPTIONS.get(col, col)))
    return feats


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    df = load_df(resolve_data(project))
    pos = int(df["pha_label"].sum())
    print(f"loaded {len(df)} NEOs | pha={pos} ({100*pos/len(df):.1f}%)", flush=True)

    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="Near-Earth object orbital + physical features from JPL SBDB, "
                    "for predicting potentially-hazardous (PHA) classification.",
        primary_key=["spkid"],
        features=feature_list(df),
        online_enabled=False,
        statistics_config=True,
    )
    fg.insert(df, wait=True)
    print(f"inserted {len(df)} rows into {FG_NAME} v{FG_VERSION}", flush=True)


if __name__ == "__main__":
    main()
