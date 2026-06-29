"""Feature pipeline (F stage): Gaia DR3 asteroid reflectance spectra -> FG.

Pulls the mean reflectance spectra of asteroids from Gaia DR3 (VizieR I/359/ssor),
pivots the long table to one 16-band spectrum per numbered asteroid, and writes
the offline feature group `asteroid_reflectance`. These spectra are the model's
features (composition signal); the label (albedo) lives in a separate FG and the
join happens in the feature view, never here.

I/O-bound (one HTTP pull + one FG insert), so it runs fine from the terminal pod.
Self-contained single file.
"""
import time

import pandas as pd
import requests
import hopsworks
from hsfs.feature import Feature

FG_NAME = "asteroid_reflectance"
FG_VERSION = 1
TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
# Gaia DR3 internally-calibrated reflectance wavelengths (nm), 16 bands.
BANDS = [374, 418, 462, 506, 550, 594, 638, 682, 726, 770, 814, 858, 902,
         946, 990, 1034]
BAND_COLS = [f"r{b}" for b in BANDS]


def _tap(query, maxrec=1_500_000, timeout=180):
    for attempt in range(3):
        try:
            r = requests.get(TAP, params={"REQUEST": "doQuery", "LANG": "ADQL",
                             "FORMAT": "json", "MAXREC": str(maxrec),
                             "QUERY": query}, timeout=timeout)
            d = r.json()
            return pd.DataFrame(d["data"], columns=[c["name"] for c in d["metadata"]])
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


def pull_reflectance() -> pd.DataFrame:
    """One 16-band spectrum per numbered asteroid (reflectance normalised to 1 at
    550 nm by Gaia). Keep only complete 16-band spectra."""
    parts = []
    for lo in range(1, 620001, 40000):
        parts.append(_tap(f'SELECT "MPC","lambda","ReflSp" FROM "I/359/ssor" '
                          f'WHERE "MPC">={lo} AND "MPC"<={lo + 39999}'))
    g = pd.concat([p for p in parts if len(p)], ignore_index=True)
    piv = g.pivot_table(index="MPC", columns="lambda", values="ReflSp")
    piv = piv.reindex(columns=BANDS).dropna()
    piv.columns = BAND_COLS
    out = piv.reset_index().rename(columns={"MPC": "number"})
    out["number"] = out["number"].astype("int64")
    return out


def feature_list():
    feats = [Feature("number", "bigint", description="Minor-planet number (primary key)")]
    feats += [Feature(c, "double", description=f"Gaia DR3 reflectance at {b} nm "
                      "(normalised to 1.0 at 550 nm)") for c, b in zip(BAND_COLS, BANDS)]
    return feats


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    df = pull_reflectance()
    print(f"pulled {len(df)} complete 16-band spectra", flush=True)

    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=FG_VERSION,
        description="Gaia DR3 mean reflectance spectra of numbered asteroids "
                    "(16 bands, 374-1034 nm). Composition features.",
        primary_key=["number"], features=feature_list(),
        online_enabled=False, statistics_config=True)
    fg.insert(df, wait=True)
    print(f"inserted {len(df)} rows into {FG_NAME} v{FG_VERSION}", flush=True)


if __name__ == "__main__":
    main()
