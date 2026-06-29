"""Feature pipeline (F stage): NEOWISE albedo + diameter -> FG.

Pulls thermal-model albedos and diameters from the NEOWISE main-belt catalogue
(Masiero et al. 2011, VizieR J/ApJ/741/68), one row per numbered asteroid. This
is the LABEL source: the model learns Gaia reflectance -> albedo, and the join to
the reflectance FG happens in the feature view. Multiple WISE solutions per object
are reduced to the median.

I/O-bound; runs fine from the terminal pod. Self-contained single file.
"""
import pandas as pd
import requests
import hopsworks
from hsfs.feature import Feature

FG_NAME = "asteroid_albedo"
FG_VERSION = 1
TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def _tap(query, maxrec=500000, timeout=180):
    r = requests.get(TAP, params={"REQUEST": "doQuery", "LANG": "ADQL",
                     "FORMAT": "json", "MAXREC": str(maxrec), "QUERY": query},
                     timeout=timeout)
    d = r.json()
    return pd.DataFrame(d["data"], columns=[c["name"] for c in d["metadata"]])


def pull_albedo() -> pd.DataFrame:
    m = _tap('SELECT "MPC","pV","Diam","HMag" FROM "J/ApJ/741/68/table1"')
    # MPC is zero-padded number for numbered asteroids ("00002" -> 2).
    m["number"] = pd.to_numeric(m["MPC"].str.strip(), errors="coerce")
    for c in ("pV", "Diam", "HMag"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["number", "pV"])
    m = m[m["pV"] > 0]
    g = m.groupby("number").agg(albedo=("pV", "median"),
                                diameter=("Diam", "median"),
                                h_mag=("HMag", "median")).reset_index()
    g["number"] = g["number"].astype("int64")
    return g


def feature_list():
    return [
        Feature("number", "bigint", description="Minor-planet number (primary key)"),
        Feature("albedo", "double", description="NEOWISE visible geometric albedo pV (label)"),
        Feature("diameter", "double", description="NEOWISE thermal-model diameter (km)"),
        Feature("h_mag", "double", description="Absolute magnitude H (mag)"),
    ]


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    df = pull_albedo()
    print(f"pulled {len(df)} asteroids with NEOWISE albedo", flush=True)

    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=FG_VERSION,
        description="NEOWISE thermal-model albedo + diameter + H per numbered "
                    "asteroid (Masiero+ 2011). Label source for albedo prediction.",
        primary_key=["number"], features=feature_list(),
        online_enabled=False, statistics_config=True)
    fg.insert(df, wait=True)
    print(f"inserted {len(df)} rows into {FG_NAME} v{FG_VERSION}", flush=True)


if __name__ == "__main__":
    main()
