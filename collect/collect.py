"""Collect near-Earth objects (NEOs) from the JPL Small-Body Database.

Free, no key, one request returns the whole NEO catalogue with orbital and
physical fields plus the `pha` (potentially hazardous) flag.

Label: pha (Y/N) -> 1/0.

Leakage guard: PHA is DEFINED as MOID <= 0.05 AU and H <= 22.0. So the model
must not see `moid` or `H`, nor the size proxies `diameter`/`albedo` (derived
from H). It predicts PHA from orbit GEOMETRY alone (a, e, i, q, ad, period, ...).
That captures the "does this orbit come close to Earth" half of the definition;
it cannot see the size half. Honest and non-trivial, same discipline as the
vaporware project (exclude the defining features).
"""
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "neos.jsonl"
API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

# Pulled fields. Orbital elements + rotation + class are model features;
# label + leakage/definition fields are pulled for the record but excluded
# from the model (see feature pipeline).
FIELDS = [
    "full_name", "spkid", "neo", "pha", "class",          # id + label + orbit class
    "a", "e", "i", "om", "w", "q", "ad", "per", "n", "ma", "tp",  # orbital geometry
    "rot_per",                                             # rotation period
    "H", "diameter", "albedo", "moid",                     # leakage/definition (excluded)
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    params = {"fields": ",".join(FIELDS), "sb-group": "neo", "full-prec": "false"}
    print("querying JPL SBDB for all NEOs...", flush=True)
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    cols = payload["fields"]
    rows = payload["data"]
    print(f"got {len(rows)} NEOs, {len(cols)} fields", flush=True)

    n_pha = 0
    with open(OUT, "w") as f:
        for row in rows:
            rec = dict(zip(cols, row))
            rec["pha_label"] = 1 if rec.get("pha") == "Y" else 0
            n_pha += rec["pha_label"]
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT} | pha={n_pha} "
          f"({100*n_pha/len(rows):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
