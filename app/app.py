"""Asteroid Doomsday-o-meter — Streamlit app.

Type a near-Earth asteroid name (Apophis, Bennu, Eros, ...). The app pulls its
live orbit from the JPL Small-Body Database and scores how "potentially
hazardous" the model thinks it is, from orbit geometry alone (no MOID, no size).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

FEATURE_COLS = ["class", "a", "e", "i", "om", "w", "q", "ad", "per", "n", "ma", "rot_per"]
MODEL_NAME = "asteroid_pha"
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api"
HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Asteroid Doomsday-o-meter", page_icon="🪨", layout="centered")


@st.cache_resource
def load_model():
    try:
        import hopsworks
        mr = hopsworks.login().get_model_registry()
        d = mr.get_model(MODEL_NAME, version=1).download()
        return joblib.load(Path(d) / "model.joblib")
    except Exception as e:
        local = HERE.parent / "artifact" / "model.joblib"
        if local.exists():
            return joblib.load(local)
        raise RuntimeError(f"could not load model: {e}")


@st.cache_data(ttl=3600)
def fetch_asteroid(name: str) -> dict:
    """Look up an object in JPL SBDB and return the model's feature columns."""
    r = requests.get(SBDB, params={"sstr": name, "phys-par": "true"}, timeout=20)
    r.raise_for_status()
    d = r.json()
    if "object" not in d:
        raise ValueError(d.get("message", "not found"))
    elems = {e["name"]: e.get("value") for e in d.get("orbit", {}).get("elements", [])}
    phys = {p["name"]: p.get("value") for p in d.get("phys_par", [])}
    fullname = d["object"].get("fullname", name)
    klass = d["object"].get("orbit_class", {}).get("code", "UNK")

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    a, e = f(elems.get("a")), f(elems.get("e"))
    row = {
        "class": klass,
        "a": a, "e": e, "i": f(elems.get("i")), "om": f(elems.get("om")),
        "w": f(elems.get("w")), "q": f(elems.get("q")), "ad": f(elems.get("ad")),
        "per": f(elems.get("per")), "n": f(elems.get("n")), "ma": f(elems.get("ma")),
        "rot_per": f(phys.get("rot_per")),
    }
    if row["ad"] is None and a is not None and e is not None:
        row["ad"] = a * (1 + e)
    return {"fullname": fullname, "is_neo": d["object"].get("neo", False), "row": row}


model = load_model()

st.title("🪨 Asteroid Doomsday-o-meter")
st.caption("Type a near-Earth asteroid. We pull its orbit from JPL and score how "
           "'potentially hazardous' it reads, from orbit geometry alone. The "
           "model never sees the actual close-approach distance or the size, "
           "which is what *defines* hazardous. So this is a guess from shape.")

name = st.text_input("Asteroid name or designation", value="Apophis",
                     placeholder="Apophis, Bennu, Eros, 2024 YR4, ...")

if st.button("Score it", type="primary") and name.strip():
    try:
        info = fetch_asteroid(name)
    except Exception as e:
        st.error(f"lookup failed: {e}")
        st.stop()
    if not info["is_neo"]:
        st.warning(f"{info['fullname']} is not a near-Earth object; the model is "
                   "trained on NEOs only. Scoring anyway.")
    X = pd.DataFrame([info["row"]])[FEATURE_COLS]
    score = float(model.predict_proba(X)[0, 1]) * 100

    st.subheader(info["fullname"])
    st.metric("Doomsday score (model)", f"{score:.0f} / 100")
    st.progress(min(int(score), 100))
    if score >= 70:
        st.error("Orbit reads hazardous. Note: a guess from shape, not a real alert.")
    elif score >= 35:
        st.warning("Borderline orbit.")
    else:
        st.success("Orbit reads benign.")

    st.subheader("Orbit the model saw")
    r = info["row"]
    c = st.columns(3)
    c[0].metric("semi-major axis a (AU)", f"{r['a']:.3f}" if r["a"] else "n/a")
    c[1].metric("eccentricity e", f"{r['e']:.3f}" if r["e"] else "n/a")
    c[2].metric("inclination i (deg)", f"{r['i']:.2f}" if r["i"] else "n/a")
    c[0].metric("perihelion q (AU)", f"{r['q']:.3f}" if r["q"] else "n/a")
    c[1].metric("aphelion ad (AU)", f"{r['ad']:.3f}" if r["ad"] else "n/a")
    c[2].metric("orbit class", r["class"])
    with st.expander("All features"):
        st.json(r)

st.divider()
st.caption("Label = JPL `pha` flag. Model predicts from orbital geometry only "
           "(MOID, H, diameter, albedo excluded — they define the flag). "
           "ROC-AUC 0.86. Built on Hopsworks. Not a planetary-defense tool.")
