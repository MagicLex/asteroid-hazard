"""Asteroid Doomsday-o-meter — Streamlit app.

Type a near-Earth asteroid name (Apophis, Bennu, Eros, ...). The app pulls its
live orbit from the JPL Small-Body Database and scores how "potentially
hazardous" the model thinks it is, from orbit geometry alone (no MOID, no size).
"""
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib import pyplot as plt

FEATURE_COLS = ["class", "a", "e", "i", "om", "w", "q", "ad", "per", "n", "ma", "rot_per"]
MODEL_NAME = "asteroid_pha"
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api"
SBDB_QUERY = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
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
    # Real MOID — shown in the viz for honesty, NEVER fed to the model.
    orbit = d.get("orbit", {})
    moid = f(orbit.get("moid"))
    if moid is None:
        moid = f(elems.get("moid"))
    return {"fullname": fullname, "is_neo": d["object"].get("neo", False),
            "row": row, "moid": moid}


def orbit_xyz(a, e, i, om, w, n=720):
    """Sample an orbital ellipse in heliocentric ecliptic coords (AU).

    Keplerian elements -> 3D points via the standard perifocal rotation
    (argument of perihelion w, inclination i, longitude of node om).
    """
    i, om, w = np.radians(i), np.radians(om), np.radians(w)
    nu = np.linspace(0, 2 * np.pi, n)
    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    xp, yp = r * np.cos(nu), r * np.sin(nu)  # perifocal plane
    co, so, ci, si, cw, sw = (np.cos(om), np.sin(om), np.cos(i),
                              np.sin(i), np.cos(w), np.sin(w))
    x = (co * cw - so * sw * ci) * xp + (-co * sw - so * cw * ci) * yp
    y = (so * cw + co * sw * ci) * xp + (-so * sw + co * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.vstack([x, y, z])


def closest_pair(p, q):
    """Min distance between two sampled orbits and the two points realising it."""
    d2 = ((p[:, :, None] - q[:, None, :]) ** 2).sum(axis=0)
    ia, ib = np.unravel_index(d2.argmin(), d2.shape)
    return np.sqrt(d2[ia, ib]), p[:, ia], q[:, ib]


def orbit_figure(row, moid):
    earth = orbit_xyz(1.0, 0.0167, 0.0, 0.0, 102.9)
    ast = orbit_xyz(row["a"], row["e"], row["i"], row["om"], row["w"])
    dmin, pa, pe = closest_pair(ast, earth)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.4), facecolor="#0b0e11")
    for ax, (h, v), title in ((ax1, (0, 1), "Top-down (ecliptic plane)"),
                              (ax2, (0, 2), "Edge-on (shows inclination)")):
        ax.set_facecolor("#0b0e11")
        ax.plot(earth[h], earth[v], color="#3b82f6", lw=1.6, label="Earth")
        ax.plot(ast[h], ast[v], color="#d97706", lw=1.6, label="Asteroid")
        ax.plot([pa[h], pe[h]], [pa[v], pe[v]], color="#ef4444", lw=1.0,
                ls="--", marker="o", ms=3, label="closest approach")
        ax.scatter([0], [0], color="#fbbf24", s=40, zorder=5)  # Sun
        ax.set_title(title, color="#cbd5e1", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(colors="#475569", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#1f2937")
    ax1.legend(loc="upper right", fontsize=7, facecolor="#1a1f2b",
               edgecolor="#1f2937", labelcolor="#cbd5e1")
    fig.tight_layout()
    return fig, dmin


@st.cache_data(ttl=86400)
def neo_designations():
    """All NEO designations from the JPL catalogue (the same source the model
    was built on), so the random picker draws a real object, not a fixed list."""
    r = requests.get(SBDB_QUERY, params={"fields": "pdes,name", "sb-group": "neo",
                                         "full-prec": "false"}, timeout=120)
    r.raise_for_status()
    d = r.json()
    pi, ni = d["fields"].index("pdes"), d["fields"].index("name")
    # prefer the friendly name (Apophis), fall back to the designation (2004 MN4)
    return [row[ni] or row[pi] for row in d["data"] if row[ni] or row[pi]]


def pick_random():
    st.session_state.asteroid = random.choice(neo_designations())
    st.session_state.go = True


model = load_model()

st.title("🪨 Asteroid Doomsday-o-meter")
st.caption("Type a near-Earth asteroid. We pull its orbit from JPL and score how "
           "'potentially hazardous' it reads, from orbit geometry alone. The "
           "model never sees the actual close-approach distance or the size, "
           "which is what *defines* hazardous. So this is a guess from shape.")

if "asteroid" not in st.session_state:
    st.session_state.asteroid = "Apophis"
name = st.text_input("Asteroid name or designation", key="asteroid",
                     placeholder="Apophis, Bennu, Eros, 2024 YR4, ...")
st.caption("Try a named one — Apophis, Bennu, Eros, Ryugu, Didymos, Toutatis, "
           "Phaethon — or any provisional designation like `2024 YR4`. Names come "
           "from the [JPL Small-Body Database](https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html). "
           "Hit 🎲 for a real random NEO drawn from the catalogue.")
c1, c2 = st.columns([1, 1])
run = c1.button("Score it", type="primary", use_container_width=True)
c2.button("🎲 Random NEO", on_click=pick_random, use_container_width=True)

if (run or st.session_state.pop("go", False)) and name.strip():
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

    st.subheader("The orbit, drawn")
    if r["a"] and r["e"] is not None and r["i"] is not None:
        fig, dmin = orbit_figure(r, info["moid"])
        st.pyplot(fig, use_container_width=True)
        cc = st.columns(2)
        cc[0].metric("Real MOID (JPL)", f"{info['moid']:.4f} AU"
                     if info["moid"] is not None else "n/a",
                     help="Minimum orbit intersection distance with Earth. "
                          "PHA is defined as MOID ≤ 0.05 AU. The model never "
                          "sees this — it guesses it from the orbit's shape.")
        cc[1].metric("Closest the two orbits get", f"{dmin:.4f} AU",
                     help="Computed here from the drawn ellipses. Approximates "
                          "MOID; small differences are sampling and Earth's "
                          "real orbit vs the circle drawn.")
        if info["moid"] is not None:
            hazard_geom = info["moid"] <= 0.05
            st.caption(
                f"Orbits come within **{info['moid']:.3f} AU**. PHA needs ≤ 0.05 AU "
                + ("and a big enough body. This one **clears the distance bar** — "
                   "whether it's a PHA then turns on size, which the model can't see."
                   if hazard_geom else
                   "— this orbit **stays clear of Earth**, so it's not a PHA "
                   "regardless of size."))
    else:
        st.info("Not enough orbital elements returned to draw the orbit.")

st.divider()
st.caption("Label = JPL `pha` flag. Model predicts from orbital geometry only "
           "(MOID, H, diameter, albedo excluded — they define the flag). "
           "ROC-AUC 0.86. Built on Hopsworks. Not a planetary-defense tool.")
