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

# plotly is pinned in app-requirements.txt, but a cloned-env image rebuild can lag
# behind an app restart, so a fresh pod may boot before plotly is baked in.
# Self-heal: install it at startup only if missing (no-op once the image catches
# up). Apps installing a dep at startup is the sanctioned pattern here.
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("plotly") is None:
    for _cmd in (["-m", "uv", "pip", "install", "--system", "plotly"],
                 ["-m", "pip", "install", "--user", "plotly"]):
        if subprocess.run([sys.executable, *_cmd]).returncode == 0:
            break
    importlib.invalidate_caches()
import plotly.graph_objects as go

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
    elem_list = d.get("orbit", {}).get("elements", [])
    elems = {e["name"]: e.get("value") for e in elem_list}
    sigmas_raw = {e["name"]: e.get("sigma") for e in elem_list}
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
    # Real JPL 1-sigma orbit uncertainties per element (the basis for the fan).
    sigmas = {k: (f(sigmas_raw.get(k)) or 0.0) for k in ["a", "e", "i", "om", "w"]}
    return {"fullname": fullname, "is_neo": d["object"].get("neo", False),
            "row": row, "moid": moid, "sigmas": sigmas}


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


def _sphere(center, radius, color, n=18):
    """A small shaded sphere trace (Sun / Earth) for the 3D scene."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n))
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return go.Surface(x=x, y=y, z=z, showscale=False, opacity=1.0,
                      colorscale=[[0, color], [1, color]], hoverinfo="skip",
                      lighting=dict(ambient=0.6, diffuse=0.8))


@st.cache_data(show_spinner=False)
def _orbit_geometry(elem, sig, exp, k=14, nh=720, nl=360):
    """Cached geometry for the 3D view, keyed on (elements, sigmas, exp). Moving
    the slider recomputes only on a real change, and revisited values are instant.
    Returns the nominal orbit (high-res, accurate closest approach) plus the single
    worst-case variation orbit — the sampled orbit that passes closest to Earth —
    and the close-approach spread."""
    a, e, i, om, w = elem
    factor = 10.0 ** exp
    earth_hi = orbit_xyz(1.0, 0.0167, 0.0, 0.0, 102.9, nh)
    nominal = orbit_xyz(a, e, i, om, w, nh)
    dmin, pa, pe = closest_pair(nominal, earth_hi)
    earth_lo = orbit_xyz(1.0, 0.0167, 0.0, 0.0, 102.9, nl)
    rng = np.random.default_rng(0)                         # seeded → stable
    sa, se, si, som, sw = sig
    samples, dists = [], []
    for _ in range(k):
        o = orbit_xyz(max(a + rng.standard_normal() * sa * factor, 0.05),
                      min(max(e + rng.standard_normal() * se * factor, 0.0), 0.99),
                      i + rng.standard_normal() * si * factor,
                      om + rng.standard_normal() * som * factor,
                      w + rng.standard_normal() * sw * factor, nl)
        samples.append(o)
        dists.append(closest_pair(o, earth_lo)[0])
    dists = np.array(dists)
    worst = samples[int(dists.argmin())]                   # passes closest to Earth
    return (earth_hi, nominal, worst, pa, pe, float(dmin),
            float(dists.min()), float(dists.max()))


def orbit_figure(row, sigmas, exp):
    """3D heliocentric view from cached geometry: Sun, Earth's orbit, the nominal
    asteroid orbit, the worst-case variation orbit, and the closest approach.
    Returns the figure plus the close-approach numbers for the readout."""
    elem = (row["a"], row["e"], row["i"], row["om"], row["w"])
    sig = tuple(sigmas.get(c, 0.0) for c in ["a", "e", "i", "om", "w"])
    earth, nominal, worst, pa, pe, dmin, lo, hi = _orbit_geometry(elem, sig, exp)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=earth[0], y=earth[1], z=earth[2], mode="lines", name="Earth orbit",
        line=dict(color="#3b82f6", width=5), hoverinfo="name"))
    fig.add_trace(go.Scatter3d(
        x=worst[0], y=worst[1], z=worst[2], mode="lines", hoverinfo="name",
        name=f"worst case within ×10^{exp}",
        line=dict(color="#f472b6", width=3, dash="dot")))
    fig.add_trace(go.Scatter3d(
        x=nominal[0], y=nominal[1], z=nominal[2], mode="lines",
        name="Asteroid orbit (nominal)", line=dict(color="#f59e0b", width=5),
        hoverinfo="name"))
    fig.add_trace(go.Scatter3d(
        x=[pa[0], pe[0]], y=[pa[1], pe[1]], z=[pa[2], pe[2]],
        mode="lines+markers", name=f"closest approach ({dmin:.3f} AU)",
        line=dict(color="#ef4444", width=4, dash="dash"),
        marker=dict(size=3, color="#ef4444"), hoverinfo="name"))
    fig.add_trace(_sphere([0, 0, 0], 0.06, "#fbbf24"))
    fig.add_trace(_sphere(pe, 0.03, "#3b82f6"))

    ax = dict(showbackground=False, showgrid=True, gridcolor="#1f2937",
              zeroline=False, showticklabels=False, title="")
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#0b0e11", showlegend=True, uirevision="keep",
        legend=dict(font=dict(color="#cbd5e1", size=12), x=0, y=0.98,
                    bgcolor="rgba(26,31,43,0.6)"),
        scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data",
                   bgcolor="#0b0e11",
                   camera=dict(eye=dict(x=1.3, y=1.3, z=0.9),
                               center=dict(x=0, y=0, z=0))))  # pivot on the Sun
    return fig, dmin, lo, hi


@st.cache_data(ttl=86400)
def neo_designations():
    """NEO designations from the JPL catalogue for the random picker. Capped so the
    response stays small and fast (the full 42k-row pull times out), and biased to
    NAMED objects (Apophis, Bennu, ...) which make for a friendlier random pick."""
    r = requests.get(SBDB_QUERY, params={"fields": "pdes,name", "sb-group": "neo",
                                         "full-prec": "false", "limit": "8000"},
                     timeout=30)
    r.raise_for_status()
    d = r.json()
    pi, ni = d["fields"].index("pdes"), d["fields"].index("name")
    named = [row[ni] for row in d["data"] if row[ni]]
    desigs = [row[pi] for row in d["data"] if row[pi]]
    return named or desigs


def pick_random():
    try:
        st.session_state.asteroid = random.choice(neo_designations())
        st.session_state.go = True
    except Exception:
        st.toast("JPL catalogue is slow right now — try again in a moment.")


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

# Score only on a click / random pick, and stash the result. Everything below
# renders from session_state so adjusting the slider does NOT re-fetch or
# re-score — it just re-reads the (cached) orbit geometry.
if (run or st.session_state.pop("go", False)) and name.strip():
    try:
        st.session_state.scored = fetch_asteroid(name)
    except Exception as e:
        st.error(f"lookup failed: {e}")
        st.session_state.pop("scored", None)

if st.session_state.get("scored"):
    info = st.session_state.scored
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

    st.subheader("The orbit, in 3D")
    st.caption("Drag to rotate (the view pivots on the Sun), scroll to zoom. "
               "Earth's orbit in blue, the asteroid's nominal orbit in bright "
               "orange. The pink dotted line is the worst-case variation within "
               "the (exaggerated) uncertainty — the one that passes closest to "
               "Earth; the red dashed line is the nominal closest approach.")
    if r["a"] and r["e"] is not None and r["i"] is not None:
        # Auto-pick an exaggeration so the cone is visible: real 1-sigma is tiny
        # (these orbits are very well determined), so scale it up and SAY so.
        sig_a = info["sigmas"].get("a", 0.0)
        default_exp = 6
        if sig_a > 0:
            default_exp = int(min(9, max(3, round(np.log10(0.01 / sig_a)))))
        exp = st.slider(
            "Exaggerate orbit uncertainty (×10ⁿ)", 3, 9, default_exp,
            key=f"exagg_{info['fullname']}",
            help="The real JPL 1σ orbit uncertainty is microscopic for catalogued "
                 "objects, so the cone is exaggerated to be visible. This is a "
                 "geometric what-if, NOT a JPL impact probability.")

        fig, dmin, lo, hi = orbit_figure(r, info["sigmas"], exp)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

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

        # Collision forecast: how the close-approach distance spreads across the
        # exaggerated variation orbits. Honest framing — not an impact probability.
        st.markdown("**Collision forecast** — nominal vs worst-case variation")
        LD = 0.00257  # 1 lunar distance in AU
        fc = st.columns(2)
        fc[0].metric("Nominal closest approach", f"{dmin / LD:.1f} lunar dist",
                     help=f"{dmin:.4f} AU. 1 lunar distance = {LD} AU.")
        fc[1].metric(f"Worst case at ×10^{exp}", f"{lo / LD:.1f} – {hi / LD:.1f} LD",
                     help=f"{lo:.4f} – {hi:.4f} AU across orbits sampled within "
                          f"the (exaggerated) 1σ uncertainty.")
        st.caption(
            f"Even with the real orbit uncertainty blown up **×10^{exp}**, the "
            f"closest any variation brings this object is **{lo / LD:.1f} lunar "
            f"distances** ({lo:.4f} AU) — Earth is {0.0000426 / LD:.4f} LD wide for "
            "scale. This is a geometric sensitivity sketch from JPL's per-element "
            "1σ, not a real impact probability (that needs the full covariance and "
            "time integration). The closer the worst case sits to the nominal as you "
            "lower the exaggeration, the better-pinned the orbit.")

        if info["moid"] is not None:
            hazard_geom = info["moid"] <= 0.05
            st.caption(
                f"Nominal orbits come within **{info['moid']:.3f} AU**. PHA needs "
                "≤ 0.05 AU "
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
