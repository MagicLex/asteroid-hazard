"""Asteroid Doomsday-o-meter — Gaia spectrum -> size -> impact.

Type an asteroid. We pull its Gaia DR3 reflectance spectrum, predict its albedo
from that spectrum (a model trained on the 3% of asteroids NASA has actually
measured), turn albedo + brightness into a size, and a size into an impact
scenario. The headline: for objects with no measured albedo (97% of them), we do
materially better than the blind constant-albedo guess everyone falls back to.
"""
import importlib.util
import random
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Deps that may lag the cloned-env image rebuild — self-heal on cold start.
for _pkg in ("plotly", "xgboost"):
    if importlib.util.find_spec(_pkg) is None:
        for _cmd in (["-m", "uv", "pip", "install", "--system", _pkg],
                     ["-m", "pip", "install", "--user", _pkg]):
            if subprocess.run([sys.executable, *_cmd]).returncode == 0:
                break
        importlib.invalidate_caches()
import plotly.graph_objects as go  # noqa: E402

BANDS = [374, 418, 462, 506, 550, 594, 638, 682, 726, 770, 814, 858, 902,
         946, 990, 1034]
BAND_COLS = [f"r{b}" for b in BANDS]
MODEL_NAME = "asteroid_albedo"
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api"
TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Asteroid Doomsday-o-meter", page_icon="🪨", layout="wide")


@st.cache_resource
def load_model():
    import hopsworks
    mr = hopsworks.login().get_model_registry()
    m = max(mr.get_models(MODEL_NAME), key=lambda x: x.version)  # latest, not v1
    return joblib.load(Path(m.download()) / "model.joblib"), m.version


@st.cache_data(ttl=86400)
def random_pool():
    """Asteroids that have a Gaia spectrum but NO NEOWISE albedo — the model's
    actual target population. Random picks from here so the 'NASA never measured
    this, here's ours' case actually shows up."""
    import hopsworks
    fs = hopsworks.login().get_feature_store()
    refl = set(fs.get_feature_group("asteroid_reflectance", 1).select(["number"]).read()["number"])
    alb = set(fs.get_feature_group("asteroid_albedo", 1).select(["number"]).read()["number"])
    pool = sorted(refl - alb)
    return pool or sorted(refl)


@st.cache_data(ttl=3600)
def fetch_spectrum(query: str):
    """Resolve a number or name to its Gaia DR3 16-band spectrum."""
    q = query.strip()
    where = f'"MPC"={int(q)}' if q.isdigit() else f'"Name"=\'{q.lower()}\''
    r = requests.get(TAP, params={"REQUEST": "doQuery", "LANG": "ADQL",
                     "FORMAT": "json", "QUERY": f'SELECT "MPC","Name","lambda",'
                     f'"ReflSp" FROM "I/359/ssor" WHERE {where}'}, timeout=25)
    rows = r.json()["data"]
    if not rows:
        return None
    band = {int(lam): val for _, _, lam, val in rows}
    if any(b not in band for b in BANDS):
        return None
    return {"number": int(rows[0][0]), "name": rows[0][1],
            "spectrum": [band[b] for b in BANDS]}


@st.cache_data(ttl=3600)
def fetch_physical(query: str):
    """JPL SBDB: H, measured albedo/diameter (if any), and orbit a/e/i."""
    r = requests.get(SBDB, params={"sstr": query, "phys-par": "true"}, timeout=20)
    d = r.json()
    if "object" not in d:
        return {}
    def f(x):
        try: return float(x)
        except (TypeError, ValueError): return None
    phys = {p["name"]: p.get("value") for p in d.get("phys_par", [])}
    elems = {e["name"]: e.get("value") for e in d.get("orbit", {}).get("elements", [])}
    return {"fullname": d["object"].get("fullname", query).strip(),
            "H": f(phys.get("H")), "albedo": f(phys.get("albedo")),
            "diameter": f(phys.get("diameter")),
            "a": f(elems.get("a")), "e": f(elems.get("e")), "i": f(elems.get("i"))}


def predict_albedo(model, spectrum):
    X = pd.DataFrame([dict(zip(BAND_COLS, spectrum))])[BAND_COLS]
    return float(10 ** model.predict(X)[0])


def diameter_km(H, albedo):
    if H is None or albedo is None or albedo <= 0:
        return None
    return 1329.0 * 10 ** (-H / 5) / np.sqrt(albedo)


def density_from_albedo(albedo):
    """Rough bulk density (g/cm³) from albedo-inferred taxonomy."""
    if albedo < 0.10:
        return 1.3, "C-type (carbonaceous, dark)"
    if albedo < 0.25:
        return 2.7, "S-type (silicate, rocky)"
    return 3.0, "bright (E/V/M-type)"


def impact_velocity(a, e, i):
    """Hypothetical Earth-impact speed (km/s) from the orbit's Tisserand encounter
    velocity, floored at Earth's escape speed."""
    v_earth, v_esc = 29.8, 11.2
    if None in (a, e, i) or a <= 0:
        return 19.0  # typical NEO impact speed
    T = 1 / a + 2 * np.sqrt(max(a * (1 - e ** 2), 0)) * np.cos(np.radians(i))
    v_inf = v_earth * np.sqrt(max(3 - T, 0))
    return float(np.sqrt(v_inf ** 2 + v_esc ** 2))


def impact_scenario(diam_km, albedo, v_kms):
    rho, taxon = density_from_albedo(albedo)
    r = diam_km * 500.0                          # radius in m
    mass = rho * 1000 * (4 / 3) * np.pi * r ** 3  # kg
    v = v_kms * 1000.0
    E_j = 0.5 * mass * v ** 2
    E_mt = E_j / 4.184e15                         # megatons TNT
    # transient -> final crater (Collins pi-scaling, vertical, rock target)
    g, rho_t = 9.81, 2500.0
    L = diam_km * 1000.0
    D_tc = 1.161 * (rho * 1000 / rho_t) ** (1 / 3) * L ** 0.78 * v ** 0.44 * g ** -0.22
    D_final_km = 1.25 * D_tc / 1000.0
    return {"density": rho, "taxon": taxon, "mass": mass, "energy_mt": E_mt,
            "crater_km": D_final_km, "v_kms": v_kms}


REFERENCES = [(0.5, "Chelyabinsk (2013)"), (12, "Tunguska (1908)"),
              (1e3, "largest H-bomb"), (1e5, "global catastrophe"),
              (1e8, "Chicxulub (dinosaurs)")]


def scale_label(mt):
    out = "smaller than any recorded impact"
    for e, name in REFERENCES:
        if mt >= e:
            out = f"≈ {name}"
    return out


def size_figure(d_model_m, d_blind_m):
    """The two sizes superposed, to scale: our spectrum-based diameter (amber) vs
    the blind constant-albedo guess (grey dashed). A 300 m reference ring (≈ the
    Eiffel tower's height) gives a human sense of scale."""
    R = max(d_model_m, d_blind_m, 300) / 2
    fig = go.Figure()
    for d, color, fill, name in (
            (d_blind_m, "#6b7280", "rgba(107,114,128,0.18)", "blind 0.14 guess"),
            (d_model_m, "#f59e0b", "rgba(245,158,11,0.40)", "our estimate")):
        fig.add_shape(type="circle", x0=-d/2, y0=-d/2, x1=d/2, y1=d/2,
                      line=dict(color=color, width=2,
                                dash="dash" if color == "#6b7280" else "solid"),
                      fillcolor=fill, layer="below" if color == "#6b7280" else "above")
    fig.add_shape(type="line", x0=-150, y0=-R * 0.95, x1=150, y1=-R * 0.95,
                  line=dict(color="#cbd5e1", width=2))
    fig.add_annotation(x=0, y=-R * 0.95, text="300 m", yshift=10,
                       showarrow=False, font=dict(color="#cbd5e1", size=11))
    fig.update_xaxes(range=[-R * 1.1, R * 1.1], visible=False,
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-R * 1.1, R * 1.1], visible=False)
    fig.update_layout(height=300, paper_bgcolor="#0b0e11", plot_bgcolor="#0b0e11",
                      margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                      title=dict(text="Size: ours (amber) vs the blind guess (grey)",
                                 font=dict(color="#cbd5e1", size=13)))
    return fig


def _circle(lat0, lon0, r_km, n=64):
    th = np.linspace(0, 2 * np.pi, n)
    dlat = (r_km / 111.0) * np.cos(th)
    dlon = (r_km / (111.0 * np.cos(np.radians(lat0)))) * np.sin(th)
    return lat0 + dlat, lon0 + dlon


def blast_rings(energy_mt, crater_km):
    """Damage radii (km). Overpressure/thermal radii scale as energy^(1/3)."""
    c = max(energy_mt, 1e-6) ** (1 / 3)
    return [("Windows shatter (1 psi)", 17.0 * c, "rgba(59,130,246,0.18)", "#3b82f6"),
            ("3rd-degree burns", 12.0 * c, "rgba(251,191,36,0.20)", "#fbbf24"),
            ("Most buildings collapse (5 psi)", 6.5 * c, "rgba(245,158,11,0.28)", "#f59e0b"),
            ("Total destruction (20 psi)", 2.7 * c, "rgba(239,68,68,0.38)", "#ef4444"),
            ("Crater", crater_km / 2, "rgba(124,45,18,0.85)", "#7c2d12")]


def impact_map(rings, lat, lon, city):
    fig = go.Figure()
    for name, r, fillc, line in rings:
        if r <= 0:
            continue
        la, lo = _circle(lat, lon, r)
        fig.add_trace(go.Scattermapbox(lat=la, lon=lo, fill="toself",
                      fillcolor=fillc, line=dict(color=line, width=1),
                      name=f"{name} — {r:.1f} km", hoverinfo="name"))
    fig.add_trace(go.Scattermapbox(lat=[lat], lon=[lon], mode="markers",
                  marker=dict(size=10, color="#ffffff"), hoverinfo="skip",
                  showlegend=False))
    maxr = max((r for _, r, _, _ in rings), default=10)
    zoom = float(np.clip(8.3 - np.log2(max(maxr, 1)), 1, 11))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                      mapbox=dict(style="open-street-map",
                                  center=dict(lat=lat, lon=lon), zoom=zoom),
                      legend=dict(font=dict(color="#cbd5e1", size=10), x=0, y=1,
                                  bgcolor="rgba(11,14,17,0.6)"),
                      paper_bgcolor="#0b0e11")
    return fig


CITIES = {"Paris": (48.8566, 2.3522), "New York": (40.7128, -74.0060),
          "Tokyo": (35.6762, 139.6503), "London": (51.5074, -0.1278)}


# ---------------------------------------------------------------- UI
st.markdown(
    """<div style="background:linear-gradient(90deg,#0b0e11,#1a1f2b);
       border-left:6px solid #f59e0b;padding:0.7rem 1rem;border-radius:6px;">
       <h1 style="margin:0;color:#fafafa;font-size:1.5rem;">🪨 Asteroid Doomsday-o-meter</h1>
       <span style="color:#9ca3af;">Gaia spectrum → albedo → size → impact.
       We estimate the size NASA never measured.</span></div>""",
    unsafe_allow_html=True)
st.write("")

if "ast" not in st.session_state:
    st.session_state.ast = "4"
c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
c1.text_input("Asteroid number or name", key="ast",
              placeholder="4, 21, Vesta, Eros, Ceres, ...")
assess = c2.button("Assess", type="primary", use_container_width=True)


def _pick(v):
    st.session_state.ast = str(v)
    st.session_state.assessed = True


POPULAR = [("1", "Ceres"), ("4", "Vesta"), ("2", "Pallas"), ("16", "Psyche"),
           ("433", "Eros"), ("21", "Lutetia"), ("243", "Ida"), ("951", "Gaspra")]
pcols = st.columns(len(POPULAR) + 1)
for col, (num, nm) in zip(pcols, POPULAR):
    col.button(nm, on_click=_pick, args=(num,), use_container_width=True)
pcols[-1].button("🎲 Random", use_container_width=True,
                 on_click=lambda: _pick(random.choice(random_pool())))
st.caption("Popular targets, a random draw, or type any of the 34,577 numbered "
           "asteroids with a Gaia DR3 spectrum.")

if assess or st.session_state.get("assessed"):
    st.session_state.assessed = True
    with st.spinner("Connecting to Hopsworks and loading the model…"):
        model, mver = load_model()
    query = st.session_state.ast.strip()
    spec = fetch_spectrum(query)
    if spec is None:
        st.error(f"No Gaia DR3 spectrum for “{query}”. Try a numbered main-belt "
                 "asteroid (e.g. 4, 21, 324).")
        st.stop()
    phys = fetch_physical(str(spec["number"]))
    pred_alb = predict_albedo(model, spec["spectrum"])
    meas_alb = phys.get("albedo")
    H = phys.get("H")

    title = phys.get("fullname") or f"{spec['number']} {spec['name']}"
    st.subheader(f"☄️ {title}")

    # ---- row 1: spectrum + albedo face-off | size render
    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure(go.Scatter(
            x=BANDS, y=spec["spectrum"], mode="lines+markers",
            line=dict(color="#f59e0b", width=3), marker=dict(size=5)))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="#0b0e11", plot_bgcolor="#0b0e11",
                          title=dict(text="Gaia DR3 reflectance spectrum",
                                     font=dict(color="#cbd5e1", size=13)),
                          xaxis=dict(title="wavelength (nm)", color="#475569",
                                     gridcolor="#1f2937"),
                          yaxis=dict(title="reflectance", color="#475569",
                                     gridcolor="#1f2937"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if meas_alb is not None:
            m_err = abs(pred_alb - meas_alb) / meas_alb * 100
            b_err = abs(0.14 - meas_alb) / meas_alb * 100
            st.markdown("**Albedo: NASA vs us vs the blind guess**")
            a, b, c = st.columns(3)
            a.metric("Measured (NASA)", f"{meas_alb:.3f}")
            b.metric("Our model", f"{pred_alb:.3f}", f"{m_err:.0f}% off", delta_color="off")
            c.metric("Blind 0.14", "0.140", f"{b_err:.0f}% off", delta_color="off")
            if m_err < b_err:
                st.caption(f"Not perfect ({m_err:.0f}% off here) — but the blind 0.14 "
                           f"fallback is {b_err:.0f}% off on the *same* rock, so we're "
                           f"**{b_err/max(m_err,1):.1f}× closer**. Across all measured "
                           "objects we halve the median error.")
            else:
                st.caption(f"On this one the blind guess lands closer ({b_err:.0f}% vs "
                           f"{m_err:.0f}%) — it does, sometimes. Across all objects we "
                           "still halve the median error.")
        else:
            st.markdown("**Albedo: NASA vs us**")
            a, b = st.columns(2)
            a.metric("Measured (NASA)", "— none —")
            b.metric("Our model", f"{pred_alb:.3f}")
            st.caption("NASA never measured this one. Everyone else assumes 0.14; "
                       "**we read it off the Gaia spectrum.** 😎")

    D = diameter_km(H, pred_alb)
    D_blind = diameter_km(H, 0.14)
    with right:
        if D is None:
            st.info("No absolute magnitude H from JPL — cannot size this object.")
        else:
            st.plotly_chart(size_figure(D * 1000, D_blind * 1000),
                            use_container_width=True, config={"displayModeBar": False})
            sz = f"{D*1000:,.0f} m" if D < 1 else f"{D:.2f} km"
            szb = f"{D_blind*1000:,.0f} m" if D_blind < 1 else f"{D_blind:.2f} km"
            d1, d2 = st.columns(2)
            d1.metric("Our diameter", sz)
            d2.metric("Blind-guess diameter", szb, f"{(D/D_blind-1)*100:+.0f}%",
                      delta_color="off")

    # ---- row 2: impact on a real map
    if D is not None:
        v = impact_velocity(phys.get("a"), phys.get("e"), phys.get("i"))
        sc = impact_scenario(D, pred_alb, v)
        st.markdown("### ☢️ If it hit Earth")
        info, mapc = st.columns([1, 2])
        with info:
            city = st.selectbox("Drop it on", list(CITIES))
            st.metric("Impact energy", f"{sc['energy_mt']:,.0f} Mt"
                      if sc['energy_mt'] >= 1 else f"{sc['energy_mt']:.3f} Mt")
            st.metric("Crater", f"{sc['crater_km']:.1f} km"
                      if sc['crater_km'] >= 1 else f"{sc['crater_km']*1000:,.0f} m")
            st.caption(f"Type **{sc['taxon']}**, density {sc['density']} g/cm³. "
                       f"Hypothetical vertical hit at **{v:.0f} km/s**. "
                       f"Energy {scale_label(sc['energy_mt'])}.")
        rings = blast_rings(sc["energy_mt"], sc["crater_km"])
        mapc.plotly_chart(impact_map(rings, *CITIES[city], city),
                          use_container_width=True, config={"displayModeBar": False})

    with st.expander("How this works (and what's honest about it)"):
        st.markdown(
            f"""
- **Spectrum → albedo**: model `{MODEL_NAME}` v{mver}, an XGBoost trained on the
  Gaia 16-band spectrum, learned from the ~21k asteroids NASA *did* measure
  (NEOWISE). CV R² 0.60; it halves the size error of the blind constant-albedo
  guess (diameter error ×1.34 → ×1.13).
- **Albedo → size**: the exact formula `D = 1329·10^(−H/5)/√albedo`. The model's
  only job is the albedo; the rest is physics.
- **Size → impact**: mass from an albedo-inferred density, energy = ½mv², crater
  from pi-group scaling. The impact is hypothetical (most of these are main-belt).
- **Honest**: the model never sees albedo, diameter, or H — only the Gaia
  reflectance. Where NASA has a measured albedo we show it next to ours so you can
  judge. Built on Hopsworks; the Gaia↔NEOWISE join lives in the feature view.""")
