"""Asteroid Doomsday-o-meter — Gaia spectrum -> size -> impact.

Type an asteroid. We pull its Gaia DR3 reflectance spectrum, predict its albedo
from that spectrum (a model trained on the 3% of asteroids NASA has actually
measured), turn albedo + brightness into a size, and a size into an impact
scenario. The headline: for objects with no measured albedo (97% of them), we do
materially better than the blind constant-albedo guess everyone falls back to.
"""
import importlib.util
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
c1, c2 = st.columns([3, 1])
c1.text_input("Asteroid number or name", key="ast",
              placeholder="4, 21, Vesta, Eros, Ceres, ...")
go = c2.button("Assess", type="primary", use_container_width=True)
st.caption("Works for the 34,577 numbered asteroids with a Gaia DR3 spectrum. "
           "Try `4` (Vesta), `21` (Lutetia), `1` (Ceres), `433` (Eros).")

if go or st.session_state.get("assessed"):
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

    left, right = st.columns([1, 1])

    # ---- left: the spectrum + the albedo face-off
    with left:
        fig = go.Figure(go.Scatter(
            x=BANDS, y=spec["spectrum"], mode="lines+markers",
            line=dict(color="#f59e0b", width=3), marker=dict(size=5)))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="#0b0e11", plot_bgcolor="#0b0e11",
                          title=dict(text="Gaia DR3 reflectance spectrum",
                                     font=dict(color="#cbd5e1", size=13)),
                          xaxis=dict(title="wavelength (nm)", color="#475569",
                                     gridcolor="#1f2937"),
                          yaxis=dict(title="reflectance", color="#475569",
                                     gridcolor="#1f2937"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("**Albedo: NASA vs us**")
        a, b = st.columns(2)
        if meas_alb is not None:
            a.metric("Measured (NASA)", f"{meas_alb:.3f}")
            err = abs(pred_alb - meas_alb) / meas_alb * 100
            b.metric("Predicted (our model)", f"{pred_alb:.3f}", f"{err:.0f}% off",
                     delta_color="off")
            st.caption("This one NASA *has* measured — so you can check us. "
                       "For the 97% they haven't, ours is the only number going.")
        else:
            a.metric("Measured (NASA)", "— not measured —")
            b.metric("Predicted (our model)", f"{pred_alb:.3f}")
            st.caption("NASA never measured this asteroid's albedo. The blind "
                       "fallback assumes 0.14 for everything; **we read it off the "
                       "Gaia spectrum instead.** 😎")

    # ---- right: size + impact
    with right:
        alb_for_size = meas_alb if meas_alb is not None else pred_alb
        D = diameter_km(H, alb_for_size)
        D_blind = diameter_km(H, 0.14)
        if D is None:
            st.info("No absolute magnitude H from JPL — cannot size this object.")
        else:
            v = impact_velocity(phys.get("a"), phys.get("e"), phys.get("i"))
            sc = impact_scenario(D, alb_for_size, v)
            s1, s2 = st.columns(2)
            s1.metric("Diameter (our albedo)", f"{D*1000:,.0f} m"
                      if D < 1 else f"{D:.2f} km")
            if meas_alb is None and D_blind:
                s2.metric("If you'd assumed 0.14", f"{D_blind*1000:,.0f} m"
                          if D_blind < 1 else f"{D_blind:.2f} km",
                          f"{(D/D_blind-1)*100:+.0f}%", delta_color="off")
            st.caption(f"Inferred type: **{sc['taxon']}**, density "
                       f"{sc['density']} g/cm³.")

            st.markdown("**☢️ If it hit Earth**")
            i1, i2 = st.columns(2)
            i1.metric("Impact energy", f"{sc['energy_mt']:,.0f} Mt"
                      if sc['energy_mt'] >= 1 else f"{sc['energy_mt']:.2f} Mt")
            i2.metric("Crater", f"{sc['crater_km']:.1f} km"
                      if sc['crater_km'] >= 1 else f"{sc['crater_km']*1000:,.0f} m")
            st.caption(f"Hypothetical vertical impact at **{v:.0f} km/s** "
                       f"(from the orbit). Energy {scale_label(sc['energy_mt'])}.")

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
