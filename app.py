"""Streamlit dashboard: Virtual Patient (Digital Twin). Run: streamlit run app.py"""
import os
import pandas as pd
import streamlit as st
from src.data_gen import generate
from src.model import build_features, train, FEATURES
from src.twin import PatientTwin, risk_band

st.set_page_config(page_title="Digital Twin | Virtual Patient", page_icon="🫀", layout="wide")


@st.cache_resource
def load():
    if os.path.exists("data/ehr.csv"):
        ehr, wear = pd.read_csv("data/ehr.csv"), pd.read_csv("data/wearables.csv")
    else:
        ehr, wear = generate()
    df = build_features(ehr, wear)
    model, metrics = train(df)
    return ehr, wear, df, model, metrics


ehr, wear, df, model, metrics = load()

st.title("🫀 Virtual Patient Digital Twin")
st.caption("Predicts 90-day cardiometabolic event risk from EHR + wearable data. Synthetic data only. Not medical advice.")

with st.sidebar:
    st.header("Select patient")
    pid = st.selectbox("Patient ID", df.sort_values("event_90d", ascending=False).patient_id.tolist()[:300] +
                       df.patient_id.tolist()[:50], index=0)
    st.header("What-if plan")
    steps = st.slider("Extra daily steps", 0, 6000, 0, 500)
    sleep = st.slider("Extra sleep (hrs/night)", 0.0, 2.0, 0.0, 0.25)
    row = df[df.patient_id == pid].iloc[0]
    adh = None
    if row.on_meds:
        adh = st.slider("Medication adherence", 0.2, 1.0, float(row.med_adherence), 0.05)
    quit_s = st.checkbox("Quit smoking", value=False, disabled=not row.smoker)
    kg = st.slider("Weight loss (kg)", 0, 15, 0)
    st.divider()
    st.metric("Model AUROC (held-out)", f"{metrics['auroc']:.2f}")
    st.metric("Brier score", f"{metrics['brier']:.3f}")

twin = PatientTwin(state=row[FEATURES].astype(float), model=model)
plan = dict(steps_delta=steps, sleep_delta=sleep, adherence_to=adh, quit_smoking=quit_s, weight_loss_kg=kg)
res = twin.simulate(**plan)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current 90-day risk", f"{res['baseline']:.0%}", risk_band(res["baseline"]))
c2.metric("Risk with plan", f"{res['scenario']:.0%}", f"{res['change']*100:+.0f} pts", delta_color="inverse")
c3.metric("Age / Sex", f"{int(row.age)} / {row.sex}")
c4.metric("HbA1c | SBP", f"{row.hba1c} | {int(row.sbp)}")

left, right = st.columns(2)
with left:
    st.subheader("Wearable timeline (90 days)")
    w = wear[wear.patient_id == pid].set_index("day")
    metric = st.radio("Signal", ["steps", "sleep_hrs", "resting_hr", "hrv_ms", "spo2"], horizontal=True)
    st.line_chart(w[metric])
with right:
    st.subheader("Top modifiable risk drivers")
    d = twin.drivers()
    if d.empty:
        st.success("No factor stands out; this twin is near healthy reference values.")
    else:
        st.bar_chart(d.set_index("factor")["risk_reduction"])
        st.dataframe(d.round(2), hide_index=True, use_container_width=True)

st.subheader("12-week risk forecast under the plan")
if any([steps, sleep, kg, quit_s]) or (adh is not None and adh > row.med_adherence):
    fc = twin.forecast(weeks=12, **plan).set_index("week")
    st.line_chart(fc)
    st.caption("Simulated with simple documented physiology rules (docs/ASSUMPTIONS.md); small bumps come from the tree model.")
else:
    st.info("Adjust the what-if plan in the sidebar to see the projected trajectory.")

with st.expander("Limitations and ethics"):
    st.markdown("- Trained on **synthetic** data; accuracy does not transfer to real patients.\n"
                "- What-if effects are assumptions, not causal estimates.\n"
                "- Decision support only; a clinician must review any recommendation.\n"
                "- Real deployment needs consent, de-identification and bias auditing.")
  
