import streamlit as st
import numpy as np
import pickle
import time

st.set_page_config(
    page_title="CardioSense — Heart Disease Risk Assessment",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ---------------------------------------------------------------------------
# Custom CSS — "Vital Signs" design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F5F9F7;
    --surface: #FFFFFF;
    --border: #E3EDE9;
    --coral: #E8556B;
    --coral-deep: #C43E52;
    --teal: #0F6E5F;
    --teal-deep: #0A4F44;
    --gold: #E8A23D;
    --ink: #1E2A2E;
    --muted: #5B6B68;
}

.stApp {
    background: var(--bg);
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
    color: var(--ink);
    letter-spacing: -0.01em;
}

/* Hero */
.hero-wrap {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    animation: fadeIn 0.8s ease;
}
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--coral);
    font-weight: 600;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0.3rem 0 0.6rem 0;
    line-height: 1.05;
}
.hero-title em {
    font-style: italic;
    color: var(--teal);
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ECG line animation */
.ecg-wrap {
    width: 100%;
    max-width: 700px;
    margin: 1.5rem auto 0.5rem auto;
}
.ecg-path {
    stroke: var(--coral);
    stroke-width: 2.5;
    fill: none;
    stroke-dasharray: 1400;
    stroke-dashoffset: 1400;
    animation: draw 2.8s ease-out forwards, pulse-glow 2.4s ease-in-out infinite 2.8s;
}
@keyframes draw {
    to { stroke-dashoffset: 0; }
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.55; }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Card */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    box-shadow: 0 4px 24px rgba(15, 110, 95, 0.06);
    margin-bottom: 1.2rem;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--teal);
    font-weight: 600;
    margin-bottom: 0.6rem;
    display: block;
}

/* Streamlit widget theming */
div[data-baseweb="select"] > div, .stNumberInput input, .stSlider {
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    border-color: var(--border) !important;
    background: #FBFDFC !important;
}
.stSlider [role="slider"] {
    background-color: var(--coral) !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--teal), var(--coral)) !important;
}
label {
    font-weight: 600 !important;
    color: var(--ink) !important;
    font-size: 0.9rem !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--coral), var(--coral-deep));
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2.4rem;
    font-weight: 600;
    font-size: 1.02rem;
    font-family: 'Inter', sans-serif;
    width: 100%;
    box-shadow: 0 4px 18px rgba(232, 85, 107, 0.35);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(232, 85, 107, 0.45);
    background: linear-gradient(135deg, var(--coral-deep), var(--coral));
}

/* Result */
.result-wrap {
    text-align: center;
    animation: fadeIn 0.6s ease;
    padding: 1rem 0;
}
.risk-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 4rem;
    font-weight: 600;
    line-height: 1;
    margin: 0.4rem 0;
}
.risk-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}
.factor-chip {
    display: inline-block;
    background: #F0F7F5;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    margin: 0.25rem;
    font-size: 0.85rem;
    color: var(--teal-deep);
    font-weight: 500;
}
.disclaimer {
    font-size: 0.82rem;
    color: var(--muted);
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <span class="eyebrow">Machine Learning · Cardiac Risk Model</span>
    <div class="hero-title">Know your heart's<br><em>vital signs.</em></div>
    <div class="hero-sub">Enter your clinical measurements below and get an instant, data-driven estimate
    of heart disease risk — powered by a Random Forest model trained on 300+ patient records.</div>
</div>
<div class="ecg-wrap">
<svg viewBox="0 0 700 80" width="100%" height="80">
    <path class="ecg-path" d="M0,40 L150,40 L170,40 L185,10 L200,70 L215,20 L230,40 L260,40
             L440,40 L460,40 L475,10 L490,70 L505,20 L520,40 L550,40 L700,40" />
</svg>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Patient Profile</span>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [
        "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"
    ])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", [
        "Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Exercise & Diagnostic Results</span>', unsafe_allow_html=True)
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
    oldpeak = st.slider("ST Depression (Exercise vs Rest)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
predict_clicked = st.button("Assess My Risk")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    sex_val = 1 if sex == "Male" else 0
    cp_val = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp)
    fbs_val = 1 if fbs == "Yes" else 0
    restecg_val = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
    exang_val = 1 if exang == "Yes" else 0
    slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal_val = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}[thal]

    features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                           thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
    features_scaled = scaler.transform(features)

    with st.spinner("Analyzing vitals..."):
        time.sleep(0.6)
        proba = model.predict_proba(features_scaled)[0][1]

    risk_pct = round(proba * 100, 1)
    if risk_pct < 30:
        risk_color = "#0F6E5F"
        risk_word = "Low Risk"
    elif risk_pct < 60:
        risk_color = "#E8A23D"
        risk_word = "Moderate Risk"
    else:
        risk_color = "#E8556B"
        risk_word = "Elevated Risk"

    # Gauge angle: 0% -> -90deg, 100% -> 90deg
    angle = -90 + (risk_pct / 100) * 180

    st.markdown(f"""
    <div class="card result-wrap">
        <span class="section-label">Estimated Risk</span>
        <svg viewBox="0 0 200 110" width="260" height="145" style="margin: 0.5rem auto;">
            <path d="M20,100 A80,80 0 0,1 180,100" stroke="#E3EDE9" stroke-width="14" fill="none" stroke-linecap="round"/>
            <path d="M20,100 A80,80 0 0,1 180,100" stroke="{risk_color}" stroke-width="14" fill="none" stroke-linecap="round"
                  stroke-dasharray="{(risk_pct/100)*251.2},251.2"/>
            <line x1="100" y1="100" x2="{100 + 65*np.cos(np.radians(angle))}" y2="{100 + 65*np.sin(np.radians(angle))}"
                  stroke="{risk_color}" stroke-width="4" stroke-linecap="round"/>
            <circle cx="100" cy="100" r="7" fill="{risk_color}"/>
        </svg>
        <div class="risk-number" style="color:{risk_color};">{risk_pct}%</div>
        <div class="risk-label">{risk_word}</div>
    </div>
    """, unsafe_allow_html=True)

    # Contributing factors (simple heuristic based on feature importance + values)
    importances = dict(zip(
        ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
        model.feature_importances_
    ))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:4]
    labels_map = {
        'cp': 'Chest pain type', 'ca': 'Major vessels count', 'thal': 'Thalassemia status',
        'thalach': 'Max heart rate', 'oldpeak': 'ST depression', 'age': 'Age',
        'exang': 'Exercise angina', 'chol': 'Cholesterol', 'trestbps': 'Blood pressure',
        'slope': 'ST slope', 'sex': 'Sex', 'fbs': 'Fasting blood sugar', 'restecg': 'Resting ECG'
    }
    chips = "".join([f'<span class="factor-chip">{labels_map[f[0]]}</span>' for f in top_features])
    st.markdown(f"""
    <div class="card" style="text-align:center;">
        <span class="section-label">Most Influential Factors in This Model</span>
        <div style="margin-top:0.5rem;">{chips}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    This tool is a machine learning demo built on a public research dataset (UCI Heart Disease / Cleveland).
    It is <b>not a medical diagnosis</b>. Please consult a qualified healthcare professional for any real health concerns.
</div>
""", unsafe_allow_html=True)
