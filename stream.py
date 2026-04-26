import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0d0d14 !important;
}

section[data-testid="stSidebar"] { display: none; }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1rem 4rem !important; max-width: 660px !important; }

/* ── hero ── */
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 0.5px solid rgba(139,92,246,0.4);
    color: #a78bfa;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 600;
    color: #f1f0ff;
    line-height: 1.25;
    margin-bottom: 8px;
}
.hero-title span {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 13px; color: rgba(255,255,255,0.35); margin-bottom: 6px; }

/* ── progress steps ── */
.steps-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 24px 0 8px;
}
.step-item { display: flex; align-items: center; gap: 6px; }
.step-circle {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600;
}
.step-active  .step-circle { background: #7c3aed; color: #fff; }
.step-done    .step-circle { background: rgba(52,211,153,0.2); color: #34d399; border: 0.5px solid #34d399; }
.step-inactive .step-circle { background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.2); }
.step-lbl { font-size: 11px; }
.step-active   .step-lbl { color: rgba(255,255,255,0.8); }
.step-done     .step-lbl { color: #34d399; }
.step-inactive .step-lbl { color: rgba(255,255,255,0.2); }
.step-line { width: 32px; height: 0.5px; background: rgba(255,255,255,0.1); margin: 0 6px; }

/* ── section card ── */
.sec-card {
    background: #13121f;
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 22px 6px;
    margin-bottom: 16px;
}
.sec-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
}
.sec-icon {
    width: 30px; height: 30px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 14px;
}
.sec-icon-blue   { background: rgba(96,165,250,0.15); }
.sec-icon-purple { background: rgba(167,139,250,0.15); }
.sec-icon-green  { background: rgba(52,211,153,0.15); }
.sec-title {
    font-size: 11px; font-weight: 600; color: rgba(255,255,255,0.4);
    text-transform: uppercase; letter-spacing: 0.1em;
}

/* ── override streamlit widgets ── */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label {
    font-size: 12px !important;
    color: rgba(255,255,255,0.4) !important;
    font-weight: 400 !important;
    margin-bottom: 2px !important;
}
div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #a78bfa !important; font-size: 13px !important; font-weight: 500 !important;
}
div[data-baseweb="select"] > div {
    background: #0d0d14 !important;
    border: 0.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.75) !important;
}
div[data-baseweb="select"] svg { color: rgba(255,255,255,0.3) !important; }

[data-testid="stSlider"] div[role="slider"] {
    background: #fff !important;
    box-shadow: 0 0 0 3px #7c3aed !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
}

/* ── nav buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.15s ease !important;
}
button[kind="primary"], .stButton > button[data-testid*="primary"] {
    background: #7c3aed !important;
    border: none !important;
    color: #fff !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── result card ── */
.result-wrap {
    background: #13121f;
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px;
    margin-top: 8px;
    text-align: center;
}
.result-icon { font-size: 2.8rem; margin-bottom: 8px; }
.result-label-sm { font-size: 11px; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.result-big { font-size: 2rem; font-weight: 600; margin-bottom: 16px; }
.result-low    { color: #34d399; }
.result-medium { color: #fbbf24; }
.result-high   { color: #f87171; }

.prob-item { margin-bottom: 10px; }
.prob-meta { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px; }
.prob-name { color: rgba(255,255,255,0.35); }
.prob-pct  { color: rgba(255,255,255,0.5); font-weight: 500; }
.prob-track { height: 5px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
.prob-fill-low    { height: 100%; border-radius: 99px; background: #34d399; }
.prob-fill-medium { height: 100%; border-radius: 99px; background: #fbbf24; }
.prob-fill-high   { height: 100%; border-radius: 99px; background: #f87171; }

.tip-box {
    margin-top: 18px;
    background: rgba(167,139,250,0.07);
    border-left: 2px solid rgba(167,139,250,0.4);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 12px;
    color: rgba(255,255,255,0.45);
    line-height: 1.6;
    text-align: left;
}

.footer-txt {
    text-align: center;
    font-size: 11px;
    color: rgba(255,255,255,0.15);
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)


# ── load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model   = joblib.load("notebook\modell.pkl")
        columns = joblib.load("notebook\columns.pkl")
        scaler  = joblib.load("notebook\scaler.pkl")
        return model, columns, scaler, True
    except FileNotFoundError as e:
        return None, None, None, str(e)

model, feature_columns, scaler, status = load_artifacts()

# ── session state ───────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1
if "data" not in st.session_state:
    st.session_state.data = {}

step = st.session_state.step


# ── hero ────────────────────────────────────────────────────────────────────
st.markdown('<div style="text-align:center;margin-bottom:4px"><div class="hero-badge">AI-Powered Assessment</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="text-align:center">Mental Health <span>Risk Predictor</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub" style="text-align:center">Fill in your details below to get a personalized risk assessment</div>', unsafe_allow_html=True)


# ── progress steps ──────────────────────────────────────────────────────────
def step_class(n):
    if n < step: return "step-done"
    if n == step: return "step-active"
    return "step-inactive"

def step_icon(n):
    if n < step: return "✓"
    return str(n)

labels = ["Personal", "Mental", "Lifestyle", "Results"]
steps_html = '<div class="steps-wrap">'
for i, lbl in enumerate(labels, 1):
    cls = step_class(i)
    steps_html += f'<div class="step-item {cls}"><div class="step-circle">{step_icon(i)}</div><span class="step-lbl">{lbl}</span></div>'
    if i < 4:
        steps_html += '<div class="step-line"></div>'
steps_html += '</div>'
st.markdown(steps_html, unsafe_allow_html=True)


# ── model missing warning ───────────────────────────────────────────────────
if status is not True and step == 4:
    st.error(f"Model files not found: {status}\n\nMake sure `modell.pkl`, `columns.pkl`, `scaler.pkl` are next to app.py")
    st.stop()


# ╔══════════════════════════════════╗
# ║  STEP 1 — Personal               ║
# ╚══════════════════════════════════╝
if step == 1:
    st.markdown('<div class="sec-card"><div class="sec-header"><div class="sec-icon sec-icon-blue">👤</div><span class="sec-title">Personal Information</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age    = st.slider("Age", 18, 65, st.session_state.data.get("age", 30))
        gender = st.selectbox("Gender",
            ["Female", "Male", "Non-binary", "Prefer not to say"],
            index=["Female","Male","Non-binary","Prefer not to say"].index(
                st.session_state.data.get("gender","Female")))
    with c2:
        emp_opts = ["Employed", "Self-employed", "Student", "Unemployed"]
        employment_status = st.selectbox("Employment Status", emp_opts,
            index=emp_opts.index(st.session_state.data.get("employment_status","Employed")))

        env_opts = ["Hybrid", "On-site", "Remote"]
        work_environment = st.selectbox("Work Environment", env_opts,
            index=env_opts.index(st.session_state.data.get("work_environment","Hybrid")))

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Next →", use_container_width=True, type="primary"):
        st.session_state.data.update({
            "age": age, "gender": gender,
            "employment_status": employment_status,
            "work_environment": work_environment
        })
        st.session_state.step = 2
        st.rerun()


# ╔══════════════════════════════════╗
# ║  STEP 2 — Mental Health          ║
# ╚══════════════════════════════════╝
elif step == 2:
    st.markdown('<div class="sec-card"><div class="sec-header"><div class="sec-icon sec-icon-purple">🧠</div><span class="sec-title">Mental & Emotional Health</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        stress_level     = st.slider("Stress Level (1–10)",     1,  10, st.session_state.data.get("stress_level", 5))
        depression_score = st.slider("Depression Score (0–30)", 0,  30, st.session_state.data.get("depression_score", 10))
        anxiety_score    = st.slider("Anxiety Score (0–21)",    0,  21, st.session_state.data.get("anxiety_score", 7))
    with c2:
        mh_opts = ["No", "Yes"]
        mental_health_history = st.selectbox("Mental Health History", mh_opts,
            index=mh_opts.index(st.session_state.data.get("mental_health_history","No")))
        seeks_treatment = st.selectbox("Currently Seeking Treatment?", mh_opts,
            index=mh_opts.index(st.session_state.data.get("seeks_treatment","No")))
        social_support_score = st.slider("Social Support Score (0–100)", 0, 100,
            st.session_state.data.get("social_support_score", 50))

    st.markdown('</div>', unsafe_allow_html=True)

    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Next →", use_container_width=True, type="primary"):
            st.session_state.data.update({
                "stress_level": stress_level,
                "depression_score": depression_score,
                "anxiety_score": anxiety_score,
                "mental_health_history": mental_health_history,
                "seeks_treatment": seeks_treatment,
                "social_support_score": social_support_score
            })
            st.session_state.step = 3
            st.rerun()


# ╔══════════════════════════════════╗
# ║  STEP 3 — Lifestyle              ║
# ╚══════════════════════════════════╝
elif step == 3:
    st.markdown('<div class="sec-card"><div class="sec-header"><div class="sec-icon sec-icon-green">🌙</div><span class="sec-title">Lifestyle</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sleep_hours = st.slider("Sleep Hours / Night (3–10)", 3.0, 10.0,
            float(st.session_state.data.get("sleep_hours", 7.0)), step=0.5)
        physical_activity_days = st.slider("Physical Activity Days/Week (0–7)", 0, 7,
            st.session_state.data.get("physical_activity_days", 3))
    with c2:
        productivity_score = st.slider("Productivity Score (42.8–100)", 42.8, 100.0,
            float(st.session_state.data.get("productivity_score", 77.0)), step=0.1)

    st.markdown('</div>', unsafe_allow_html=True)

    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_next:
        if st.button("Predict My Risk ✦", use_container_width=True, type="primary"):
            st.session_state.data.update({
                "sleep_hours": sleep_hours,
                "physical_activity_days": physical_activity_days,
                "productivity_score": productivity_score
            })
            st.session_state.step = 4
            st.rerun()


# ╔══════════════════════════════════╗
# ║  STEP 4 — Results                ║
# ╚══════════════════════════════════╝
elif step == 4:
    d = st.session_state.data

    input_dict = {
        "age":                    d["age"],
        "mental_health_history":  1 if d["mental_health_history"] == "Yes" else 0,
        "seeks_treatment":        1 if d["seeks_treatment"] == "Yes" else 0,
        "stress_level":           d["stress_level"],
        "sleep_hours":            d["sleep_hours"],
        "physical_activity_days": d["physical_activity_days"],
        "depression_score":       d["depression_score"],
        "anxiety_score":          d["anxiety_score"],
        "social_support_score":   d["social_support_score"],
        "productivity_score":     d["productivity_score"],
        "gender_Male":                  1 if d["gender"] == "Male" else 0,
        "gender_Non-binary":            1 if d["gender"] == "Non-binary" else 0,
        "gender_Prefer not to say":     1 if d["gender"] == "Prefer not to say" else 0,
        "employment_status_Self-employed": 1 if d["employment_status"] == "Self-employed" else 0,
        "employment_status_Student":       1 if d["employment_status"] == "Student" else 0,
        "employment_status_Unemployed":    1 if d["employment_status"] == "Unemployed" else 0,
        "work_environment_On-site": 1 if d["work_environment"] == "On-site" else 0,
        "work_environment_Remote":  1 if d["work_environment"] == "Remote" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[list(feature_columns)]

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probas       = model.predict_proba(input_scaled)[0]

    icons  = {0: "✅", 1: "⚠️", 2: "🚨"}
    labels_map = {0: ("Low", "result-low"), 1: ("Medium", "result-medium"), 2: ("High", "result-high")}
    tips   = {
        0: "Great news! Keep maintaining your healthy habits — sleep, exercise, and social connections matter.",
        1: "Consider speaking with a mental health professional. Small lifestyle changes can make a big difference.",
        2: "Please reach out to a mental health professional or counselor as soon as possible. You are not alone.",
    }

    lbl, css = labels_map[prediction]
    p_low    = round(probas[0] * 100, 1)
    p_med    = round(probas[1] * 100, 1)
    p_high   = round(probas[2] * 100, 1)
    tip_text = tips[prediction]
    icon     = icons[prediction]

    # Build HTML as a plain string — no f-string interpolation inside markdown block
    result_html = (
        '<div class="result-wrap">'
        + '<div class="result-icon">' + icon + '</div>'
        + '<div class="result-label-sm">Your Mental Health Risk</div>'
        + '<div class="result-big ' + css + '">' + lbl + ' Risk</div>'

        + '<div class="prob-item">'
        + '<div class="prob-meta"><span class="prob-name">Low</span>'
        + '<span class="prob-pct">' + str(p_low) + '%</span></div>'
        + '<div class="prob-track"><div class="prob-fill-low" style="width:' + str(p_low) + '%"></div></div>'
        + '</div>'

        + '<div class="prob-item">'
        + '<div class="prob-meta"><span class="prob-name">Medium</span>'
        + '<span class="prob-pct">' + str(p_med) + '%</span></div>'
        + '<div class="prob-track"><div class="prob-fill-medium" style="width:' + str(p_med) + '%"></div></div>'
        + '</div>'

        + '<div class="prob-item">'
        + '<div class="prob-meta"><span class="prob-name">High</span>'
        + '<span class="prob-pct">' + str(p_high) + '%</span></div>'
        + '<div class="prob-track"><div class="prob-fill-high" style="width:' + str(p_high) + '%"></div></div>'
        + '</div>'

        + '<div class="tip-box">&#128161; ' + tip_text + '</div>'
        + '</div>'
    )
    st.markdown(result_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Start Over", use_container_width=True):
        st.session_state.step = 1
        st.session_state.data = {}
        st.rerun()


# ── footer ──────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-txt">For educational purposes only — not a substitute for professional medical advice.</div>', unsafe_allow_html=True)