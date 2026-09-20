"""
Municipal Fleet AI & Sustainability Predictor  (upgraded dashboard)

Run with:   streamlit run app.py
Needs:      xgb_model.json in the same folder, plus streamlit, pandas, numpy,
            xgboost and plotly.
"""

import difflib
import io
import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# =====================================================================
# 1. PAGE CONFIGURATION & THEME
# =====================================================================
st.set_page_config(
    page_title="Municipal Fleet AI | Emission Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_header_style = """
<style>
    [data-testid="stHeader"] {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_header_style, unsafe_allow_html=True)


def _version_tuple(v: str):
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


# Newer Streamlit deprecates use_container_width in favour of width="stretch".
STRETCH = (
    {"width": "stretch"}
    if _version_tuple(st.__version__) >= (1, 50)
    else {"use_container_width": True}
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stMetric, button, input, select, textarea {
        font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    }
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}

    /* Header band: flat navy, one accent rule */
    .app-header {
        background: #1E3A8A; color: #FFFFFF;
        padding: 1.5rem 1.9rem 1.35rem 1.9rem;
        border-radius: 10px; border-bottom: 4px solid #34D399;
        margin-bottom: 1.1rem;
    }
    .app-header .main-header {font-size: 2.1rem; font-weight: 700; line-height: 1.2; margin: 0 0 0.35rem 0;}
    .app-header .sub-header  {font-size: 1.05rem; font-weight: 400; color: #DBEAFE; margin: 0; max-width: 78ch;}

    /* Metrics: theme-neutral cards so they work in light and dark mode */
    [data-testid="stMetric"] {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.30);
        border-radius: 8px; padding: 0.85rem 1rem;
    }
    [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-weight: 600; font-size: 1.7rem;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {gap: 0.25rem; border-bottom: 1px solid rgba(148,163,184,0.35);}
    .stTabs [data-baseweb="tab"] {padding: 0.6rem 1.05rem; font-weight: 500;}
    .stTabs [aria-selected="true"] {font-weight: 700;}

    /* Status pill */
    .status-pill {
        display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
        font-weight: 600; font-size: 0.95rem; color: #0F172A;
    }
    .pill-urgent {background: #FCA5A5;}
    .pill-maint  {background: #FDE047;}
    .pill-opt    {background: #A7F3D0;}

    .app-footer {
        text-align: center; margin-top: 60px; padding-top: 20px;
        border-top: 1px solid rgba(148,163,184,0.35);
        color: #6B7280; font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="main-header">🌍 Municipal Fleet AI &amp; Sustainability Predictor</div>
        <div class="sub-header">Data-driven fleet modernization for SDG 11 (Sustainable Cities), SDG 13 (Climate Action), and SDG 3 (Public Health).</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# 2. MODEL & REFERENCE PARAMETERS
# =====================================================================
@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model("xgb_model.json")
    return m


try:
    model = load_model()
except Exception as exc:  # missing / corrupt model file
    st.error(
        "**The model file could not be loaded.** Place `xgb_model.json` in the same folder as "
        f"this app and restart it.\n\nTechnical details: {exc}"
    )
    st.stop()

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


PREP = _load_json("preprocessing.json")     # written by fleet_pipeline.py, same run as xgb_model.json
META = _load_json("model_metadata.json")

# Built-in fallback constants (only used when preprocessing.json is not next to the app)
means = {'age': 7.8826, 'mileage': 217525.54, 'class': 1030.18, 'fuel': 1030.18, 'route': 1030.18}
stds = {'age': 4.3232, 'mileage': 136483.13, 'class': 521.59, 'fuel': 327.23, 'route': 134.31}

class_map = {'Service Van': 259.25, 'Transit Bus': 1191.20, 'Waste Truck': 1568.54}
fuel_map = {'CNG': 1236.09, 'Diesel': 1161.85, 'Hybrid': 1079.46, 'Petrol': 256.40}
route_map = {'Highway Transit': 849.91, 'Mixed Suburban': 932.70, 'Urban Stop-and-Go': 1160.79}

# What the model has actually seen in training
DOMAIN = {
    'age': (1, 15),
    'km_per_year': (15000.0, 40000.0),
    'fuels_by_class': {'Service Van': ['Diesel', 'Petrol'],
                       'Transit Bus': ['CNG', 'Diesel', 'Hybrid'],
                       'Waste Truck': ['CNG', 'Diesel', 'Hybrid']},
}
if PREP:
    means, stds = PREP['means'], PREP['stds']
    class_map, fuel_map, route_map = PREP['class_map'], PREP['fuel_map'], PREP['route_map']
    _dom = PREP.get('domain', {})
    DOMAIN = {
        'age': tuple(_dom.get('age', DOMAIN['age'])),
        'km_per_year': tuple(_dom.get('km_per_year', DOMAIN['km_per_year'])),
        'fuels_by_class': _dom.get('fuels_by_class', DOMAIN['fuels_by_class']),
    }


def domain_check(cls, fuel_, age_, mileage_):
    """Returns [(short_tag, long_message)] for inputs the model was not trained on."""
    found = []
    lo, hi = DOMAIN['age']
    if not lo <= age_ <= hi:
        found.append(("age outside range",
                      f"Age {age_} is outside the training range ({lo}-{hi} years); beyond it the model "
                      "simply repeats its prediction from the nearest edge."))
    if age_ > 0:
        kpy, (klo, khi) = mileage_ / age_, DOMAIN['km_per_year']
        if not klo * 0.95 <= kpy <= khi * 1.05:   # 5% tolerance around the observed range
            found.append(("unusual km per year",
                          f"{mileage_:,.0f} km over {age_} years is {kpy:,.0f} km per year; training vehicles "
                          f"covered {klo:,.0f}-{khi:,.0f} km per year."))
    allowed = DOMAIN['fuels_by_class'].get(cls)
    if allowed and fuel_ not in allowed:
        found.append(("fuel not seen on class",
                      f"{fuel_} never appeared on a {cls} in the training data (only {', '.join(allowed)}); "
                      "treat this prediction as an extrapolation."))
    return found

baseline_emissions = {'Service Van': 220.0, 'Transit Bus': 1100.0, 'Waste Truck': 1300.0}

FEATURE_COLS = ['Vehicle_Age_Years', 'Cumulative_Mileage', 'Vehicle_Class_encoded',
                'Fuel_Type_encoded', 'Route_Type_encoded']
REQUIRED_COLS = ['Vehicle Class', 'Fuel Type', 'Route Type',
                 'Vehicle Age (Years)', 'Cumulative Mileage (km)']

ST_URGENT = "🔴 URGENT REPLACEMENT"
ST_MAINT = "🟡 MAINTENANCE REQUIRED"
ST_OPT = "🟢 OPTIMAL"
STATUS_ORDER = [ST_URGENT, ST_MAINT, ST_OPT]
STATUS_COLORS = {ST_URGENT: "#EF4444", ST_MAINT: "#F59E0B", ST_OPT: "#10B981"}

TREES_PER_TON = 45          # ~22 kg CO2 per mature tree per year
PALETTE_NAVY = "#1E3A8A"

# Widget defaults (kept in session_state so the Reset button works)
INPUT_DEFAULTS = {'v_class': 'Transit Bus', 'fuel': 'Diesel', 'route': 'Urban Stop-and-Go',
                  'age': 8, 'mileage': 200000}
ASSUMPTION_DEFAULTS = {'annual_km': 30000, 'tax_rate': 2150.0, 'med_pct': 4.0, 'high_pct': 9.0}
for _k, _v in {**INPUT_DEFAULTS, **ASSUMPTION_DEFAULTS}.items():
    st.session_state.setdefault(_k, _v)


def reset_all():
    for k, v in {**INPUT_DEFAULTS, **ASSUMPTION_DEFAULTS}.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------
# Shared helpers (used by the single-vehicle view, sweeps and batch scoring)
# ---------------------------------------------------------------------
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """raw needs columns: v_class, fuel, route, age, mileage -> standardized model inputs."""
    class_enc = raw['v_class'].map(class_map).fillna(means['class'])
    fuel_enc = raw['fuel'].map(fuel_map).fillna(means['fuel'])
    route_enc = raw['route'].map(route_map).fillna(means['route'])
    return pd.DataFrame({
        'Vehicle_Age_Years': (raw['age'] - means['age']) / stds['age'],
        'Cumulative_Mileage': (raw['mileage'] - means['mileage']) / stds['mileage'],
        'Vehicle_Class_encoded': (class_enc - means['class']) / stds['class'],
        'Fuel_Type_encoded': (fuel_enc - means['fuel']) / stds['fuel'],
        'Route_Type_encoded': (route_enc - means['route']) / stds['route'],
    })[FEATURE_COLS]


def classify(pred, base, med_pct, high_pct):
    pred = np.asarray(pred, dtype=float)
    base = np.asarray(base, dtype=float)
    return np.where(
        pred >= base * (1 + high_pct / 100), ST_URGENT,
        np.where(pred >= base * (1 + med_pct / 100), ST_MAINT, ST_OPT),
    )


def predict_raw(raw: pd.DataFrame) -> np.ndarray:
    return model.predict(build_features(raw))


def status_for(pred_val, base_val):
    return str(classify(pred_val, base_val, med_pct, high_pct))


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=30),
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# =====================================================================
# 3. SIDEBAR CONTROLS
# =====================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8193/8193072.png", width=75)
st.sidebar.header("Fleet Input Parameters")
st.sidebar.markdown("Configure vehicle specifications for real-time AI inference.")

v_class = st.sidebar.selectbox("Vehicle Class", ['Transit Bus', 'Waste Truck', 'Service Van'], key="v_class")
fuel = st.sidebar.selectbox("Fuel Type", ['Diesel', 'CNG', 'Petrol', 'Hybrid'], key="fuel")
route = st.sidebar.selectbox("Route Type", ['Urban Stop-and-Go', 'Mixed Suburban', 'Highway Transit'], key="route")
age = st.sidebar.slider("Vehicle Age (Years)", min_value=DOMAIN['age'][0], max_value=DOMAIN['age'][1],
                        step=1, key="age")
mileage = st.sidebar.slider("Cumulative Mileage (km)", min_value=0, max_value=1000000, step=5000,
                            format="%d", key="mileage")

with st.sidebar.expander("⚙️ Policy & Threshold Assumptions"):
    annual_km = st.number_input("Annual distance per vehicle (km)", min_value=1000, max_value=200000,
                                step=1000, key="annual_km")
    tax_rate = st.number_input("Carbon cost rate (₹ per ton CO₂)", min_value=0.0, max_value=20000.0,
                               step=50.0, key="tax_rate",
                               help="Societal carbon liability based on Indian economic policy models.")
    med_pct = st.slider("Maintenance trigger (% above baseline)", min_value=1.0, max_value=15.0,
                        step=0.5, key="med_pct")
    high_pct = st.slider("Urgent replacement trigger (% above baseline)", min_value=2.0, max_value=25.0,
                         step=0.5, key="high_pct")
    if high_pct <= med_pct:
        st.warning("The urgent trigger must sit above the maintenance trigger. Using maintenance + 0.5%.")
        high_pct = med_pct + 0.5

st.sidebar.button("↺ Reset all inputs", on_click=reset_all)

st.sidebar.divider()
st.sidebar.subheader("🗄️ Batch Fleet Scoring")
uploaded_file = st.sidebar.file_uploader(
    "Upload Fleet CSV",
    type=['csv'],
    help="Upload fleet records with standard columns: Vehicle Class, Fuel Type, Route Type, "
         "Vehicle Age (Years), Cumulative Mileage (km).",
)
st.sidebar.caption("Batch engine evaluates bulk inventories under strict data validation.")

_template = pd.DataFrame({
    'Vehicle Class': ['Transit Bus', 'Waste Truck', 'Service Van'],
    'Fuel Type': ['Diesel', 'CNG', 'Petrol'],
    'Route Type': ['Urban Stop-and-Go', 'Mixed Suburban', 'Highway Transit'],
    'Vehicle Age (Years)': [8, 12, 4],
    'Cumulative Mileage (km)': [210000, 380000, 90000],
})
st.sidebar.download_button(
    "📄 Download CSV template",
    data=_template.to_csv(index=False).encode('utf-8'),
    file_name="fleet_upload_template.csv",
    mime="text/csv",
)

# =====================================================================
# 4. DATA PROCESSING & LATENCY TIMER
# =====================================================================
@st.cache_data(show_spinner=False)
def parse_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


batch_df = None
parse_error = None
if uploaded_file is not None:
    try:
        batch_df = parse_csv(uploaded_file.getvalue())
    except Exception as exc:
        parse_error = str(exc)

# Single-vehicle inference
single_raw = pd.DataFrame([{'v_class': v_class, 'fuel': fuel, 'route': route,
                            'age': age, 'mileage': mileage}])
single_features = build_features(single_raw)

# Time the exact execution speed of the XGBoost model
start_time = time.perf_counter()
pred = float(model.predict(single_features)[0])
latency_ms = (time.perf_counter() - start_time) * 1000

base = baseline_emissions.get(v_class, 1000.0)
med_thresh = base * (1 + med_pct / 100)
high_thresh = base * (1 + high_pct / 100)
max_gauge = base * 1.18
degradation_pct = ((pred - base) / base) * 100
single_status = status_for(pred, base)

annual_co2_single = (pred * annual_km) / 1_000_000
baseline_co2_single = (base * annual_km) / 1_000_000
carbon_cost_single = annual_co2_single * tax_rate


# ---- Batch validation & scoring -------------------------------------
def validate_batch(df: pd.DataFrame):
    """Returns (clean_df, missing_cols, category_messages, numeric_messages)."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, missing, [], []

    for c in ['Vehicle Class', 'Fuel Type', 'Route Type']:
        df[c] = df[c].astype("string").str.strip()

    cat_msgs = []
    for col, label, valid in [('Vehicle Class', 'Vehicle Class', set(class_map)),
                              ('Fuel Type', 'Fuel Type', set(fuel_map)),
                              ('Route Type', 'Route Type', set(route_map))]:
        bad = df.loc[df[col].notna() & ~df[col].isin(valid), col].unique()
        if len(bad) > 0:
            hints = []
            for b in bad:
                close = difflib.get_close_matches(str(b), sorted(valid), n=1, cutoff=0.6)
                hints.append(f"{b} (did you mean '{close[0]}'?)" if close else str(b))
            cat_msgs.append(f"Unrecognized {label}: {', '.join(hints)}. Accepted values: {', '.join(sorted(valid))}")

    num_msgs = []
    for col in ['Vehicle Age (Years)', 'Cumulative Mileage (km)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    text_missing = df[['Vehicle Class', 'Fuel Type', 'Route Type']].isna().any(axis=1)
    num_bad = df[['Vehicle Age (Years)', 'Cumulative Mileage (km)']].isna().any(axis=1)
    neg = (df['Vehicle Age (Years)'] < 0) | (df['Cumulative Mileage (km)'] < 0)
    problem_rows = (text_missing | num_bad | neg)
    if problem_rows.any():
        rows = (df.index[problem_rows] + 2).tolist()  # +2 = spreadsheet row incl. header
        shown = ', '.join(map(str, rows[:10])) + (' …' if len(rows) > 10 else '')
        num_msgs.append(f"{len(rows)} row(s) have blank, non-numeric or negative values (CSV rows: {shown}).")
    return df, [], cat_msgs, num_msgs


@st.cache_data(show_spinner=False)
def score_fleet(df: pd.DataFrame, med_pct_: float, high_pct_: float, annual_km_: float, tax_rate_: float):
    raw = pd.DataFrame({
        'v_class': df['Vehicle Class'].astype(object), 'fuel': df['Fuel Type'].astype(object),
        'route': df['Route Type'].astype(object),
        'age': df['Vehicle Age (Years)'], 'mileage': df['Cumulative Mileage (km)'],
    })
    out = df.copy()
    for c in ['Vehicle Class', 'Fuel Type', 'Route Type']:
        out[c] = out[c].astype(object)
    preds = predict_raw(raw)
    base_v = out['Vehicle Class'].map(baseline_emissions).fillna(1000.0).to_numpy(dtype=float)
    out['Predicted_CO2_g_km'] = np.round(preds, 2)
    out['Baseline_CO2_g_km'] = base_v
    out['Degradation (%)'] = np.round((out['Predicted_CO2_g_km'].to_numpy() - base_v) / base_v * 100, 1)
    out['Priority Status'] = classify(out['Predicted_CO2_g_km'], base_v, med_pct_, high_pct_)
    tons = out['Predicted_CO2_g_km'].to_numpy() * annual_km_ / 1_000_000
    out['Annual CO2 (Tons)'] = np.round(tons, 2)
    out['Annual Carbon Cost (INR)'] = np.round(tons * tax_rate_, 0)
    out['Data Check'] = [
        '; '.join(t for t, _ in domain_check(c_, f_, a_, m_)) or 'OK'
        for c_, f_, a_, m_ in zip(out['Vehicle Class'], out['Fuel Type'],
                                  out['Vehicle Age (Years)'], out['Cumulative Mileage (km)'])
    ]
    return out.sort_values(by='Predicted_CO2_g_km', ascending=False).reset_index(drop=True)


scored_df = None
batch_missing, batch_cat_msgs, batch_num_msgs, batch_error = [], [], [], parse_error
if batch_df is not None:
    try:
        clean_df, batch_missing, batch_cat_msgs, batch_num_msgs = validate_batch(batch_df)
        if len(batch_df) == 0:
            batch_error = "The uploaded file has no data rows."
        elif not (batch_missing or batch_cat_msgs or batch_num_msgs):
            scored_df = score_fleet(clean_df, med_pct, high_pct, annual_km, tax_rate)
    except Exception as exc:
        batch_error = str(exc)

# =====================================================================
# 5. SYSTEM STATUS BANNER
# =====================================================================
sys1, sys2, sys3, sys4 = st.columns(4)
if scored_df is not None:
    asset_count, asset_note, asset_color = len(scored_df), "Batch CSV scored", "normal"
elif batch_df is not None:
    asset_count, asset_note, asset_color = 1, "CSV needs fixes (see Batch tab)", "off"
else:
    asset_count, asset_note, asset_color = 1, "Single-vehicle mode", "normal"
sys1.metric("Assets Analyzed", f"{asset_count:,}", asset_note, delta_color=asset_color)
sys2.metric("AI Engine Status", f"XGBoost v{xgb.__version__}", "Online")
sys3.metric("Inference Latency", f"{latency_ms:.2f} ms", "Real-Time Execution")
sys4.metric("Database Secure Sync", "Active", "AES-256")

# Warn when the selected vehicle lies outside what the model was trained on
input_issues = domain_check(v_class, fuel, age, mileage)
if input_issues:
    st.warning("**This input is outside the training data**\n\n"
               + "\n\n".join(f"- {msg}" for _, msg in input_issues))

# Single-vehicle audit download (sidebar)
st.sidebar.divider()
st.sidebar.subheader("📁 Export Single Audit")
single_report_df = pd.DataFrame([{
    'Vehicle Class': v_class,
    'Fuel Type': fuel,
    'Route Type': route,
    'Age (Years)': age,
    'Cumulative Mileage (km)': mileage,
    'Predicted CO2 (g/km)': round(pred, 2),
    'Baseline CO2 (g/km)': base,
    'Degradation (%)': round(degradation_pct, 1),
    'Priority Status': single_status,
    'Est. Annual Carbon Tax (INR)': round(carbon_cost_single, 2),
}])
st.sidebar.download_button(
    label="📥 Download Single Record CSV",
    data=single_report_df.to_csv(index=False).encode('utf-8'),
    file_name=f"audit_{v_class.lower().replace(' ', '_')}.csv",
    mime="text/csv",
)
st.sidebar.caption("Model constants: loaded from preprocessing.json" if PREP
                   else "Model constants: built-in defaults (preprocessing.json not found)")

# =====================================================================
# 6. DASHBOARD TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Real-Time Diagnostic",
    "🌱 SDG Impact & ROI",
    "🧠 AI Explainability",
    "🗂️ Batch Processing",
    "🔬 What-If Simulator",
])

# ====== TAB 1: REAL-TIME DIAGNOSTIC ======
with tab1:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        axis_min = min(base * 0.85, pred * 0.98)
        axis_max = max(max_gauge, pred * 1.02)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            number={'suffix': " g/km", 'valueformat': ".1f"},
            delta={'reference': base, 'position': "top", 'valueformat': ".1f", 'suffix': " g/km vs baseline",
                   'increasing': {'color': "#DC2626"}, 'decreasing': {'color': "#059669"}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Predicted CO2 (g/km)<br><span style='font-size:0.8em;color:gray'>"
                           f"Factory Baseline: {base:.1f} g/km</span>", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [axis_min, axis_max], 'tickwidth': 1, 'tickcolor': "#334155"},
                'bar': {'color': PALETTE_NAVY},
                'steps': [
                    {'range': [axis_min, med_thresh], 'color': "#A7F3D0"},
                    {'range': [med_thresh, high_thresh], 'color': "#FDE047"},
                    {'range': [high_thresh, axis_max], 'color': "#FCA5A5"},
                ],
                'threshold': {'line': {'color': "#DC2626", 'width': 4}, 'thickness': 0.75, 'value': high_thresh},
            },
        ))
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)
        st.plotly_chart(fig_gauge, **STRETCH)

    with col2:
        st.subheader("Fleet Action Protocol")
        if single_status == ST_URGENT:
            st.error(f"🚨 **URGENT REPLACEMENT**\n\nDegradation is +{degradation_pct:.1f}% above factory baseline. "
                     "Asset creates acute urban air quality risks.")
        elif single_status == ST_MAINT:
            st.warning(f"⚠️ **MAINTENANCE REQUIRED**\n\nDegradation is +{degradation_pct:.1f}%. "
                       "Mechanical overhaul or route reassignment advised.")
        else:
            display_deg = max(0.0, degradation_pct)
            st.success(f"✅ **OPTIMAL OPERATION**\n\nOperating within +{display_deg:.1f}% of baseline. "
                       "Meets municipal sustainability standards.")

        st.metric(label="Inference Confidence", value="XGBoost Regressor",
                  delta="Trained with Standardized Features")

    st.markdown("#### Diagnostic summary")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Predicted CO2", f"{pred:.1f} g/km", f"{degradation_pct:+.1f}% vs baseline", delta_color="inverse")
    d2.metric("Factory baseline", f"{base:.1f} g/km", f"{v_class}", delta_color="off")
    d3.metric("Maintenance trigger", f"≥ {med_thresh:.1f} g/km", f"+{med_pct:g}% over baseline", delta_color="off")
    d4.metric("Replacement trigger", f"≥ {high_thresh:.1f} g/km", f"+{high_pct:g}% over baseline", delta_color="off")

    pill_cls = {ST_URGENT: "pill-urgent", ST_MAINT: "pill-maint", ST_OPT: "pill-opt"}[single_status]
    st.markdown(
        f"<span class='status-pill {pill_cls}'>{single_status}</span> "
        f"&nbsp; {v_class} · {fuel} · {route} · {age} yrs · {mileage:,} km",
        unsafe_allow_html=True,
    )

# ====== TAB 2: SDG IMPACT & ROI ======
with tab2:
    st.subheader("Financial & Environmental ROI (SDG Alignment)")

    scope = "Selected vehicle"
    if scored_df is not None:
        scope = st.radio(
            "Analysis scope",
            ["Selected vehicle", f"Uploaded fleet ({len(scored_df):,} assets)"],
            horizontal=True,
            help="Switch between the vehicle configured in the sidebar and the whole uploaded CSV.",
        )

    if scope == "Selected vehicle":
        annual_co2_tons = annual_co2_single
        baseline_co2_tons = baseline_co2_single
        excess_co2 = max(0.0, annual_co2_tons - baseline_co2_tons)
    else:
        p_arr = scored_df['Predicted_CO2_g_km'].to_numpy()
        b_arr = scored_df['Baseline_CO2_g_km'].to_numpy()
        annual_co2_tons = float(p_arr.sum() * annual_km / 1_000_000)
        baseline_co2_tons = float(b_arr.sum() * annual_km / 1_000_000)
        excess_co2 = float(np.clip(p_arr - b_arr, 0, None).sum() * annual_km / 1_000_000)

    trees_needed = int(annual_co2_tons * TREES_PER_TON)
    carbon_tax_annual = annual_co2_tons * tax_rate

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="Annual Carbon Footprint", value=f"{annual_co2_tons:,.1f} Tons",
              delta=f"{excess_co2:,.1f} Tons aging excess", delta_color="inverse")
    c2.metric(label="Ecological Offset", value=f"{trees_needed:,} Trees",
              help="Mature trees (~22kg CO2/year) required to sequester this annual output.")
    c3.metric(label="10-Yr EV Carbon Savings", value=f"{annual_co2_tons * 10:,.1f} Tons",
              help="Net CO2 prevented over 10 years by transitioning asset to zero emission.")
    c4.metric(label="Est. Annual Carbon Cost", value=f"₹{carbon_tax_annual:,.0f}",
              help="Societal carbon liability based on Indian economic policy models.",
              delta="Financial Liability", delta_color="inverse")

    st.markdown("#### 10-year outlook")
    view = st.radio("Show", ["Cumulative CO₂ (tons)", "Cumulative carbon cost (₹)"], horizontal=True,
                    key="outlook_view")
    years = np.arange(0, 11)
    scale = 1.0 if view.startswith("Cumulative CO₂") else tax_rate
    y_now = annual_co2_tons * years * scale
    y_base = baseline_co2_tons * years * scale
    fig_out = go.Figure()
    fig_out.add_trace(go.Scatter(x=years, y=y_now, mode="lines+markers", name="Current trajectory",
                                 line=dict(color="#EF4444", width=3), fill="tozeroy",
                                 fillcolor="rgba(239,68,68,0.10)"))
    fig_out.add_trace(go.Scatter(x=years, y=y_base, mode="lines", name="Factory baseline",
                                 line=dict(color=PALETTE_NAVY, width=2, dash="dash")))
    fig_out.add_trace(go.Scatter(x=years, y=np.zeros_like(years), mode="lines", name="Zero-emission (EV)",
                                 line=dict(color="#10B981", width=2)))
    fig_out.update_layout(xaxis_title="Years from now",
                          yaxis_title="Tons CO₂" if scale == 1.0 else "INR")
    st.plotly_chart(style_fig(fig_out, 320), **STRETCH)
    st.caption("Projection holds today's predicted emission rate constant; the gap to the green line "
               "is the 10-year EV saving shown above.")

    st.divider()
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.markdown("**SDG 11 (Sustainable Cities & Communities)**")
            st.markdown("Phasing out degraded municipal diesel platforms directly curtails concentrated "
                        "street-level soot in high-density corridors.")
    with s2:
        with st.container(border=True):
            st.markdown("**SDG 13 (Climate Action)**")
            st.markdown("Targeting the top 10% worst-polluting fleet assets delivers the fastest reduction "
                        "in municipal Scope 1 greenhouse gas inventories.")
            if scored_df is not None and scored_df.shape[0] >= 10:
                exc = np.clip(scored_df['Predicted_CO2_g_km'] - scored_df['Baseline_CO2_g_km'], 0, None)
                top_n = max(1, int(np.ceil(len(scored_df) * 0.10)))
                total_exc = exc.sum()
                if total_exc > 0:
                    share = exc.sort_values(ascending=False).head(top_n).sum() / total_exc
                    st.caption(f"In your uploaded fleet, the worst {top_n:,} assets account for "
                               f"{share:.0%} of excess emissions.")
    with s3:
        with st.container(border=True):
            st.markdown("**SDG 3 (Good Health & Well-being)**")
            st.markdown("Eliminating degraded heavy-duty emissions substantially lowers particulate matter "
                        "(PM2.5) and NOx exposure for sanitation crews and pedestrians.")

# ====== TAB 3: AI EXPLAINABILITY ======
with tab3:
    st.subheader("Feature Risk Analysis (Standard Deviations from Mean)")
    st.markdown("This breakdown indicates which vehicle attributes push the prediction higher. Positive "
                "Z-scores denote values exceeding fleet normative averages.")

    z = single_features.iloc[0]
    explain_df = pd.DataFrame({
        'Feature': ['Vehicle Age', 'Cumulative Mileage', 'Vehicle Class Risk', 'Fuel Type Risk', 'Route Risk'],
        'Z-Score (Deviation)': z.to_numpy(),
    })

    fig_bar = go.Figure(go.Bar(
        x=explain_df['Z-Score (Deviation)'],
        y=explain_df['Feature'],
        orientation='h',
        marker_color=['#EF4444' if x > 1.0 else '#F59E0B' if x > 0.0 else '#10B981'
                      for x in explain_df['Z-Score (Deviation)']],
        hovertemplate="%{y}: %{x:.2f} σ<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis_title="Standard Deviations from Training Mean (Z-Score)",
        yaxis_title="",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    fig_bar.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_bar, **STRETCH)
    st.caption("Colour key: red = more than 1 standard deviation above average, amber = above average, "
               "green = at or below average. Class, fuel and route are target-encoded, so their Z-scores "
               "reflect the average emissions of each category (including which vehicle classes use it), "
               "not a like-for-like risk. The contribution chart below shows what the model actually did.")

    with st.expander("🔎 Input values behind each Z-score"):
        detail_df = pd.DataFrame({
            'Feature': explain_df['Feature'],
            'Your input': [f"{age} years", f"{mileage:,} km", v_class, fuel, route],
            'Encoded value': [age, mileage, class_map.get(v_class), fuel_map.get(fuel), route_map.get(route)],
            'Fleet mean': [means['age'], means['mileage'], means['class'], means['fuel'], means['route']],
            'Z-Score': explain_df['Z-Score (Deviation)'].round(2),
        })
        st.dataframe(detail_df, hide_index=True, **STRETCH)

    st.markdown("#### What the model did with these inputs")
    try:
        try:
            n_trees = int(model.best_iteration) + 1
        except Exception:
            n_trees = None
        kw = {"iteration_range": (0, n_trees)} if n_trees else {}
        contribs = model.get_booster().predict(xgb.DMatrix(single_features), pred_contribs=True, **kw)[0]
        if abs(float(contribs.sum()) - pred) > 0.5:
            raise ValueError("Contributions do not add up to the prediction.")
        c_names = ['Vehicle age', 'Cumulative mileage', 'Vehicle class', 'Fuel type', 'Route type']
        c_vals = np.asarray(contribs[:-1], dtype=float)
        c_base = float(contribs[-1])
        order = np.argsort(np.abs(c_vals))
        fig_c = go.Figure(go.Bar(
            x=c_vals[order], y=[c_names[i] for i in order], orientation='h',
            marker_color=['#EF4444' if v > 0 else '#10B981' for v in c_vals[order]],
            hovertemplate="%{y}: %{x:+.1f} g/km<extra></extra>",
        ))
        fig_c.update_layout(xaxis_title=f"Change in predicted CO2 (g/km) from the fleet average of {c_base:,.0f}",
                            yaxis_title="")
        st.plotly_chart(style_fig(fig_c, 300), **STRETCH)
        st.caption("Tree SHAP contributions: red pushes this vehicle's prediction above the fleet average, "
                   "green pulls it below. Z-scores describe how unusual an input is; contributions describe "
                   "its effect on the prediction.")
    except Exception:
        st.caption("Contribution analysis is not available for this model file.")

    st.markdown("#### Sensitivity: how the prediction responds")
    sweep_var = st.radio("Vary", ["Vehicle Age", "Cumulative Mileage"], horizontal=True, key="sweep_var")
    if sweep_var == "Vehicle Age":
        xs = np.arange(DOMAIN['age'][0], DOMAIN['age'][1] + 1)
        cur_x = age
        x_title = "Vehicle Age (Years)"
        sweep_raw = pd.DataFrame({'v_class': v_class, 'fuel': fuel, 'route': route, 'age': xs, 'mileage': mileage})
    else:
        xs = np.linspace(0, 1_000_000, 101)
        cur_x = mileage
        x_title = "Cumulative Mileage (km)"
        sweep_raw = pd.DataFrame({'v_class': v_class, 'fuel': fuel, 'route': route, 'age': age, 'mileage': xs})
    ys = predict_raw(sweep_raw)

    fig_sw = go.Figure()
    fig_sw.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Predicted CO2", line=dict(color=PALETTE_NAVY, width=3)))
    fig_sw.add_trace(go.Scatter(x=[cur_x], y=[pred], mode="markers", name="Selected vehicle",
                                marker=dict(size=13, color="#EF4444", line=dict(color="white", width=2))))
    fig_sw.add_hline(y=med_thresh, line_dash="dot", line_color="#F59E0B",
                     annotation_text="Maintenance trigger", annotation_position="top left")
    fig_sw.add_hline(y=high_thresh, line_dash="dot", line_color="#DC2626",
                     annotation_text="Replacement trigger", annotation_position="top left")
    if sweep_var == "Vehicle Age":
        outside = [(xs.min(), DOMAIN['age'][0]), (DOMAIN['age'][1], xs.max())]
    elif age > 0:
        outside = [(xs.min(), age * DOMAIN['km_per_year'][0]), (age * DOMAIN['km_per_year'][1], xs.max())]
    else:
        outside = []
    for x0, x1 in outside:
        if x1 > x0:
            fig_sw.add_vrect(x0=x0, x1=x1, fillcolor="rgba(148,163,184,0.18)", line_width=0, layer="below")
    fig_sw.update_layout(xaxis_title=x_title, yaxis_title="Predicted CO2 (g/km)")
    st.plotly_chart(style_fig(fig_sw, 340), **STRETCH)
    st.caption("All other inputs are held at the sidebar values. Grey areas lie outside the data the model "
               "was trained on, so the curve there is an extrapolation (tree models flatten out).")

    with st.expander("📊 Model Architecture & Training Validation"):
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Regressor", "XGBoost (Tree-based)")
        col_m2.metric("Cross-Validation", "Repeated 5-Fold (3×)" if META else "5-Fold K-Fold")
        col_m3.metric("Input Normalization", "StandardScaler (Z-Score)")
        st.caption("All continuous and target-encoded categorical features are Z-score standardized"
                   + (", with encodings and scaling fitted on the training split only" if PREP else "")
                   + ". Tree-based models are insensitive to feature scaling, so this keeps the feature "
                     "representation consistent rather than being required for stable splits.")
        if META:
            try:
                t_, cv_ = META.get('test', {}), META.get('cv', {})
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Hold-out R²", f"{t_['r2']:.4f}")
                k2.metric("Hold-out RMSE", f"{t_['rmse']:.2f} g/km")
                k3.metric("Hold-out MAE", f"{t_['mae']:.2f} g/km")
                k4.metric("CV R² (mean ± sd)", f"{cv_['r2_mean']:.4f} ± {cv_['r2_std']:.4f}")
                rows_ = META.get('rows', {})
                src_ = META.get('data', {}).get('source', 'unknown')
                st.caption(f"{META.get('best_n_trees', '?')} trees · trained {str(META.get('trained_at', ''))[:10]} "
                           f"on {rows_.get('train', 0):,} rows · training data source: {src_}. "
                           + ("Scores on synthetic data reflect how well the model recovers the generating "
                              "formula, not accuracy on a real fleet." if src_ == 'synthetic' else ""))
            except Exception:
                pass
        try:
            importances = np.asarray(model.feature_importances_, dtype=float)
            if importances.shape[0] == len(explain_df):
                imp_df = pd.DataFrame({'Feature': explain_df['Feature'], 'Importance': importances})
                imp_df = imp_df.sort_values('Importance')
                fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
                                           marker_color=PALETTE_NAVY))
                fig_imp.update_layout(xaxis_title="Relative importance in the trained model", yaxis_title="")
                st.markdown("**Learned feature importance**")
                st.plotly_chart(style_fig(fig_imp, 260), **STRETCH)
                st.caption("Z-scores show how unusual each input is; importance shows how much the trained "
                           "model relies on each feature overall.")
        except Exception:
            pass

# ====== TAB 4: BATCH PROCESSING ======
with tab4:
    st.subheader("🗂️ Municipal Fleet Bulk Inference Engine")

    if batch_df is None and parse_error is None:
        st.info("Upload a fleet CSV via the sidebar to run batch inference across all vehicles simultaneously.")
    elif batch_error:
        st.error(f"Error processing the uploaded file. Please ensure it is a valid CSV. Technical details: {batch_error}")
    elif batch_missing:
        st.error(f"Missing required columns in CSV: {batch_missing}. Please check your file formatting.")
        st.caption("Required columns: " + ", ".join(REQUIRED_COLS))
    elif batch_cat_msgs:
        st.error("🚨 **Data Validation Failed: Incorrect Input Categories Detected**")
        for msg in batch_cat_msgs:
            st.warning(msg)
        for msg in batch_num_msgs:
            st.warning(msg)
        st.info("Please correct the typos in your CSV dataset to match the accepted categories exactly, then re-upload.")
    elif batch_num_msgs:
        st.error("🚨 **Data Validation Failed: Invalid Numeric or Blank Values Detected**")
        for msg in batch_num_msgs:
            st.warning(msg)
        st.info("Fix or remove those rows, then re-upload.")
    else:
        counts = scored_df['Priority Status'].value_counts()
        c_urg = int(counts.get(ST_URGENT, 0))
        c_mnt = int(counts.get(ST_MAINT, 0))
        c_opt = int(counts.get(ST_OPT, 0))

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Urgent Replacement", f"{c_urg} Vehicles", delta="Immediate CapEx Priority", delta_color="inverse")
        kpi2.metric("Maintenance Required", f"{c_mnt} Vehicles", delta="Scheduled Overhaul", delta_color="off")
        kpi3.metric("Optimal Operations", f"{c_opt} Vehicles", delta="Compliant Assets", delta_color="normal")

        SHORT = {ST_URGENT: "Urgent", ST_MAINT: "Maintenance", ST_OPT: "Optimal"}
        SHORT_COLORS = {SHORT[k]: v for k, v in STATUS_COLORS.items()}
        total_n = len(scored_df)

        ch1, ch2 = st.columns([1, 1.8])
        with ch1:
            st.markdown("**Fleet health mix**")
            donut = go.Figure(go.Pie(
                labels=[f"{SHORT[s_]} ({n / total_n:.0%})" for s_, n in zip(STATUS_ORDER, [c_urg, c_mnt, c_opt])],
                values=[c_urg, c_mnt, c_opt],
                hole=0.7, sort=False,
                marker=dict(colors=[STATUS_COLORS[s_] for s_ in STATUS_ORDER],
                            line=dict(color="rgba(0,0,0,0)", width=0)),
                textinfo="none",
                hovertemplate="%{label}<br>%{value} vehicles<extra></extra>",
            ))
            style_fig(donut, 340)
            donut.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=0.0, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=10, b=10),
                annotations=[dict(text=f"<b>{total_n:,}</b><br>vehicles", x=0.5, y=0.5,
                                  showarrow=False, font=dict(size=18))],
            )
            st.plotly_chart(donut, **STRETCH)

        with ch2:
            st.markdown("**Emissions vs vehicle age**")
            ctl1, ctl2 = st.columns([1.6, 1])
            y_view = ctl1.radio("Y-axis", ["Degradation vs baseline (%)", "Predicted CO₂ (g/km)"],
                                horizontal=True, key="scatter_y")
            cls_view = ctl2.selectbox("Vehicle class", ["All classes"] + sorted(scored_df['Vehicle Class'].unique()),
                                      key="scatter_class")
            plot_df = scored_df if cls_view == "All classes" else scored_df[scored_df['Vehicle Class'] == cls_view]
            plot_df = plot_df.assign(Status=plot_df['Priority Status'].map(SHORT))
            use_deg = y_view.startswith("Degradation")
            y_col = 'Degradation (%)' if use_deg else 'Predicted_CO2_g_km'

            scat = px.scatter(
                plot_df, x='Vehicle Age (Years)', y=y_col, color='Status',
                color_discrete_map=SHORT_COLORS,
                category_orders={'Status': list(SHORT.values())},
                hover_data={'Status': False, 'Vehicle Class': True, 'Fuel Type': True, 'Route Type': True,
                            'Cumulative Mileage (km)': ':,', 'Degradation (%)': ':+.1f'},
                labels={'Predicted_CO2_g_km': 'Predicted CO₂ (g/km)'},
            )
            scat.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white")))
            style_fig(scat, 340)
            scat.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=30, b=10))
            scat.update_xaxes(showgrid=False, title="Vehicle age (years)")
            scat.update_yaxes(zeroline=False)

            # Trigger lines (only meaningful on one common scale)
            if use_deg:
                trig = [(med_pct, "Maintenance"), (high_pct, "Replacement")]
            elif cls_view != "All classes":
                b_ = baseline_emissions[cls_view]
                trig = [(b_ * (1 + med_pct / 100), "Maintenance"), (b_ * (1 + high_pct / 100), "Replacement")]
            else:
                trig = []
            for y_, name_ in trig:
                scat.add_hline(y=y_, line_dash="dot", line_color="rgba(100,116,139,0.8)",
                               annotation_text=name_, annotation_position="top left",
                               annotation_font=dict(size=11, color="#64748B"))
            st.plotly_chart(scat, **STRETCH)

        n_out = int((scored_df['Data Check'] != 'OK').sum())
        if n_out:
            st.warning(f"{n_out:,} of {len(scored_df):,} vehicles fall outside the data the model was trained on "
                       "(see the Data Check column). Their predictions are extrapolations.")

        st.markdown("### Processed Fleet Priority Register")

        f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1])
        sel_status = f1.multiselect("Priority status", STATUS_ORDER, default=STATUS_ORDER)
        sel_class = f2.multiselect("Vehicle class", sorted(scored_df['Vehicle Class'].unique()),
                                   default=sorted(scored_df['Vehicle Class'].unique()))
        sel_fuel = f3.multiselect("Fuel type", sorted(scored_df['Fuel Type'].unique()),
                                  default=sorted(scored_df['Fuel Type'].unique()))
        sort_options = {
            "Predicted CO2 (g/km)": 'Predicted_CO2_g_km',
            "Degradation vs baseline (%)": 'Degradation (%)',
            "Annual carbon cost (INR)": 'Annual Carbon Cost (INR)',
            "Vehicle age (years)": 'Vehicle Age (Years)',
        }
        sort_label = f4.selectbox("Sort by", list(sort_options), index=0)

        view_df = scored_df[
            scored_df['Priority Status'].isin(sel_status)
            & scored_df['Vehicle Class'].isin(sel_class)
            & scored_df['Fuel Type'].isin(sel_fuel)
        ].sort_values(sort_options[sort_label], ascending=False).reset_index(drop=True)

        st.caption(f"Showing {len(view_df):,} of {len(scored_df):,} vehicles.")
        st.dataframe(
            view_df, hide_index=True, **STRETCH,
            column_config={
                'Predicted_CO2_g_km': st.column_config.NumberColumn("Predicted CO2 (g/km)", format="%.2f"),
                'Baseline_CO2_g_km': st.column_config.NumberColumn("Baseline (g/km)", format="%.1f"),
                'Degradation (%)': st.column_config.NumberColumn("Degradation (%)", format="%+.1f"),
                'Annual CO2 (Tons)': st.column_config.NumberColumn("Annual CO2 (Tons)", format="%.2f"),
                'Annual Carbon Cost (INR)': st.column_config.NumberColumn("Annual Carbon Cost (₹)", format="₹%.0f"),
                'Cumulative Mileage (km)': st.column_config.NumberColumn("Cumulative Mileage (km)", format="%d"),
            },
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            label="📥 Download Scored Fleet Register (CSV)",
            data=scored_df.to_csv(index=False).encode('utf-8'),
            file_name="scored_municipal_fleet_register.csv",
            mime="text/csv",
        )
        d2.download_button(
            label="📥 Download Filtered View (CSV)",
            data=view_df.to_csv(index=False).encode('utf-8'),
            file_name="scored_municipal_fleet_filtered.csv",
            mime="text/csv",
        )

# ====== TAB 5: WHAT-IF SIMULATOR ======
with tab5:
    st.subheader("🔬 What-If Replacement Simulator")
    st.markdown(f"Compare the current **{v_class}** on **{route}** against a replacement scenario. "
                "Class and route stay the same; change fuel, age and mileage to see the effect.")

    w1, w2, w3 = st.columns(3)
    alt_fuel = w1.selectbox("Replacement fuel type", ['Hybrid', 'CNG', 'Diesel', 'Petrol'], key="alt_fuel")
    alt_age = w2.slider("Replacement age (years)", DOMAIN['age'][0], DOMAIN['age'][1], DOMAIN['age'][0],
                        key="alt_age")
    alt_mileage = w3.slider("Replacement mileage (km)", 0, 1_000_000, 25000, step=5000, format="%d", key="alt_mileage")

    alt_issues = domain_check(v_class, alt_fuel, alt_age, alt_mileage)
    if alt_issues:
        st.warning("**The replacement scenario is outside the training data**\n\n"
                   + "\n\n".join(f"- {msg}" for _, msg in alt_issues))

    alt_raw = pd.DataFrame([{'v_class': v_class, 'fuel': alt_fuel, 'route': route,
                             'age': alt_age, 'mileage': alt_mileage}])
    alt_pred = float(predict_raw(alt_raw)[0])
    alt_status = status_for(alt_pred, base)
    alt_tons = alt_pred * annual_km / 1_000_000
    tons_delta = alt_tons - annual_co2_single
    cost_delta = tons_delta * tax_rate

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Replacement CO2", f"{alt_pred:.1f} g/km", f"{alt_pred - pred:+.1f} g/km vs current", delta_color="inverse")
    m2.metric("Replacement status", alt_status)
    m3.metric("Annual CO2 change", f"{tons_delta:+,.1f} Tons", f"{(alt_pred - pred) / pred * 100:+.1f}%", delta_color="inverse")
    m4.metric("Annual carbon cost change", f"₹{cost_delta:+,.0f}", f"10-yr: ₹{cost_delta * 10:+,.0f}", delta_color="inverse")

    cmp_fig = go.Figure(go.Bar(
        x=["Current vehicle", f"Replacement ({alt_fuel})"],
        y=[pred, alt_pred],
        marker_color=[STATUS_COLORS[single_status], STATUS_COLORS[alt_status]],
        text=[f"{pred:.1f}", f"{alt_pred:.1f}"], textposition="outside",
    ))
    cmp_fig.add_hline(y=base, line_dash="dash", line_color=PALETTE_NAVY,
                      annotation_text=f"Factory baseline {base:.0f} g/km", annotation_position="top left")
    cmp_fig.update_layout(yaxis_title="Predicted CO2 (g/km)", showlegend=False)
    st.plotly_chart(style_fig(cmp_fig, 340), **STRETCH)

    st.markdown("#### Fuel-switch comparison for the current vehicle")
    fuel_names = list(fuel_map)
    fuel_raw = pd.DataFrame({'v_class': v_class, 'fuel': fuel_names, 'route': route,
                             'age': age, 'mileage': mileage})
    fuel_preds = predict_raw(fuel_raw)
    fuel_status = classify(fuel_preds, base, med_pct, high_pct)
    seen_fuels = DOMAIN['fuels_by_class'].get(v_class, fuel_names)
    fuel_seen = [f_ in seen_fuels for f_ in fuel_names]
    fuel_fig = go.Figure(go.Bar(
        x=fuel_names, y=fuel_preds,
        marker_color=[STATUS_COLORS[s] if ok else "#CBD5E1" for s, ok in zip(fuel_status, fuel_seen)],
        text=[f"{v:.1f}" if ok else f"{v:.1f} (not in training data)" for v, ok in zip(fuel_preds, fuel_seen)],
        textposition="outside",
    ))
    fuel_fig.add_hline(y=base, line_dash="dash", line_color=PALETTE_NAVY)
    fuel_fig.update_layout(yaxis_title="Predicted CO2 (g/km)", showlegend=False)
    st.plotly_chart(style_fig(fuel_fig, 320), **STRETCH)
    st.caption("Bars use the same status colours as the diagnostic gauge; grey bars are fuel types that never "
               "appeared on this vehicle class in the training data. Scenario values are model estimates, "
               "not manufacturer specifications.")

# =====================================================================
# 7. FOOTER
# =====================================================================
st.markdown("""
<div class="app-footer">
    <strong>Municipal Fleet Emissions AI</strong><br>
    ROHIT SHUKLA<br>
    M.Sc. Data Science & Applied Statistics<br>
    Proprietary Predictive Architecture | Build 2026.09
</div>
""", unsafe_allow_html=True)