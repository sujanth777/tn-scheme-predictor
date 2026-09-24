"""
🏛️ TN Scheme Eligibility Predictor
===================================
A clean, production-grade civic-tech web application for predicting citizen
eligibility across 20 Tamil Nadu welfare schemes using MultiOutput XGBoost
and SHAP explainability.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap

# ==============================================================================
# 1. PAGE SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="TN Scheme Eligibility Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# 2. EXACT CSS STYLING MATCHING IMAGE 2
# ==============================================================================
st.markdown(
    """
    <style>
    /* ── Import Clean Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Force Light Theme Globally ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Hide sidebar completely for a clean web-app layout */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1080px !important;
        margin: 0 auto !important;
    }

    /* ── Header Card ── */
    .navbar-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon {
        width: 44px;
        height: 44px;
        background: #006644;
        color: white;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
    }
    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .brand-badge {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .brand-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        margin: 2px 0 0 0;
        line-height: 1.4;
    }

    /* ── Streamlit Tabs Styled as Nav Buttons (Top Right) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #F1F5F9;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 8px 18px !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    /* ── Citizen Advisory Banner ── */
    .advisory-banner {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 14px 18px;
        color: #92400E;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-bottom: 22px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .advisory-icon {
        font-size: 1.2rem;
        line-height: 1;
        margin-top: 1px;
    }

    /* ── Main Form Card ── */
    .form-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px 28px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
    }

    /* ── Quick Presets Card ── */
    .presets-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 24px;
    }
    .presets-title {
        color: #334155;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Preset Buttons */
    div[data-testid="column"] button {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 6px 12px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="column"] button:hover {
        border-color: #006644 !important;
        color: #006644 !important;
        background: #F0FDF4 !important;
    }

    /* ── Section Dividers ── */
    .form-section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 22px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #F1F5F9;
    }
    .form-section-title:first-child {
        margin-top: 0;
    }

    /* ── Streamlit Form Input Styling ── */
    div[data-baseweb="input"] {
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        color: #334155 !important;
        margin-bottom: 4px !important;
    }
    .currency-hint {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 600;
        margin-top: -8px;
        margin-bottom: 8px;
    }

    /* ── Primary Action Button ── */
    button[kind="primary"] {
        background-color: #006644 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 12px 24px !important;
        margin-top: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 102, 68, 0.2) !important;
        transition: background-color 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: #004d33 !important;
    }

    /* ── Results Cards ── */
    .stat-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .stat-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.04em;
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        margin: 4px 0 2px 0;
    }
    .stat-sub {
        font-size: 0.72rem;
        color: #94A3B8;
    }

    .scheme-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #006644;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .scheme-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 6px;
    }
    .scheme-name {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0F172A;
    }
    .scheme-badge {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        white-space: nowrap;
    }
    .scheme-dept {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 8px;
    }
    .scheme-desc {
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 10px;
    }
    .conf-track {
        height: 8px;
        background: #F1F5F9;
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 4px;
    }
    .conf-fill {
        height: 100%;
        background: #006644;
        border-radius: 4px;
    }

    .factor-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }
    .factor-pos {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #14532D;
    }
    .factor-neg {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .stat-strip { grid-template-columns: repeat(2, 1fr); }
        .navbar-card { flex-direction: column; align-items: flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. METADATA & FIELD DEFINITIONS
# ==============================================================================
FIELD_LABELS = {
    "Age": "Age",
    "Income": "Annual Income",
    "Gender": "Gender",
    "Student": "Student Status",
    "GovtSchoolStudent": "Govt School Student",
    "Farmer": "Farmer",
    "Widow": "Widow",
    "Disability": "Disability",
    "WorkingWoman": "Working Woman",
    "SC": "Scheduled Caste (SC)",
    "ST": "Scheduled Tribe (ST)",
    "LandOwner": "Agricultural Land Owner",
    "Pregnant": "Pregnant / Lactating Mother",
    "TNResident": "Tamil Nadu Resident",
    "NoPermanentHouse": "No Permanent House / Kutcha",
    "BPL": "Below Poverty Line (BPL)",
    "PrimaryEarnerDeceased": "Primary Earner Deceased",
    "GirlChild": "Family with Girl Child",
    "EnrolledTraining": "Enrolled in Skill Training",
}

SCHEME_META = {
    "Pudhumai Penn": {
        "cat": "Higher Education",
        "dept": "Social Welfare & Women Empowerment Dept",
        "desc": "Financial assistance of ₹1,000/month for girl students from government schools pursuing collegiate degrees or diplomas.",
    },
    "Kalaignar Magalir Urimai Thittam": {
        "cat": "Women Empowerment",
        "dept": "Special Programme Implementation Dept",
        "desc": "Basic monthly income grant of ₹1,000 transferred to eligible women heads of households.",
    },
    "Tamil Pudhalvan Scheme": {
        "cat": "Higher Education",
        "dept": "Higher Education Dept",
        "desc": "Monthly assistance of ₹1,000 for male students from government schools pursuing higher education.",
    },
    "CM Breakfast Scheme": {
        "cat": "School Nutrition",
        "dept": "Social Welfare & Nutritious Meal Dept",
        "desc": "Nutritious morning breakfast for primary school children (Class 1 to 5) across Tamil Nadu.",
    },
    "Old Age Pension": {
        "cat": "Social Security",
        "dept": "Revenue and Disaster Management Dept",
        "desc": "Monthly pension of ₹1,200 for destitute elderly citizens aged 60 and above living below poverty threshold.",
    },
    "Widow Pension": {
        "cat": "Social Security",
        "dept": "Social Welfare Dept",
        "desc": "Monthly financial pension provided to destitute widows for livelihood and dignity.",
    },
    "Disability Pension": {
        "cat": "Differently Abled",
        "dept": "Differently Abled Welfare Dept",
        "desc": "Monthly financial support and rehabilitation pension for persons with benchmark disabilities.",
    },
    "CM Comprehensive Health Insurance": {
        "cat": "Healthcare",
        "dept": "Health & Family Welfare Dept",
        "desc": "Cashless hospitalization coverage up to ₹5 Lakhs per family per year in empanelled network hospitals.",
    },
    "PM-KISAN": {
        "cat": "Agriculture",
        "dept": "Agriculture & Farmers Welfare Dept",
        "desc": "Direct income support of ₹6,000 per year in three equal installments to cultivable landholding farmer families.",
    },
    "PM Awas Yojana (PMAY)": {
        "cat": "Housing",
        "dept": "Rural Development & Housing Dept",
        "desc": "Housing subsidy to construct permanent pucca dwellings with basic amenities for homeless/kutcha households.",
    },
    "PM Ujjwala Yojana": {
        "cat": "Clean Energy",
        "dept": "Civil Supplies & Consumer Protection Dept",
        "desc": "Deposit-free LPG cooking gas connection for adult women belonging to Below Poverty Line (BPL) families.",
    },
    "Ayushman Bharat (PM-JAY)": {
        "cat": "Healthcare",
        "dept": "National Health Mission",
        "desc": "Secondary and tertiary healthcare hospital coverage up to ₹5 Lakhs per family per year for vulnerable households.",
    },
    "PM Matru Vandana Yojana": {
        "cat": "Maternity Benefit",
        "dept": "Women & Child Development",
        "desc": "Direct cash incentive of ₹5,000 for pregnant women and lactating mothers for the first living child.",
    },
    "National Family Benefit Scheme": {
        "cat": "Family Welfare",
        "dept": "Social Security & Revenue Dept",
        "desc": "Lump sum compensation of ₹20,000 to bereaved BPL families upon unexpected death of primary breadwinner.",
    },
    "SC Scholarship": {
        "cat": "Scholarship",
        "dept": "Adi Dravidar and Tribal Welfare Dept",
        "desc": "Full tuition fee reimbursement, maintenance grants, and educational allowances for Scheduled Caste students.",
    },
    "ST Scholarship": {
        "cat": "Scholarship",
        "dept": "Tribal Welfare Dept",
        "desc": "Educational scholarship, hostel maintenance, and study grants for students from Scheduled Tribe communities.",
    },
    "Agricultural Input Subsidy": {
        "cat": "Agriculture",
        "dept": "Agriculture & Farmers Welfare Dept",
        "desc": "Subsidized certified high-yield seeds, fertilizers, and agricultural equipment for registered cultivators.",
    },
    "Dr. Muthulakshmi Reddy Maternity Benefit": {
        "cat": "Maternity Benefit",
        "dept": "Public Health & Preventive Medicine Dept",
        "desc": "Financial maternity assistance of ₹18,000 with maternal nutrition kit for underprivileged pregnant mothers.",
    },
    "CM Girl Child Protection Scheme": {
        "cat": "Child Welfare",
        "dept": "Social Welfare Dept",
        "desc": "Long-term fixed deposit bond maturing with high return for underprivileged families having only one or two girl children.",
    },
    "Naan Mudhalvan": {
        "cat": "Youth & Skills",
        "dept": "Tamil Nadu Skill Development Corporation (TNSDC)",
        "desc": "Industry-aligned skill training, coding bootcamps, and career mentoring for collegiate youth in Tamil Nadu.",
    },
}

# ==============================================================================
# 4. DATA & MODEL LOADERS
# ==============================================================================
@st.cache_resource(show_spinner="Loading machine learning model...")
def load_model_artifacts():
    model = joblib.load("scheme_model.pkl")
    cols = joblib.load("model_feature_columns.pkl")
    return model, cols

@st.cache_data(show_spinner="Loading reference data...")
def load_reference_data():
    rules_df = pd.read_excel("scheme_rules_full.xlsx")
    comp_df = pd.read_csv("model_comparison.csv") if os.path.exists("model_comparison.csv") else pd.DataFrame()
    f1_df = pd.read_csv("per_scheme_f1.csv") if os.path.exists("per_scheme_f1.csv") else pd.DataFrame()
    return rules_df, comp_df, f1_df

try:
    model, expected_feature_columns = load_model_artifacts()
    rules_df, comparison_df, f1_df = load_reference_data()
    scheme_display_names = list(rules_df["SchemeName"])
except Exception as e:
    st.error(f"❌ Failed to load model assets: {e}")
    st.stop()

# ==============================================================================
# 5. SHAP EXPLAINABILITY ENGINE
# ==============================================================================
def explain_scheme_prediction(estimator, encoded_row_df, raw_user_dict, feature_columns_list):
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(encoded_row_df).values[0]
        top_indices = np.argsort(np.abs(shap_values))[::-1][:3]
        explanations = []

        for idx in top_indices:
            feat_name = feature_columns_list[idx]
            sv = float(shap_values[idx])

            if feat_name == "Age":
                label, val_str = "Age", str(raw_user_dict["Age"])
            elif feat_name == "Income":
                label, val_str = "Annual Income", f"₹{raw_user_dict['Income']:,}"
            elif "_" in feat_name:
                field, dummy_val = feat_name.rsplit("_", 1)
                label = FIELD_LABELS.get(field, field)
                val_str = raw_user_dict.get(field, dummy_val)
            else:
                label = FIELD_LABELS.get(feat_name, feat_name)
                val_str = str(raw_user_dict.get(feat_name, ""))

            impact = "strongly increased eligibility" if sv > 1.0 else ("increased eligibility" if sv > 0 else ("strongly decreased eligibility" if sv < -1.0 else "decreased eligibility"))
            explanations.append({"text": f"{label} ({val_str}) {impact}", "score": sv, "is_pos": sv > 0})

        return explanations, "SHAP"
    except Exception:
        fi = estimator.feature_importances_
        top_indices = np.argsort(fi)[::-1][:3]
        explanations = []
        for idx in top_indices:
            feat_name = feature_columns_list[idx]
            weight = float(fi[idx])
            if feat_name in ["Age", "Income"]:
                label = feat_name
                val_str = f"₹{raw_user_dict['Income']:,}" if feat_name == "Income" else str(raw_user_dict["Age"])
            elif "_" in feat_name:
                field, dummy_val = feat_name.rsplit("_", 1)
                label = FIELD_LABELS.get(field, field)
                val_str = raw_user_dict.get(field, dummy_val)
            else:
                label = FIELD_LABELS.get(feat_name, feat_name)
                val_str = str(raw_user_dict.get(feat_name, ""))
            explanations.append({"text": f"{label} ({val_str}) — feature importance: {weight:.1%}", "score": weight, "is_pos": True})
        return explanations, "Top factors"

explain_prediction = explain_scheme_prediction

# ==============================================================================
# 6. HEADER & TOP BAR (MATCHING IMAGE 2 EXACTLY)
# ==============================================================================
st.markdown(
    """
    <div class="navbar-card">
        <div class="brand-group">
            <div class="brand-icon">🏛️</div>
            <div>
                <div class="brand-title-wrap">
                    <h1 class="brand-title">TN Scheme Eligibility Predictor</h1>
                    <span class="brand-badge">Govt of Tamil Nadu</span>
                </div>
                <p class="brand-subtitle">
                    Check eligibility for 20 Tamil Nadu welfare schemes using AI/ML decision intelligence
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_check, tab_directory, tab_analytics, tab_about = st.tabs([
    "Eligibility Check",
    "Scheme Directory",
    "Model Performance",
    "About Platform",
])

# ==============================================================================
# TAB 1: ELIGIBILITY CHECK
# ==============================================================================
with tab_check:
    # ── Advisory Banner (Matching Image 2) ──
    st.markdown(
        """
        <div class="advisory-banner">
            <div class="advisory-icon">🛡️</div>
            <div>
                <b>Citizen Screening Advisory:</b> This machine learning tool provides automated screening based on state welfare rules. Always confirm exact eligibility and submit applications through designated Tamil Nadu e-Sevai or departmental portals.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Demo Presets Definition ──
    PRESETS = {
        "farmer": {
            "name": "Small Cultivator / Farmer",
            "data": {
                "Age": 42, "Gender": "Male", "Income": 180000, "Student": "No",
                "GovtSchoolStudent": "No", "Farmer": "Yes", "Widow": "No",
                "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
                "LandOwner": "Yes", "Pregnant": "No", "TNResident": "Yes",
                "NoPermanentHouse": "No", "BPL": "No", "PrimaryEarnerDeceased": "No",
                "GirlChild": "No", "EnrolledTraining": "No",
            },
        },
        "student": {
            "name": "College Girl Student",
            "data": {
                "Age": 20, "Gender": "Female", "Income": 120000, "Student": "Yes",
                "GovtSchoolStudent": "Yes", "Farmer": "No", "Widow": "No",
                "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
                "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
                "NoPermanentHouse": "No", "BPL": "No", "PrimaryEarnerDeceased": "No",
                "GirlChild": "No", "EnrolledTraining": "No",
            },
        },
        "senior": {
            "name": "Elderly Resident (BPL)",
            "data": {
                "Age": 68, "Gender": "Male", "Income": 45000, "Student": "No",
                "GovtSchoolStudent": "No", "Farmer": "No", "Widow": "No",
                "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
                "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
                "NoPermanentHouse": "Yes", "BPL": "Yes", "PrimaryEarnerDeceased": "No",
                "GirlChild": "No", "EnrolledTraining": "No",
            },
        },
        "working": {
            "name": "BPL Working Mother",
            "data": {
                "Age": 38, "Gender": "Female", "Income": 85000, "Student": "No",
                "GovtSchoolStudent": "No", "Farmer": "No", "Widow": "No",
                "Disability": "No", "WorkingWoman": "Yes", "SC": "No", "ST": "No",
                "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
                "NoPermanentHouse": "No", "BPL": "Yes", "PrimaryEarnerDeceased": "No",
                "GirlChild": "Yes", "EnrolledTraining": "No",
            },
        },
    }

    if "profile" not in st.session_state:
        st.session_state.profile = PRESETS["student"]["data"].copy()

    # ── Quick-Fill Demo Profiles (Matching Image 2) ──
    st.markdown(
        """
        <div class="presets-container">
            <div class="presets-title">
                <span>✨</span> QUICK-FILL DEMO PROFILES (FOR INSTANT TESTING)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for i, (k, preset) in enumerate(PRESETS.items()):
        with cols[i]:
            if st.button(f"👤 {preset['name']}", key=f"btn_{k}", use_container_width=True):
                st.session_state.profile = preset["data"].copy()
                st.rerun()

    p = st.session_state.profile

    def get_index(options, key, fallback="No"):
        val = p.get(key, fallback)
        return options.index(val) if val in options else 0

    # ── The Input Form (Matching Image 2) ──
    with st.form("eligibility_form"):
        # Section 1: Personal Information
        st.markdown('<div class="form-section-title"><span>👤</span> 1. Personal Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (Years)", 1, 100, int(p.get("Age", 20)))
        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"], index=get_index(["Female", "Male"], "Gender", "Female"))
        with c3:
            tn_res = st.selectbox("Tamil Nadu Resident", ["Yes", "No"], index=get_index(["Yes", "No"], "TNResident", "Yes"))

        c4, c5 = st.columns(2)
        with c4:
            student = st.selectbox("Currently a Student", ["No", "Yes"], index=get_index(["No", "Yes"], "Student"))
        with c5:
            govt_sch = st.selectbox("Studied in Government School", ["No", "Yes"], index=get_index(["No", "Yes"], "GovtSchoolStudent"))

        # Section 2: Socio-Economic & Occupation
        st.markdown('<div class="form-section-title"><span>💼</span> 2. Socio-Economic & Occupation</div>', unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        with c6:
            income = st.number_input("Annual Household Income (INR)", 0, 1000000, int(p.get("Income", 120000)), step=5000)
            st.markdown(f'<div class="currency-hint">₹{int(income):,}</div>', unsafe_allow_html=True)
        with c7:
            bpl = st.selectbox("Below Poverty Line (BPL)", ["No", "Yes"], index=get_index(["No", "Yes"], "BPL"))
        with c8:
            farmer = st.selectbox("Farmer / Cultivator", ["No", "Yes"], index=get_index(["No", "Yes"], "Farmer"))

        c9, c10, c11 = st.columns(3)
        with c9:
            land = st.selectbox("Agricultural Land Owner", ["No", "Yes"], index=get_index(["No", "Yes"], "LandOwner"))
        with c10:
            working_woman = st.selectbox("Working Woman", ["No", "Yes"], index=get_index(["No", "Yes"], "WorkingWoman"))
        with c11:
            training = st.selectbox("Enrolled in Skill Training", ["No", "Yes"], index=get_index(["No", "Yes"], "EnrolledTraining"))

        # Section 3: Social Category
        st.markdown('<div class="form-section-title"><span>👥</span> 3. Social Category</div>', unsafe_allow_html=True)
        c12, c13 = st.columns(2)
        with c12:
            sc = st.selectbox("Scheduled Caste (SC)", ["No", "Yes"], index=get_index(["No", "Yes"], "SC"))
        with c13:
            st_cat = st.selectbox("Scheduled Tribe (ST)", ["No", "Yes"], index=get_index(["No", "Yes"], "ST"))

        # Section 4: Special Circumstances
        st.markdown('<div class="form-section-title"><span>🛡️</span> 4. Special Circumstances</div>', unsafe_allow_html=True)
        c14, c15, c16 = st.columns(3)
        with c14:
            disability = st.selectbox("Differently Abled (Disability)", ["No", "Yes"], index=get_index(["No", "Yes"], "Disability"))
        with c15:
            widow = st.selectbox("Widow", ["No", "Yes"], index=get_index(["No", "Yes"], "Widow"))
        with c16:
            pregnant = st.selectbox("Pregnant / Lactating", ["No", "Yes"], index=get_index(["No", "Yes"], "Pregnant"))

        c17, c18, c19 = st.columns(3)
        with c17:
            no_house = st.selectbox("No Permanent House (Kutcha)", ["No", "Yes"], index=get_index(["No", "Yes"], "NoPermanentHouse"))
        with c18:
            deceased = st.selectbox("Primary Breadwinner Deceased", ["No", "Yes"], index=get_index(["No", "Yes"], "PrimaryEarnerDeceased"))
        with c19:
            girl_child = st.selectbox("Family with Girl Child", ["No", "Yes"], index=get_index(["No", "Yes"], "GirlChild"))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Check Eligibility →", type="primary", use_container_width=True)

    # ── Inference & Results ──
    if submitted:
        raw_input = {
            "Age": int(age), "Gender": gender, "Income": int(income),
            "Student": student, "GovtSchoolStudent": govt_sch, "Farmer": farmer,
            "Widow": widow, "Disability": disability, "WorkingWoman": working_woman,
            "SC": sc, "ST": st_cat, "LandOwner": land, "Pregnant": pregnant,
            "TNResident": tn_res, "NoPermanentHouse": no_house, "BPL": bpl,
            "PrimaryEarnerDeceased": deceased, "GirlChild": girl_child,
            "EnrolledTraining": training,
        }
        st.session_state.profile = raw_input.copy()

        try:
            raw_df = pd.DataFrame([raw_input])
            encoded = pd.get_dummies(raw_df).reindex(columns=expected_feature_columns, fill_value=0).astype(int)

            preds = model.predict(encoded)[0]
            probas = model.predict_proba(encoded)

            evaluated = []
            for i, sname in enumerate(scheme_display_names):
                is_elig = bool(preds[i] == 1)
                conf = float(probas[i][0, 1])
                meta = SCHEME_META.get(sname, {"cat": "Welfare", "dept": "Govt of TN", "desc": ""})
                evaluated.append({
                    "name": sname, "is_elig": is_elig, "conf": conf,
                    "estimator": model.estimators_[i],
                    "rules": rules_df.iloc[i].to_dict() if i < len(rules_df) else {},
                    "meta": meta
                })

            eligible = sorted([s for s in evaluated if s["is_elig"]], key=lambda x: x["conf"], reverse=True)
            ineligible = sorted([s for s in evaluated if not s["is_elig"]], key=lambda x: x["conf"], reverse=True)

            # Summary Stats
            st.markdown("---")
            top_conf = f"{eligible[0]['conf'] * 100:.1f}%" if eligible else "0.0%"

            st.markdown(
                f"""
                <div class="stat-strip">
                    <div class="stat-card">
                        <div class="stat-label">Schemes Evaluated</div>
                        <div class="stat-val">20</div>
                        <div class="stat-sub">Welfare Programs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label" style="color: #065F46;">Eligible Schemes</div>
                        <div class="stat-val" style="color: #006644;">{len(eligible)}</div>
                        <div class="stat-sub">Criteria Satisfied</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Top Match Score</div>
                        <div class="stat-val">{top_conf}</div>
                        <div class="stat-sub">Highest Match</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Not Eligible</div>
                        <div class="stat-val" style="color: #64748B;">{len(ineligible)}</div>
                        <div class="stat-sub">Criteria Unmet</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Eligible Schemes List
            st.markdown("### ✅ Eligible Schemes")
            if eligible:
                for s in eligible:
                    pct = s["conf"] * 100
                    m = s["meta"]
                    r = s["rules"]
                    inc_limit = r.get("IncomeLimit", "No Limit")
                    inc_str = f"₹{inc_limit:,}" if isinstance(inc_limit, (int, float)) else str(inc_limit)

                    st.markdown(
                        f"""
                        <div class="scheme-card">
                            <div class="scheme-card-header">
                                <div>
                                    <span class="scheme-name">{s['name']}</span>
                                    <span class="scheme-badge" style="margin-left: 8px;">{m['cat']}</span>
                                </div>
                                <span class="scheme-badge">Eligible • {pct:.1f}% Match</span>
                            </div>
                            <div class="scheme-dept">🏢 {m['dept']}</div>
                            <div class="scheme-desc">{m['desc']}</div>
                            <div class="conf-track">
                                <div class="conf-fill" style="width: {pct}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"Why did the model recommend {s['name']}?", expanded=False):
                        factors, method = explain_scheme_prediction(s["estimator"], encoded, raw_input, expected_feature_columns)
                        st.caption(f"Top contributing decision factors ({method}):")
                        for f in factors:
                            cls = "factor-pos" if f["is_pos"] else "factor-neg"
                            arrow = "▲" if f["is_pos"] else "▼"
                            sign = "+" if f["score"] > 0 else ""
                            st.markdown(
                                f"""
                                <div class="factor-pill {cls}">
                                    <span>{arrow} <b>{f['text']}</b></span>
                                    <span style="font-family: monospace; font-weight: 700;">{sign}{f['score']:.2f}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
            else:
                st.info("No schemes matched this profile based on eligibility thresholds.")

            # Ineligible Schemes Collapsed
            with st.expander(f"Schemes Not Currently Eligible ({len(ineligible)})", expanded=False):
                for s in ineligible:
                    pct = s["conf"] * 100
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; padding: 8px 12px; background: white; border: 1px solid #E2E8F0; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem;">
                            <span><b>{s['name']}</b> ({s['meta']['cat']})</span>
                            <span style="color: #64748B; font-weight: 600;">{pct:.1f}% Match</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as err:
            st.error(f"❌ Evaluation Error: {err}")

# ==============================================================================
# TAB 2: SCHEME DIRECTORY
# ==============================================================================
with tab_directory:
    st.markdown("### Tamil Nadu Welfare Schemes Directory")
    st.caption("Browse all 20 welfare schemes and their official eligibility criteria")

    s1, s2 = st.columns([2, 1])
    with s1:
        kw = st.text_input("🔍 Search schemes", placeholder="Filter by scheme name, department...")
    with s2:
        cats = ["All"] + sorted(list(set(m["cat"] for m in SCHEME_META.values())))
        sel_cat = st.selectbox("Filter Category", cats)

    filtered = []
    for idx, row in rules_df.iterrows():
        sname = row["SchemeName"]
        meta = SCHEME_META.get(sname, {"cat": "Welfare", "dept": "Govt of TN", "desc": ""})
        if (sel_cat == "All" or meta["cat"] == sel_cat) and (not kw or kw.lower() in sname.lower() or kw.lower() in meta["desc"].lower()):
            filtered.append((row, meta))

    st.caption(f"Showing **{len(filtered)}** of 20 schemes")

    dcols = st.columns(2)
    for i, (row, meta) in enumerate(filtered):
        inc_lim = row.get("IncomeLimit", "No Limit")
        inc_str = f"₹{inc_lim:,}" if isinstance(inc_lim, (int, float)) else str(inc_lim)
        with dcols[i % 2]:
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <span style="font-weight: 800; color: #0F172A; font-size: 1rem;">{row['SchemeName']}</span>
                        <span class="scheme-badge">{meta['cat']}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 6px;">🏢 {meta['dept']}</div>
                    <div style="font-size: 0.84rem; color: #334155; margin-bottom: 10px; line-height: 1.4;">{meta['desc']}</div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 10px; font-size: 0.78rem; display: flex; justify-content: space-between;">
                        <span>Target: <b>{row.get('Gender', 'Any')}</b></span>
                        <span>Age: <b>{row.get('MinAge', 0)}-{row.get('MaxAge', 100)} yrs</b></span>
                        <span>Income: <b>{inc_str}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# TAB 3: MODEL PERFORMANCE
# ==============================================================================
with tab_analytics:
    st.markdown("### Machine Learning Benchmarks & Validation")
    st.caption("Performance metrics of the MultiOutput XGBoost classification model")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Macro F1", "99.7%", help="Fair metric across all 20 schemes")
    m2.metric("Micro F1", "99.8%", help="Overall prediction accuracy")
    m3.metric("Exact Match", "99.3%", help="All 20 schemes correctly predicted per citizen")
    m4.metric("Model", "XGBoost (20 Trees)")

    if not comparison_df.empty:
        st.markdown("#### 1. Multi-Model Benchmark Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if not f1_df.empty:
        st.markdown("#### 2. Per-Scheme F1 Breakdown")
        st.dataframe(f1_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 4: ABOUT PLATFORM
# ==============================================================================
with tab_about:
    st.markdown("### About the TN Scheme Eligibility Predictor")
    st.markdown(
        """
        The **TN Scheme Eligibility Predictor** is an AI/ML-powered public service advisory prototype
        evaluating citizen demographic and socio-economic profiles against **20 official Tamil Nadu state welfare schemes**.

        ##### System Pipeline
        1. **Citizen Input:** 19 demographic, financial, and occupational features.
        2. **Preprocessing:** Converted into a 36-dimensional one-hot encoded matrix.
        3. **Inference:** MultiOutputClassifier (XGBoost) evaluates 20 schemes simultaneously.
        4. **Explainability:** SHAP TreeExplainer identifies positive and negative decision drivers.

        ##### Official Portals
        - **TNeGA e-Sevai:** [https://www.tnesevai.tn.gov.in](https://www.tnesevai.tn.gov.in)
        - **Kalaignar Magalir Urimai Thittam:** [https://kmut.tn.gov.in](https://kmut.tn.gov.in)
        - **Pudhumai Penn Scheme:** [https://pudhumaipenn.tn.gov.in](https://pudhumaipenn.tn.gov.in)
        - **Naan Mudhalvan Youth Skill:** [https://www.naanmudhalvan.tn.gov.in](https://www.naanmudhalvan.tn.gov.in)
        """
    )

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; color: #94A3B8; font-size: 0.78rem; flex-wrap: wrap; gap: 8px;">
        <div>🏛️ <b>TN Scheme Eligibility Predictor</b> • Tamil Nadu Welfare Advisory Engine</div>
        <div>CS5403 Machine Learning Engineering Project</div>
    </div>
    """,
    unsafe_allow_html=True,
)
