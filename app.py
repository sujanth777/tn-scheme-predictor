"""
TN Scheme Eligibility Predictor
===============================
A professional civic-tech application for evaluating citizen eligibility 
across 20 Tamil Nadu Government welfare schemes using a trained 
MultiOutputClassifier (XGBoost) model with SHAP explainability.
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
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. DESIGN SYSTEM & REFINED CSS
# ==============================================================================
st.markdown(
    """
    <style>
    /* Modern, clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px !important;
    }

    /* Top Header Section */
    .app-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #006644;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .app-header-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F2942;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-header-desc {
        font-size: 0.98rem;
        color: #475569;
        margin: 0 0 14px 0;
        line-height: 1.5;
        max-width: 800px;
    }
    .app-header-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .meta-pill {
        display: inline-flex;
        align-items: center;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #334155;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
    }

    /* Metric Summary Strip */
    .stat-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .stat-card-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .stat-card-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .stat-card-sub {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Form Section Panels */
    .form-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }
    .form-panel-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #1E293B;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F1F5F9;
    }

    /* Scheme Results Card */
    .scheme-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #10B981;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .scheme-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 6px;
        flex-wrap: wrap;
    }
    .scheme-card-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F2942;
    }
    .scheme-card-dept {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 8px;
    }
    .scheme-card-desc {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    /* Badges & Indicators */
    .status-badge-eligible {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .category-badge {
        background: #F1F5F9;
        color: #475569;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
    }

    /* Progress bar track */
    .conf-meter {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .conf-meter-track {
        flex: 1;
        height: 8px;
        background: #F1F5F9;
        border-radius: 6px;
        overflow: hidden;
    }
    .conf-meter-bar {
        height: 100%;
        background: #10B981;
        border-radius: 6px;
    }
    .conf-meter-pct {
        font-size: 0.85rem;
        font-weight: 700;
        color: #065F46;
        min-width: 48px;
        text-align: right;
    }

    /* Explainability Factors */
    .shap-factor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.86rem;
    }
    .factor-pos {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }
    .factor-neg {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }
    .factor-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Criteria Grid Box */
    .criteria-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 10px;
    }
    .criteria-cell {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 8px 10px;
    }
    .criteria-cell-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .criteria-cell-val {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1E293B;
    }

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #F8FAFC;
        padding: 5px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 8px 20px !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0F2942 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    /* Primary Button */
    button[kind="primary"] {
        background-color: #006644 !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        transition: background-color 0.2s ease;
    }
    button[kind="primary"]:hover {
        background-color: #004d33 !important;
    }

    @media (max-width: 768px) {
        .stat-strip {
            grid-template-columns: repeat(2, 1fr);
        }
        .criteria-strip {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. METADATA & DOMAIN KNOWLEDGE
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
# 4. DATA & MODEL LOADERS (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner="Loading trained machine learning model...")
def load_model_artifacts():
    model_path = "scheme_model.pkl"
    cols_path = "model_feature_columns.pkl"
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        raise FileNotFoundError("Model artifacts (scheme_model.pkl, model_feature_columns.pkl) are missing.")
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    return model, feature_cols


@st.cache_data(show_spinner="Loading scheme reference rules...")
def load_reference_data():
    rules_path = "scheme_rules_full.xlsx"
    if not os.path.exists(rules_path):
        raise FileNotFoundError("Scheme rules spreadsheet (scheme_rules_full.xlsx) is missing.")
    rules_df = pd.read_excel(rules_path)
    comp_df = pd.read_csv("model_comparison.csv") if os.path.exists("model_comparison.csv") else pd.DataFrame()
    f1_df = pd.read_csv("per_scheme_f1.csv") if os.path.exists("per_scheme_f1.csv") else pd.DataFrame()
    return rules_df, comp_df, f1_df


try:
    model, expected_feature_columns = load_model_artifacts()
    rules_df, comparison_df, f1_df = load_reference_data()
    scheme_display_names = list(rules_df["SchemeName"])
except Exception as e:
    st.error(f"❌ Application Error: Failed to load required model assets: {e}")
    st.stop()


# ==============================================================================
# 5. EXPLAINABILITY & PREDICTION ENGINE
# ==============================================================================
def explain_scheme_prediction(estimator, encoded_row_df, raw_user_dict, feature_columns_list):
    """
    Computes top 3 feature importance contributions for a specific scheme's prediction.
    Attempts SHAP TreeExplainer first; falls back to estimator.feature_importances_ gracefully.
    """
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(encoded_row_df).values[0]

        top_indices = np.argsort(np.abs(shap_values))[::-1][:3]
        explanations = []

        for idx in top_indices:
            feat_name = feature_columns_list[idx]
            sv = float(shap_values[idx])

            if feat_name == "Age":
                label = "Age"
                val_str = str(raw_user_dict["Age"])
            elif feat_name == "Income":
                label = "Annual Income"
                val_str = f"₹{raw_user_dict['Income']:,}"
            elif "_" in feat_name:
                field, dummy_val = feat_name.rsplit("_", 1)
                label = FIELD_LABELS.get(field, field)
                val_str = raw_user_dict.get(field, dummy_val)
            else:
                label = FIELD_LABELS.get(feat_name, feat_name)
                val_str = str(raw_user_dict.get(feat_name, ""))

            if sv > 1.0:
                impact_text = "strongly increased eligibility"
            elif sv > 0:
                impact_text = "increased eligibility"
            elif sv < -1.0:
                impact_text = "strongly decreased eligibility"
            else:
                impact_text = "decreased eligibility"

            explanations.append({
                "label": label,
                "val_str": val_str,
                "text": f"{label} ({val_str}) {impact_text}",
                "score": sv,
                "is_positive": sv > 0,
            })

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

            explanations.append({
                "label": label,
                "val_str": val_str,
                "text": f"{label} ({val_str}) — relative importance: {weight:.1%}",
                "score": weight,
                "is_positive": True,
            })

        return explanations, "Top factors"


# Alias for backward compatibility
explain_prediction = explain_scheme_prediction


def preprocess_citizen_input(raw_dict, expected_columns):
    """
    Transforms raw citizen input into one-hot dummy encoded vector aligned exactly
    with model_feature_columns.pkl.
    """
    raw_df = pd.DataFrame([raw_dict])
    encoded_df = pd.get_dummies(raw_df).reindex(columns=expected_columns, fill_value=0).astype(int)
    if encoded_df.shape[1] != len(expected_columns):
        raise ValueError(
            f"Encoding error: expected {len(expected_columns)} features, but got {encoded_df.shape[1]}."
        )
    return encoded_df


# ==============================================================================
# 6. SIDEBAR CONTROLS & DEMO PROFILES
# ==============================================================================
DEMO_PRESETS = {
    "Select a pre-filled profile (optional)": None,
    "College Girl Student (Pudhumai Penn candidate)": {
        "Age": 20, "Gender": "Female", "Income": 120000, "Student": "Yes",
        "GovtSchoolStudent": "Yes", "Farmer": "No", "Widow": "No",
        "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
        "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
        "NoPermanentHouse": "No", "BPL": "No", "PrimaryEarnerDeceased": "No",
        "GirlChild": "No", "EnrolledTraining": "No",
    },
    "Elderly Low-Income Resident (Old Age Pension candidate)": {
        "Age": 68, "Gender": "Male", "Income": 45000, "Student": "No",
        "GovtSchoolStudent": "No", "Farmer": "No", "Widow": "No",
        "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
        "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
        "NoPermanentHouse": "Yes", "BPL": "Yes", "PrimaryEarnerDeceased": "No",
        "GirlChild": "No", "EnrolledTraining": "No",
    },
    "Small Cultivator / Farmer (PM-KISAN candidate)": {
        "Age": 42, "Gender": "Male", "Income": 180000, "Student": "No",
        "GovtSchoolStudent": "No", "Farmer": "Yes", "Widow": "No",
        "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
        "LandOwner": "Yes", "Pregnant": "No", "TNResident": "Yes",
        "NoPermanentHouse": "No", "BPL": "No", "PrimaryEarnerDeceased": "No",
        "GirlChild": "No", "EnrolledTraining": "No",
    },
    "BPL Working Woman (Magalir Urimai candidate)": {
        "Age": 38, "Gender": "Female", "Income": 85000, "Student": "No",
        "GovtSchoolStudent": "No", "Farmer": "No", "Widow": "No",
        "Disability": "No", "WorkingWoman": "Yes", "SC": "No", "ST": "No",
        "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
        "NoPermanentHouse": "No", "BPL": "Yes", "PrimaryEarnerDeceased": "No",
        "GirlChild": "Yes", "EnrolledTraining": "No",
    },
}

with st.sidebar:
    st.markdown("### 🏛️ TN Welfare Portal")
    st.caption("AI-Powered Scheme Eligibility Advisory")

    st.markdown("---")
    st.markdown("**⚡ Quick-Fill Profile (Demo Mode)**")
    preset_choice = st.selectbox(
        "Load sample profile:",
        options=list(DEMO_PRESETS.keys()),
        index=0,
        help="Quickly pre-fill the form with standard citizen archetypes for evaluation.",
    )

    if preset_choice and DEMO_PRESETS[preset_choice] is not None:
        if st.button("Apply Profile to Form", use_container_width=True):
            st.session_state.citizen_profile = DEMO_PRESETS[preset_choice].copy()
            st.rerun()

    st.markdown("---")
    st.markdown("**ℹ️ Citizen Advisory**")
    st.caption(
        "This platform is an automated screening aid. Eligibility predictions "
        "do not constitute official sanction. Final approvals require verified "
        "application via official Tamil Nadu e-Sevai portals."
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#94A3B8; text-align:center;'>"
        "Tamil Nadu Welfare Eligibility Engine • v2.1"
        "</div>",
        unsafe_allow_html=True,
    )


# Initialize session state for citizen profile if not set
if "citizen_profile" not in st.session_state:
    st.session_state.citizen_profile = {
        "Age": 22, "Gender": "Female", "Income": 120000, "Student": "Yes",
        "GovtSchoolStudent": "Yes", "Farmer": "No", "Widow": "No",
        "Disability": "No", "WorkingWoman": "No", "SC": "No", "ST": "No",
        "LandOwner": "No", "Pregnant": "No", "TNResident": "Yes",
        "NoPermanentHouse": "No", "BPL": "No", "PrimaryEarnerDeceased": "No",
        "GirlChild": "No", "EnrolledTraining": "No",
    }


# ==============================================================================
# 7. MAIN HEADER & HERO SECTION
# ==============================================================================
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">
            <span>🏛️</span>
            <span>TN Scheme Eligibility Predictor</span>
        </div>
        <div class="app-header-desc">
            Check your eligibility for Tamil Nadu government welfare schemes using your personal and socioeconomic information.
        </div>
        <div class="app-header-badges">
            <span class="meta-pill">MultiOutput XGBoost</span>
            <span class="meta-pill">SHAP Explainability</span>
            <span class="meta-pill">20 Welfare Schemes Evaluated</span>
            <span class="meta-pill">Model Macro F1: 99.7%</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 8. NAVIGATION TABS
# ==============================================================================
tab_check, tab_directory, tab_analytics, tab_about = st.tabs([
    "🎯 Eligibility Check",
    "📋 Scheme Directory",
    "📊 Model Performance",
    "ℹ️ About the Platform",
])


# ==============================================================================
# TAB 1: ELIGIBILITY CHECK (MAIN WORKFLOW)
# ==============================================================================
with tab_check:
    cur = st.session_state.citizen_profile

    def get_opt_idx(options_list, key, fallback="No"):
        val = cur.get(key, fallback)
        return options_list.index(val) if val in options_list else 0

    st.markdown("#### Enter Citizen Details")
    st.caption("Please provide accurate demographic and socio-economic information below.")

    with st.form("citizen_eligibility_form"):
        # Section 1: Personal Information
        st.markdown(
            """
            <div class="form-panel-title">
                <span>👤</span> <span>1. Personal Information</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            in_age = st.number_input(
                "Age (Years)",
                min_value=1,
                max_value=100,
                value=int(cur.get("Age", 22)),
                step=1,
                help="Citizen's current completed age in years.",
            )
        with p_c2:
            gender_options = ["Female", "Male"]
            in_gender = st.selectbox(
                "Gender",
                options=gender_options,
                index=get_opt_idx(gender_options, "Gender", "Female"),
            )
        with p_c3:
            tn_options = ["Yes", "No"]
            in_tn_resident = st.selectbox(
                "Tamil Nadu Resident",
                options=tn_options,
                index=get_opt_idx(tn_options, "TNResident", "Yes"),
                help="Citizen possesses legal domicile/nativity in Tamil Nadu.",
            )

        p_c4, p_c5 = st.columns(2)
        with p_c4:
            in_student = st.selectbox(
                "Currently Enrolled Student",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "Student"),
                help="Citizen is actively enrolled in school, diploma, or collegiate education.",
            )
        with p_c5:
            in_govt_school = st.selectbox(
                "Government School Education",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "GovtSchoolStudent"),
                help="Studied Class 6 to 12 in Tamil Nadu Government schools.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 2: Socio-Economic & Occupation
        st.markdown(
            """
            <div class="form-panel-title">
                <span>💼</span> <span>2. Socio-Economic & Occupation</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        o_c1, o_c2, o_c3 = st.columns(3)
        with o_c1:
            in_income = st.number_input(
                "Annual Household Income (INR)",
                min_value=0,
                max_value=1000000,
                value=int(cur.get("Income", 120000)),
                step=5000,
                help="Combined annual family income in Indian Rupees (INR).",
            )
        with o_c2:
            in_bpl = st.selectbox(
                "Below Poverty Line (BPL)",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "BPL"),
                help="Family holds BPL ration card or Antyodaya Anna Yojana (AAY) card.",
            )
        with o_c3:
            in_farmer = st.selectbox(
                "Farmer / Cultivator",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "Farmer"),
                help="Citizen's primary occupation is agricultural cultivation.",
            )

        o_c4, o_c5, o_c6 = st.columns(3)
        with o_c4:
            in_land_owner = st.selectbox(
                "Agricultural Land Owner",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "LandOwner"),
                help="Holds registered ownership of cultivable agricultural land.",
            )
        with o_c5:
            in_working_woman = st.selectbox(
                "Working Woman",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "WorkingWoman"),
                help="Woman employed in organized or unorganized occupational sector.",
            )
        with o_c6:
            in_enrolled_training = st.selectbox(
                "Enrolled in Skill Training",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "EnrolledTraining"),
                help="Enrolled in authorized skill training courses (e.g., Naan Mudhalvan).",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 3: Social Category
        st.markdown(
            """
            <div class="form-panel-title">
                <span>👥</span> <span>3. Social Category</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            in_sc = st.selectbox(
                "Scheduled Caste (SC)",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "SC"),
            )
        with s_c2:
            in_st = st.selectbox(
                "Scheduled Tribe (ST)",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "ST"),
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 4: Special Circumstances
        st.markdown(
            """
            <div class="form-panel-title">
                <span>🛡️</span> <span>4. Special Circumstances</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        x_c1, x_c2, x_c3 = st.columns(3)
        with x_c1:
            in_disability = st.selectbox(
                "Differently Abled (Disability)",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "Disability"),
                help="Citizen holds recognized disability benchmark certificate.",
            )
        with x_c2:
            in_widow = st.selectbox(
                "Widow",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "Widow"),
            )
        with x_c3:
            in_pregnant = st.selectbox(
                "Pregnant / Lactating Mother",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "Pregnant"),
            )

        x_c4, x_c5, x_c6 = st.columns(3)
        with x_c4:
            in_no_house = st.selectbox(
                "No Permanent House (Kutcha)",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "NoPermanentHouse"),
                help="Household does not own a pucca/concrete permanent residential house.",
            )
        with x_c5:
            in_earner_deceased = st.selectbox(
                "Primary Breadwinner Deceased",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "PrimaryEarnerDeceased"),
                help="Primary earning member of the family has passed away.",
            )
        with x_c6:
            in_girl_child = st.selectbox(
                "Family with Girl Child",
                options=["No", "Yes"],
                index=get_opt_idx(["No", "Yes"], "GirlChild"),
                help="Household has eligible female child beneficiaries.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Primary Submission Action
        submit_btn = st.form_submit_button(
            "Check Eligibility",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------------------------
    # PREDICTION & RESULTS PRESENTATION
    # --------------------------------------------------------------------------
    if submit_btn:
        active_input = {
            "Age": int(in_age),
            "Gender": str(in_gender),
            "Income": int(in_income),
            "Student": str(in_student),
            "GovtSchoolStudent": str(in_govt_school),
            "Farmer": str(in_farmer),
            "Widow": str(in_widow),
            "Disability": str(in_disability),
            "WorkingWoman": str(in_working_woman),
            "SC": str(in_sc),
            "ST": str(in_st),
            "LandOwner": str(in_land_owner),
            "Pregnant": str(in_pregnant),
            "TNResident": str(in_tn_resident),
            "NoPermanentHouse": str(in_no_house),
            "BPL": str(in_bpl),
            "PrimaryEarnerDeceased": str(in_earner_deceased),
            "GirlChild": str(in_girl_child),
            "EnrolledTraining": str(in_enrolled_training),
        }
        st.session_state.citizen_profile = active_input.copy()

        try:
            # 1. Preprocessing with strict encoding parity
            encoded_row = preprocess_citizen_input(active_input, expected_feature_columns)

            # 2. MultiOutputClassifier Inference
            raw_predictions = model.predict(encoded_row)[0]
            raw_probabilities = model.predict_proba(encoded_row)

            # 3. Compile structured scheme outcomes
            evaluated_schemes = []
            for idx, scheme_name in enumerate(scheme_display_names):
                is_eligible = bool(raw_predictions[idx] == 1)
                confidence_score = float(raw_probabilities[idx][0, 1])
                rule_row = rules_df.iloc[idx].to_dict() if idx < len(rules_df) else {}
                meta_info = SCHEME_META.get(scheme_name, {
                    "cat": "Welfare",
                    "dept": "Government of Tamil Nadu",
                    "desc": "Tamil Nadu State Welfare Program.",
                })

                evaluated_schemes.append({
                    "index": idx,
                    "name": scheme_name,
                    "is_eligible": is_eligible,
                    "confidence": confidence_score,
                    "estimator": model.estimators_[idx],
                    "rules": rule_row,
                    "meta": meta_info,
                })

            eligible_list = [s for s in evaluated_schemes if s["is_eligible"]]
            eligible_list.sort(key=lambda s: s["confidence"], reverse=True)

            ineligible_list = [s for s in evaluated_schemes if not s["is_eligible"]]
            ineligible_list.sort(key=lambda s: s["confidence"], reverse=True)

            # ------------------------------------------------------------------
            # Results Dashboard
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### 📋 Eligibility Results")

            highest_conf_str = f"{eligible_list[0]['confidence'] * 100:.1f}%" if eligible_list else "0.0%"
            avg_conf_val = (
                f"{np.mean([s['confidence'] for s in eligible_list]) * 100:.1f}%"
                if eligible_list else "0.0%"
            )

            st.markdown(
                f"""
                <div class="stat-strip">
                    <div class="stat-card">
                        <div class="stat-card-label">Schemes Evaluated</div>
                        <div class="stat-card-value">20</div>
                        <div class="stat-card-sub">Active State Welfare Schemes</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-label">Eligible Schemes</div>
                        <div class="stat-card-value" style="color: #059669;">{len(eligible_list)}</div>
                        <div class="stat-card-sub">Criteria Satisfied</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-label">Highest Confidence</div>
                        <div class="stat-card-value" style="color: #0F2942;">{highest_conf_str}</div>
                        <div class="stat-card-sub">Top Predicted Match</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-label">Not Eligible</div>
                        <div class="stat-card-value" style="color: #64748B;">{len(ineligible_list)}</div>
                        <div class="stat-card-sub">Criteria Unmet</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Section: Eligible Schemes
            st.markdown("#### Eligible Schemes")
            st.caption("Based on the input profile, the citizen is predicted eligible for the following programs:")

            if eligible_list:
                for scheme in eligible_list:
                    conf_pct = scheme["confidence"] * 100
                    s_name = scheme["name"]
                    meta = scheme["meta"]
                    rules = scheme["rules"]

                    st.markdown(
                        f"""
                        <div class="scheme-card">
                            <div class="scheme-card-header">
                                <div>
                                    <span class="scheme-card-name">{s_name}</span>
                                    <span class="category-badge" style="margin-left: 8px;">{meta['cat']}</span>
                                    <div class="scheme-card-dept">🏢 {meta['dept']}</div>
                                </div>
                                <div>
                                    <span class="status-badge-eligible">Eligible</span>
                                </div>
                            </div>
                            <div class="scheme-card-desc">{meta['desc']}</div>
                            <div class="conf-meter">
                                <div class="conf-meter-track">
                                    <div class="conf-meter-bar" style="width: {conf_pct}%;"></div>
                                </div>
                                <div class="conf-meter-pct">{conf_pct:.1f}%</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Explainability Sub-Panel
                    with st.expander(f"Why did the model give this result? — {s_name}", expanded=False):
                        factors, method_label = explain_scheme_prediction(
                            estimator=scheme["estimator"],
                            encoded_row_df=encoded_row,
                            raw_user_dict=active_input,
                            feature_columns_list=expected_feature_columns,
                        )

                        st.caption(f"Top decision factors identified via {method_label}:")
                        for factor in factors:
                            factor_class = "factor-pos" if factor["is_positive"] else "factor-neg"
                            arrow = "▲" if factor["is_positive"] else "▼"
                            sign = "+" if factor["score"] > 0 else ""
                            st.markdown(
                                f"""
                                <div class="shap-factor-row {factor_class}">
                                    <span>{arrow} <b>{factor['text']}</b></span>
                                    <span class="factor-score">{sign}{factor['score']:.2f}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Official Eligibility Criteria reference
                        inc_limit = rules.get("IncomeLimit", "No Limit")
                        inc_display = f"₹{inc_limit:,}" if isinstance(inc_limit, (int, float)) else str(inc_limit)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.caption("Official Scheme Eligibility Guidelines:")
                        st.markdown(
                            f"""
                            <div class="criteria-strip">
                                <div class="criteria-cell">
                                    <div class="criteria-cell-label">Gender</div>
                                    <div class="criteria-cell-val">{rules.get('Gender', 'Any')}</div>
                                </div>
                                <div class="criteria-cell">
                                    <div class="criteria-cell-label">Age Bracket</div>
                                    <div class="criteria-cell-val">{rules.get('MinAge', 0)} – {rules.get('MaxAge', 100)} yrs</div>
                                </div>
                                <div class="criteria-cell">
                                    <div class="criteria-cell-label">Income Ceiling</div>
                                    <div class="criteria-cell-val">{inc_display}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Technical details toggle
                        with st.expander("Technical Model Details", expanded=False):
                            st.caption(
                                f"Model Estimator: XGBClassifier (Tree #{scheme['index']})\n"
                                f"Output Probability: {scheme['confidence']:.4f}\n"
                                f"Binary Decision: {int(scheme['is_eligible'])}\n"
                                f"Explainability Engine: {method_label}"
                            )
            else:
                st.info(
                    "No schemes currently match this citizen profile based on the model's eligibility thresholds. "
                    "You may verify the details above or consult official Tamil Nadu e-Sevai portals for special assistance programs."
                )

            # Section: Ineligible Schemes (Collapsed)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander(f"Schemes Not Currently Eligible ({len(ineligible_list)})", expanded=False):
                if ineligible_list:
                    st.caption("The citizen profile does not currently satisfy eligibility criteria for the following programs:")
                    for inelig in ineligible_list:
                        conf_pct = inelig["confidence"] * 100
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; margin-bottom: 6px;">
                                <div>
                                    <span style="font-weight: 600; color: #334155;">{inelig['name']}</span>
                                    <span class="category-badge" style="margin-left: 6px;">{inelig['meta']['cat']}</span>
                                </div>
                                <span style="font-size: 0.8rem; font-weight: 600; color: #94A3B8;">{conf_pct:.1f}% Match</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("This profile satisfies eligibility criteria for all 20 evaluated schemes.")

        except Exception as err:
            st.error(
                f"❌ Error during eligibility evaluation: {str(err)}\n\n"
                "Please verify that all input values are formatted correctly."
            )


# ==============================================================================
# TAB 2: SCHEME DIRECTORY
# ==============================================================================
with tab_directory:
    st.markdown("#### Tamil Nadu Welfare Schemes Directory")
    st.caption("Official reference criteria for all 20 welfare programs supported by the model.")

    d_col1, d_col2 = st.columns([2, 1])
    with d_col1:
        search_kw = st.text_input("🔍 Search schemes:", placeholder="Search by name, department, or keyword...")
    with d_col2:
        cat_filter_options = ["All Categories"] + sorted(list(set([m["cat"] for m in SCHEME_META.values()])))
        selected_category = st.selectbox("Filter by Category:", options=cat_filter_options)

    # Filter rules
    visible_schemes = []
    for idx, row in rules_df.iterrows():
        s_name = row["SchemeName"]
        meta = SCHEME_META.get(s_name, {"cat": "Welfare", "dept": "Government of Tamil Nadu", "desc": ""})
        category_matches = (selected_category == "All Categories") or (meta["cat"] == selected_category)
        keyword_matches = (
            not search_kw or
            search_kw.lower() in s_name.lower() or
            search_kw.lower() in meta["dept"].lower() or
            search_kw.lower() in meta["desc"].lower() or
            search_kw.lower() in meta["cat"].lower()
        )
        if category_matches and keyword_matches:
            visible_schemes.append((idx, row, meta))

    st.markdown(f"Showing **{len(visible_schemes)}** of 20 Schemes")

    grid_cols = st.columns(2)
    for i, (idx, row, meta) in enumerate(visible_schemes):
        inc_lim = row.get("IncomeLimit", "No Limit")
        inc_str = f"₹{inc_lim:,}" if isinstance(inc_lim, (int, float)) else str(inc_lim)
        col_target = grid_cols[i % 2]

        with col_target:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px 18px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                        <span style="font-size: 1.05rem; font-weight: 700; color: #0F2942;">{row['SchemeName']}</span>
                        <span class="category-badge">{meta['cat']}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">🏢 {meta['dept']}</div>
                    <div style="font-size: 0.85rem; color: #334155; line-height: 1.45; margin-bottom: 12px;">{meta['desc']}</div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 10px; font-size: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                            <span style="color: #64748B;">Target Gender:</span> <b>{row.get('Gender', 'Any')}</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                            <span style="color: #64748B;">Age Range:</span> <b>{row.get('MinAge', 0)} – {row.get('MaxAge', 100)} yrs</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                            <span style="color: #64748B;">Income Ceiling:</span> <b>{inc_str}</b>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #64748B;">TN Residency:</span> <b>{row.get('TNResident', 'Any')}</b>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 3: MODEL PERFORMANCE & BENCHMARKS
# ==============================================================================
with tab_analytics:
    st.markdown("#### Machine Learning Evaluation & Benchmarks")
    st.caption("Performance metrics of the multi-label XGBoost classification model.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Macro F1", "99.7%", help="Fair evaluation across all schemes, balanced for rare classes.")
    m2.metric("Micro F1", "99.8%", help="Overall label accuracy across the complete test matrix.")
    m3.metric("Exact Match", "99.3%", help="Strict accuracy where all 20 schemes are correctly predicted per citizen.")
    m4.metric("Estimators", "20 Trees", help="MultiOutputClassifier wrapping 20 dedicated XGBoost binary classifiers.")

    st.markdown("<br>", unsafe_allow_html=True)

    if not comparison_df.empty:
        st.markdown("**1. Multi-Model Benchmark Comparison**")
        st.caption("Comparison of candidate models evaluated during pipeline selection:")
        st.dataframe(
            comparison_df,
            column_config={
                "Exact Match Accuracy": st.column_config.ProgressColumn("Exact Match", format="%.3f", min_value=0, max_value=1),
                "F1 (micro)": st.column_config.ProgressColumn("F1 (micro)", format="%.3f", min_value=0, max_value=1),
                "F1 (macro)": st.column_config.ProgressColumn("F1 (macro)", format="%.3f", min_value=0, max_value=1),
            },
            use_container_width=True,
            hide_index=True,
        )

    if not f1_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**2. Per-Scheme F1 Score Breakdown**")
        with st.expander("View individual scheme test scores", expanded=False):
            st.dataframe(
                f1_df,
                column_config={
                    "F1": st.column_config.ProgressColumn("F1 Score", format="%.3f", min_value=0.9, max_value=1.0)
                },
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**3. Training Dataset Distributions**")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if os.path.exists("eligible_counts_per_scheme.png"):
            st.image("eligible_counts_per_scheme.png", caption="Eligible Citizen Counts per Welfare Scheme", use_container_width=True)
        if os.path.exists("income_distribution.png"):
            st.image("income_distribution.png", caption="Income Distribution in Training Dataset", use_container_width=True)
    with v_col2:
        if os.path.exists("age_distribution.png"):
            st.image("age_distribution.png", caption="Age Distribution in Training Dataset", use_container_width=True)
        if os.path.exists("confusion_matrices_weakest_schemes.png"):
            st.image("confusion_matrices_weakest_schemes.png", caption="Confusion Matrices for Lowest F1 Schemes", use_container_width=True)


# ==============================================================================
# TAB 4: ABOUT THE PLATFORM
# ==============================================================================
with tab_about:
    st.markdown("#### About the Platform")
    st.markdown(
        """
        The **TN Scheme Eligibility Predictor** is an AI-powered advisory tool designed to assist 
        citizens, social workers, and administrative volunteers in discovering applicable welfare programs 
        funded by the Government of Tamil Nadu.

        ##### How It Works
        1. **Citizen Input:** The user provides 19 demographic, financial, and occupational attributes.
        2. **Feature Alignment:** Inputs are one-hot encoded into a 36-dimensional vector strictly matching the training format.
        3. **Multi-Output Prediction:** A multi-label XGBoost classifier evaluates eligibility across 20 state schemes simultaneously.
        4. **Explainability Layer:** SHAP (SHapley Additive exPlanations) analyzes individual tree weights to explain which citizen attributes most strongly influenced each recommendation.

        ##### Data & Training Disclosures
        - **Synthetic Training Dataset:** Because real citizen social welfare records are private and protected, 
          the model was trained on synthetic citizen profiles generated strictly from official state eligibility guidelines.
        - **Advisory Status:** This software is an academic engineering project (CS5403 PBL Review). 
          It is not an official government sanctioning authority. Citizens must submit formal applications through 
          designated government portals.
        """
    )

    st.markdown("---")
    st.markdown("##### Official Government Portals")
    st.markdown(
        "- **TNeGA / Tamil Nadu e-Sevai Portal:** [https://www.tnesevai.tn.gov.in](https://www.tnesevai.tn.gov.in)\n"
        "- **Kalaignar Magalir Urimai Thittam:** [https://kmut.tn.gov.in](https://kmut.tn.gov.in)\n"
        "- **Pudhumai Penn Scheme Portal:** [https://pudhumaipenn.tn.gov.in](https://pudhumaipenn.tn.gov.in)\n"
        "- **Naan Mudhalvan Youth Skill Portal:** [https://www.naanmudhalvan.tn.gov.in](https://www.naanmudhalvan.tn.gov.in)\n"
        "- **Chief Minister's Comprehensive Health Insurance (CMCHIS):** [https://cmchistn.com](https://cmchistn.com)"
    )


# ==============================================================================
# 9. FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; color: #94A3B8; font-size: 0.82rem; flex-wrap: wrap; gap: 8px;">
        <div>
            🏛️ <b>TN Scheme Eligibility Predictor</b> • Tamil Nadu Welfare Advisory Engine
        </div>
        <div>
            Machine Learning Engineering Project • CS5403
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
