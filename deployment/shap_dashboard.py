import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# App configuration
# ======================================================
st.set_page_config(
    page_title="SHAP Explainability Dashboard",
    layout="wide"
)

st.title("🔍 Model Explainability Dashboard (SHAP)")
st.markdown(
    "This dashboard provides global and local explanations of the trained model "
    "using SHAP values."
)

# ======================================================
# Resolve project root
# ======================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
X_TRAIN_PATH = ROOT_DIR / "data" / "processed" / "X_train.csv"
X_TEST_PATH = ROOT_DIR / "data" / "processed" / "X_test.csv"

# ======================================================
# Sidebar: Artifact status
# ======================================================
st.sidebar.header("📦 Artifact Status")

shap_ready = True
error_messages = []

if not MODEL_PATH.exists():
    shap_ready = False
    error_messages.append("❌ Model file missing")

if not X_TRAIN_PATH.exists() or not X_TEST_PATH.exists():
    shap_ready = False
    error_messages.append("❌ Processed data files missing")

if shap_ready:
    st.sidebar.success("✅ All SHAP artifacts available")
else:
    st.sidebar.error("⚠️ SHAP not ready")
    for msg in error_messages:
        st.sidebar.warning(msg)
    st.sidebar.info(
        "Run pipeline steps:\n"
        "`python workflows/pipeline.py`"
    )

# ======================================================
# Load artifacts (only if ready)
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    return X_train, X_test

if shap_ready:
    model = load_model()
    X_train, X_test = load_data()
    st.success("Model and processed data loaded successfully")

# ======================================================
# SHAP content (guarded)
# ======================================================
if not shap_ready:
    st.warning(
        "SHAP visualizations are disabled because required artifacts are missing."
    )
    st.stop()

# ======================================================
# SHAP Global Explanations
# ======================================================
st.header("🌍 Global SHAP Explanations")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Feature Importance (Top 10)")
    fig_bar = plt.figure(figsize=(6, 5))
    shap.plots.bar(
        shap_values,
        max_display=10,
        show=False
    )
    st.pyplot(fig_bar, clear_figure=True)

with col2:
    st.subheader("SHAP Summary (Top 10)")
    fig_summary = plt.figure(figsize=(6, 6))
    shap.plots.beeswarm(
        shap_values,
        max_display=10,
        show=False
    )
    st.pyplot(fig_summary, clear_figure=True)


st.header("🎯 Local Explanation & Prediction Distribution")

left_col, right_col = st.columns([2, 1])

# ---------- LEFT: Single Prediction Explanation ----------
with left_col:
    st.subheader("Single Prediction Explanation")

    row_id = st.slider(
        "Select a test data point",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0
    )

    row = X_test.iloc[[row_id]]
    shap_row = explainer(row)

    fig_waterfall = plt.figure(figsize=(7, 4))
    shap.plots.waterfall(
        shap_row[0],
        max_display=8,
        show=False
    )
    st.pyplot(fig_waterfall, clear_figure=True)

    st.markdown("**Input Feature Values**")
    st.dataframe(row, use_container_width=True)

# ---------- RIGHT: Pie Chart ----------
with right_col:
    st.subheader("Prediction Distribution")

    y_pred = model.predict(X_test)

    labels = ["Class 0", "Class 1"]
    values = [
        (y_pred == 0).sum(),
        (y_pred == 1).sum()
    ]

    fig_pie, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_facecolor("none")

    wedges, texts, autotexts = ax.pie(
        values,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11}
    )

    ax.legend(
        wedges,
        labels,
        title="Classes",
        loc="center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False
    )

    ax.axis("equal")
    plt.tight_layout()

    st.pyplot(fig_pie, clear_figure=True)
