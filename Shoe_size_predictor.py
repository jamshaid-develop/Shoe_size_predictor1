import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shoe Size Predictor",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.stApp { background-color: #0f1117; color: #ffffff; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1d2e !important;
    border-right: 1px solid #2d3250;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    padding: 4px 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2235, #252a40);
    border: 1px solid #2d3250;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card .label {
    font-size: 0.78rem;
    color: #8b92a5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #4fc3f7;
}

/* Section title */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #4fc3f7;
    border-left: 4px solid #4fc3f7;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Page title */
.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.95rem;
    color: #8b92a5;
    margin-bottom: 1.5rem;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #1a237e, #283593);
    border: 1px solid #3949ab;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 8px 30px rgba(63,81,181,0.3);
}
.result-box .rlabel { font-size: 1rem; color: #90caf9; margin-bottom: 0.5rem; }
.result-box .rvalue { font-size: 4rem; font-weight: 900; color: #ffffff; margin: 0; }
.result-box .rsub   { color: #90caf9; font-size: 0.88rem; margin-top: 0.5rem; }

/* Input card */
.input-card {
    background: #1e2235;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.4rem 0;
}
.input-card .ilabel {
    font-size: 0.75rem; color: #8b92a5;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.input-card .ivalue {
    font-size: 1.5rem; font-weight: 700; color: #4fc3f7;
}

/* Divider */
hr { border-color: #2d3250; }

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1976d2);
    color: white; border: none; border-radius: 10px;
    padding: 0.7rem 1.5rem; font-size: 1rem;
    font-weight: 600; width: 100%;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(21,101,192,0.4);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1e88e5);
    box-shadow: 0 6px 20px rgba(21,101,192,0.6);
}

/* Streamlit default overrides for dark */
.stDataFrame { background: #1e2235; }
[data-testid="stMetricValue"] { color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data & Model ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "shoe_size_dataset.csv")
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, "shoe_size_model.pkl")
    if not os.path.exists(path):
        return None, None, None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        return bundle.get("model"), bundle.get("scaler"), bundle.get("features")
    return bundle, None, ["height_cm", "weight_kg"]

df = load_data()
model, scaler, features = load_model()
model_loaded = model is not None

# chart style
def set_style():
    plt.rcParams.update({
        "figure.facecolor":  "#1e2235",
        "axes.facecolor":    "#1e2235",
        "axes.edgecolor":    "#2d3250",
        "axes.labelcolor":   "#ffffff",
        "xtick.color":       "#8b92a5",
        "ytick.color":       "#8b92a5",
        "text.color":        "#ffffff",
        "grid.color":        "#2d3250",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
    })

ACCENT   = "#4fc3f7"
ACCENT2  = "#ef5350"
ACCENT3  = "#66bb6a"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, "#ffa726", "#ab47bc"]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👟 Shoe Size Predictor")
    st.markdown("---")

    st.markdown("### 🗂️ Navigation")
    page = st.radio("", [
        "🏠 Overview",
        "📊 Distributions",
        "🔗 Correlation",
        "📈 Model Performance",
        "🔍 Predict"
    ], label_visibility="collapsed")

    st.markdown("---")

    if page == "🔍 Predict":
        st.markdown("### 📥 Enter Details")
        height = st.slider("📏 Height (cm)", 140.0, 220.0, 170.0, 0.5)
        weight = st.slider("⚖️ Weight (kg)",  40.0, 150.0,  70.0, 0.5)
        predict_btn = st.button("🔍 Predict Shoe Size", disabled=not model_loaded)
    else:
        height, weight, predict_btn = 170.0, 70.0, False

    st.markdown("---")
    st.markdown("### 🤖 Model Status")
    if model_loaded:
        st.success("✅ Model Ready")
        st.markdown("🌲 Random Forest · ⚖️ StandardScaler")
    else:
        st.error("❌ Model not found")

    st.markdown("---")
    st.markdown("<small style='color:#8b92a5'>500 samples · 2 features · UK shoe size</small>",
                unsafe_allow_html=True)


# ─── Helper: metric card ─────────────────────────────────────────────────────
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="page-title">👟 Shoe Size Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Exploratory Data Analysis · Random Forest Model · 500 Samples</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Samples",   f"{len(df):,}")
    with c2: metric_card("Avg Height",      f"{df['height_cm'].mean():.1f} cm")
    with c3: metric_card("Avg Weight",      f"{df['weight_kg'].mean():.1f} kg")
    with c4: metric_card("Avg Shoe Size",   f"{df['shoe_size'].mean():.1f}")

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    with c5: metric_card("Min Shoe Size",   f"{df['shoe_size'].min():.1f}")
    with c6: metric_card("Max Shoe Size",   f"{df['shoe_size'].max():.1f}")
    with c7: metric_card("Height Range",    f"{df['height_cm'].min():.0f}–{df['height_cm'].max():.0f}")
    with c8: metric_card("Weight Range",    f"{df['weight_kg'].min():.0f}–{df['weight_kg'].max():.0f}")

    # Dataset preview
    st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**First 10 Rows**")
        st.dataframe(df.head(10), use_container_width=True)

    with col_r:
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().round(2), use_container_width=True)

    # Dataset info
    st.markdown('<div class="section-title">🗂️ Dataset Info</div>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column":    df.columns.tolist(),
        "Dtype":     [str(df[c].dtype) for c in df.columns],
        "Non-Null":  [df[c].notna().sum() for c in df.columns],
        "Nulls":     [df[c].isna().sum() for c in df.columns],
        "Unique":    [df[c].nunique() for c in df.columns],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Distributions":
    st.markdown('<div class="page-title">📊 Feature Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">How height, weight, and shoe size are distributed across 500 samples</div>', unsafe_allow_html=True)
    st.markdown("---")

    set_style()

    # Row 1: histograms
    st.markdown('<div class="section-title">📉 Histograms</div>', unsafe_allow_html=True)
    cols_hist = st.columns(3)
    hist_config = [
        ("height_cm", "Height (cm)", ACCENT),
        ("weight_kg", "Weight (kg)", ACCENT2),
        ("shoe_size", "Shoe Size",   ACCENT3),
    ]
    for col, (feature, label, color) in zip(cols_hist, hist_config):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df[feature], bins=20, color=color, alpha=0.85, edgecolor="#0f1117")
            ax.set_title(label, fontsize=11, fontweight="bold", color="#ffffff")
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Row 2: boxplots
    st.markdown('<div class="section-title">📦 Boxplots</div>', unsafe_allow_html=True)
    cols_box = st.columns(3)
    for col, (feature, label, color) in zip(cols_box, hist_config):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            bp = ax.boxplot(df[feature], patch_artist=True, notch=False,
                            medianprops=dict(color="#ffffff", linewidth=2))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.8)
            ax.set_title(f"{label} Boxplot", fontsize=11, fontweight="bold", color="#ffffff")
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Row 3: scatter plots
    st.markdown('<div class="section-title">🔵 Scatter Plots vs Shoe Size</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    for col, (feature, label, color) in zip([sc1, sc2], hist_config[:2]):
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df[feature], df["shoe_size"], color=color,
                       alpha=0.5, s=18, edgecolors="none")
            m, b = np.polyfit(df[feature], df["shoe_size"], 1)
            x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
            ax.plot(x_line, m * x_line + b, color="#ffffff", linewidth=1.5, linestyle="--")
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("Shoe Size", fontsize=10)
            ax.set_title(f"{label} vs Shoe Size", fontsize=11, fontweight="bold", color="#ffffff")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Correlation":
    st.markdown('<div class="page-title">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Relationships between all features in the dataset</div>', unsafe_allow_html=True)
    st.markdown("---")

    set_style()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        corr = df.corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = False
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5, linecolor="#0f1117",
                    annot_kws={"size": 12, "weight": "bold"},
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Matrix", fontsize=12,
                     fontweight="bold", color="#ffffff", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown('<div class="section-title">📊 Correlation Bar Chart</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        corr_shoe = df.corr()["shoe_size"].drop("shoe_size").sort_values()
        colors = [ACCENT2 if v < 0 else ACCENT for v in corr_shoe]
        bars = ax.barh(corr_shoe.index, corr_shoe.values, color=colors,
                       edgecolor="#0f1117", height=0.5)
        ax.set_xlabel("Correlation with Shoe Size", fontsize=10)
        ax.set_title("Feature Correlation with Shoe Size",
                     fontsize=11, fontweight="bold", color="#ffffff")
        ax.axvline(0, color="#ffffff", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, corr_shoe.values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=10, color="#ffffff")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Pair plot style — manual
    st.markdown('<div class="section-title">🔵 Pairwise Scatter Matrix</div>', unsafe_allow_html=True)
    cols_all = ["height_cm", "weight_kg", "shoe_size"]
    n = len(cols_all)
    fig, axes = plt.subplots(n, n, figsize=(10, 8))
    for i, row_feat in enumerate(cols_all):
        for j, col_feat in enumerate(cols_all):
            ax = axes[i][j]
            if i == j:
                ax.hist(df[row_feat], bins=20, color=PALETTE[i], alpha=0.8, edgecolor="#0f1117")
            else:
                ax.scatter(df[col_feat], df[row_feat], color=PALETTE[i],
                           alpha=0.4, s=8, edgecolors="none")
            if i == n - 1: ax.set_xlabel(col_feat, fontsize=8)
            if j == 0:     ax.set_ylabel(row_feat, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Pairwise Scatter Matrix", fontsize=13,
                 fontweight="bold", color="#ffffff", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<div class="page-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Random Forest Regressor evaluation on the dataset</div>', unsafe_allow_html=True)
    st.markdown("---")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    set_style()

    X = df[["height_cm", "weight_kg"]].values
    y = df["shoe_size"].values
    X_scaled = scaler.transform(X) if scaler else X
    y_pred = model.predict(X_scaled)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    # Metrics
    mc1, mc2, mc3 = st.columns(3)
    with mc1: metric_card("MAE",  f"{mae:.4f}")
    with mc2: metric_card("RMSE", f"{rmse:.4f}")
    with mc3: metric_card("R² Score", f"{r2:.4f}")

    st.markdown("")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">✅ Actual vs Predicted</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y, y_pred, color=ACCENT, alpha=0.5, s=18, edgecolors="none")
        mn, mx = y.min(), y.max()
        ax.plot([mn, mx], [mn, mx], color=ACCENT2, linewidth=1.5,
                linestyle="--", label="Perfect Fit")
        ax.set_xlabel("Actual Shoe Size",    fontsize=10)
        ax.set_ylabel("Predicted Shoe Size", fontsize=10)
        ax.set_title("Actual vs Predicted",  fontsize=11, fontweight="bold", color="#ffffff")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown('<div class="section-title">📉 Residuals Distribution</div>', unsafe_allow_html=True)
        residuals = y - y_pred
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(residuals, bins=25, color=ACCENT3, alpha=0.85, edgecolor="#0f1117")
        ax.axvline(0, color=ACCENT2, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
        ax.set_ylabel("Frequency",                     fontsize=10)
        ax.set_title("Residuals Distribution",          fontsize=11, fontweight="bold", color="#ffffff")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    st.markdown('<div class="section-title">🌟 Feature Importance</div>', unsafe_allow_html=True)
    importances = model.feature_importances_
    feat_names  = ["Height (cm)", "Weight (kg)"]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(feat_names, importances, color=[ACCENT, ACCENT2],
                   edgecolor="#0f1117", height=0.4)
    for bar, val in zip(bars, importances):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=11, color="#ffffff")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("Random Forest Feature Importance",
                 fontsize=11, fontweight="bold", color="#ffffff")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown('<div class="page-title">🔍 Predict Your Shoe Size</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter your height and weight in the sidebar to get a prediction</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Input summary
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(f"""
        <div class="input-card">
            <div class="ilabel">📏 Height</div>
            <div class="ivalue">{height} cm</div>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""
        <div class="input-card">
            <div class="ilabel">⚖️ Weight</div>
            <div class="ivalue">{weight} kg</div>
        </div>""", unsafe_allow_html=True)

    if predict_btn:
        inp = np.array([[height, weight]])
        if scaler:
            inp = scaler.transform(inp)
        raw = model.predict(inp)[0]
        size = round(raw * 2) / 2

        st.markdown(f"""
        <div class="result-box">
            <div class="rlabel">🎯 Your Predicted Shoe Size</div>
            <div class="rvalue">{size}</div>
            <div class="rsub">🌲 Random Forest · ⚖️ StandardScaler · UK Size</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">🌍 Size Conversion Guide</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("🇬🇧 UK",        f"{size}")
        s2.metric("🇪🇺 EU",        f"{size + 33:.0f}")
        s3.metric("🇺🇸 US Men's",  f"{size + 1:.1f}")
        s4.metric("🇺🇸 US Women's",f"{size + 2.5:.1f}")
        st.info("ℹ️ Conversions are approximate. Always try shoes in-store for best fit.")

        # Where does this size fall
        st.markdown('<div class="section-title">📊 Your Size in Dataset</div>', unsafe_allow_html=True)
        set_style()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(df["shoe_size"], bins=20, color=ACCENT, alpha=0.7,
                edgecolor="#0f1117", label="Dataset")
        ax.axvline(size, color=ACCENT2, linewidth=2.5,
                   linestyle="--", label=f"Your Size: {size}")
        ax.set_xlabel("Shoe Size", fontsize=10)
        ax.set_ylabel("Count",     fontsize=10)
        ax.set_title("Your Predicted Size vs Dataset Distribution",
                     fontsize=11, fontweight="bold", color="#ffffff")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.markdown("---")
        st.info("👈 Adjust **Height** and **Weight** sliders in the sidebar, then click **Predict Shoe Size**.")

        # Show dataset distribution as teaser
        set_style()
        st.markdown('<div class="section-title">📊 Shoe Size Distribution in Dataset</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(df["shoe_size"], bins=20, color=ACCENT, alpha=0.8, edgecolor="#0f1117")
        ax.set_xlabel("Shoe Size", fontsize=10)
        ax.set_ylabel("Count",     fontsize=10)
        ax.set_title("Shoe Size Distribution (500 Samples)",
                     fontsize=11, fontweight="bold", color="#ffffff")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()