"""
Crop Recommendation System - Streamlit App
==========================================
A simple web interface for predicting the best crop based on
soil and climate conditions.
"""

import streamlit as st
import pickle
import numpy as np
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "crop_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


# ── Crop Info Dictionary ──────────────────────────────────────────────────────
CROP_INFO = {
    "rice": ("🌾", "Requires high humidity and rainfall. Best in clayey loam soil."),
    "wheat": ("🌾", "Thrives in cool climate. Needs well-drained loamy soil."),
    "maize": ("🌽", "Grows well in warm climate. Requires well-drained fertile soil."),
    "chickpea": ("🫘", "Prefers dry, cool weather. Grows in well-drained sandy loam."),
    "kidneybeans": ("🫘", "Needs warm temps and moderate rainfall. Loamy soil is ideal."),
    "pigeonpeas": ("🫘", "Drought-tolerant. Grows in a variety of soil types."),
    "mothbeans": ("🌿", "Very drought-resistant. Sandy or loamy soil preferred."),
    "mungbean": ("🌿", "Warm climate crop. Well-drained loamy soils work best."),
    "blackgram": ("🌿", "Tolerates drought. Grows well in loamy or sandy soil."),
    "lentil": ("🫘", "Cool climate crop. Well-drained loamy or clay soil."),
    "pomegranate": ("🍎", "Thrives in semi-arid climates. Tolerates poor soils."),
    "banana": ("🍌", "Needs high humidity and warmth. Deep, fertile loamy soil."),
    "mango": ("🥭", "Tropical fruit. Grows in deep, well-drained alluvial soil."),
    "grapes": ("🍇", "Thrives in warm, dry climates. Well-drained sandy loam."),
    "watermelon": ("🍉", "Warm-season crop. Sandy loam with good drainage."),
    "muskmelon": ("🍈", "Warm climate. Sandy loam soil with excellent drainage."),
    "apple": ("🍎", "Cool climate. Well-drained loamy soil with good aeration."),
    "orange": ("🍊", "Subtropical climate. Deep, fertile, well-drained soils."),
    "papaya": ("🍈", "Tropical climate. Light, well-drained alluvial soil."),
    "coconut": ("🥥", "Tropical coastal crop. Sandy loam with good water retention."),
    "cotton": ("🌸", "Warm, dry climate. Deep, well-drained black cotton soil."),
    "jute": ("🌿", "Warm, humid climate. Alluvial soil with good water retention."),
    "coffee": ("☕", "Tropical highland crop. Well-drained, slightly acidic soil."),
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 Crop Recommendation System")
st.markdown(
    "Enter your **soil nutrients** and **climate conditions** below to find out "
    "which crop is best suited for your field."
)
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Field Conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input(
        "Nitrogen (N) — kg/ha", min_value=0, max_value=200, value=50,
        help="Nitrogen content in soil (0–200 kg/ha)"
    )
    P = st.number_input(
        "Phosphorus (P) — kg/ha", min_value=0, max_value=200, value=50,
        help="Phosphorus content in soil (0–200 kg/ha)"
    )
    K = st.number_input(
        "Potassium (K) — kg/ha", min_value=0, max_value=200, value=50,
        help="Potassium content in soil (0–200 kg/ha)"
    )
    ph = st.slider(
        "Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1,
        help="pH value of the soil (0–14)"
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1,
        help="Average temperature in Celsius"
    )
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1,
        help="Relative humidity percentage"
    )
    rainfall = st.number_input(
        "Rainfall (mm)", min_value=0.0, max_value=3000.0, value=200.0, step=1.0,
        help="Average annual rainfall in mm"
    )

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Recommend Crop", use_container_width=True, type="primary"):
    try:
        model, le = load_model()
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction_enc = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        crop_name = le.inverse_transform([prediction_enc])[0]

        emoji, info = CROP_INFO.get(crop_name, ("🌱", "A great choice for your conditions."))

        # Top-3 recommendations
        top3_idx = np.argsort(probabilities)[::-1][:3]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probabilities[top3_idx]

        st.success(f"### {emoji} Recommended Crop: **{crop_name.title()}**")
        st.info(f"📌 {info}")

        st.subheader("📊 Top 3 Recommendations")
        for rank, (crop, prob) in enumerate(zip(top3_crops, top3_probs), start=1):
            emoji_c, _ = CROP_INFO.get(crop, ("🌱", ""))
            st.progress(float(prob), text=f"#{rank} {emoji_c} **{crop.title()}** — {prob * 100:.1f}% confidence")

        st.subheader("🧪 Input Summary")
        summary_data = {
            "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                          "Temperature", "Humidity", "pH", "Rainfall"],
            "Value": [f"{N} kg/ha", f"{P} kg/ha", f"{K} kg/ha",
                      f"{temperature} °C", f"{humidity} %", f"{ph}",
                      f"{rainfall} mm"]
        }
        st.table(summary_data)

    except FileNotFoundError:
        st.error(
            "⚠️ Model not found! Please run `python train_model.py` first "
            "to train and save the model."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🌱 Crop Recommendation System | Built with Python, Scikit-learn & Streamlit | "
    "Dataset: Kaggle Crop Recommendation Dataset"
)
