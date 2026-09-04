import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

# ---- Custom styling ----
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #1e2130;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 Brain Tumor MRI Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload or drag & drop a brain MRI scan for instant AI-powered classification</div>", unsafe_allow_html=True)

# ---- Sidebar info ----
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This tool classifies brain MRI scans into 4 categories using a deep learning model.")
    st.markdown("**Classes:**")
    st.markdown("- 🔴 Glioma\n- 🟠 Meningioma\n- 🟢 No Tumor\n- 🔵 Pituitary")
    st.divider()
    st.caption("⚠️ For educational/research purposes only. Not a substitute for professional medical diagnosis.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_classifier.h5")

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_icons = {'glioma': '🔴', 'meningioma': '🟠', 'notumor': '🟢', 'pituitary': '🔵'}

if "img" not in st.session_state:
    st.session_state.img = None

uploaded_file = st.file_uploader(
    "Choose an MRI image (or drag & drop it here)...",
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

if uploaded_file is not None:
    st.session_state.img = Image.open(uploaded_file).convert("RGB")

col_clear1, col_clear2 = st.columns([1, 5])
with col_clear1:
    if st.button("🔄 Clear"):
        st.session_state.img = None
        st.rerun()

st.divider()

if st.session_state.img is not None:
    img = st.session_state.img
    left_col, right_col = st.columns(2)

    with left_col:
        st.image(img, caption="Uploaded MRI Scan", use_container_width=True)
        predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")

    with right_col:
        if predict_clicked:
            with st.spinner("🧠 Analyzing scan..."):
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction))

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            icon = class_icons[predicted_class]
            st.markdown(f"### {icon} Prediction: **{predicted_class.upper()}**")

            st.metric(label="Confidence", value=f"{confidence:.2%}")
            st.progress(confidence)

            if predicted_class == "notumor":
                st.success("No tumor detected in this scan.")
            else:
                st.warning(f"Signs consistent with **{predicted_class}** detected.")

            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📊 Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": [c.capitalize() for c in class_names],
                "Probability": [prediction[0][i] for i in range(len(class_names))]
            }).set_index("Class")
            st.bar_chart(prob_df)

            if confidence > 0.9:
                st.balloons()
        else:
            st.info("Click **Predict** on the left to see the result here.")
else:
    st.info("👆 Please upload or drag & drop an MRI image to get started.")
