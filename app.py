import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

# ---- Custom CSS ----
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .header-container {
        background: linear-gradient(135deg, #4c6ef5 0%, #7048e8 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-title {
        color: white;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
    }
    .card {
        background-color: #1a1d29;
        border: 1px solid #2d3142;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6e6e6;
        margin-bottom: 0.8rem;
    }
    .result-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .badge-notumor { background-color: #1c3d2e; color: #4caf50; }
    .badge-tumor { background-color: #3d1c1c; color: #ff6b6b; }
    .confidence-text {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .class-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #2d3142;
        color: #cfcfcf;
    }
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #1a1d29;
        border: 2px dashed #4c6ef5;
        border-radius: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
    <div class="header-container">
        <div class="header-title">🧠 Brain Tumor MRI Classifier</div>
        <div class="header-subtitle">Upload or drag & drop an MRI scan to detect glioma, meningioma, pituitary tumor, or confirm no tumor</div>
    </div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This tool classifies brain MRI scans into 4 categories using a deep learning model trained with transfer learning (MobileNetV2).")
    st.markdown("**Classes:**")
    st.markdown("🔴 Glioma &nbsp;&nbsp; 🟠 Meningioma")
    st.markdown("🟢 No Tumor &nbsp;&nbsp; 🔵 Pituitary")
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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader(
    "Choose an MRI image (or drag & drop it here)...",
    type=["jpg", "jpeg", "png"],
    key=f"file_uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    st.session_state.img = Image.open(uploaded_file).convert("RGB")

col_clear1, col_clear2 = st.columns([1, 5])
with col_clear1:
    if st.button("🔄 Clear"):
        st.session_state.img = None
        st.session_state.uploader_key += 1
        st.rerun()

st.write("")

if st.session_state.img is not None:
    img = st.session_state.img
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<div class="card"><div class="card-title">📷 Uploaded Scan</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        if predict_clicked:
            with st.spinner("🧠 Analyzing scan..."):
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction))

            icon = class_icons[predicted_class]
            badge_class = "badge-notumor" if predicted_class == "notumor" else "badge-tumor"

            st.markdown(f'''
                <div class="card">
                    <div class="card-title">🎯 Result</div>
                    <span class="result-badge {badge_class}">{icon} {predicted_class.upper()}</span>
                    <div class="confidence-text">{confidence:.2%}</div>
                    <p style="color:#8a8f9c;">confidence</p>
                </div>
            ''', unsafe_allow_html=True)
            st.progress(confidence)

            rows_html = ""
            for i, class_name in enumerate(class_names):
                rows_html += f'<div class="class-row"><span>{class_icons[class_name]} {class_name.capitalize()}</span><span>{prediction[0][i]:.2%}</span></div>'

            st.markdown(f'''
                <div class="card">
                    <div class="card-title">📊 Class Probabilities</div>
                    {rows_html}
                </div>
            ''', unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Class": [c.capitalize() for c in class_names],
                "Probability": [prediction[0][i] for i in range(len(class_names))]
            }).set_index("Class")
            st.bar_chart(prob_df)
        else:
            st.markdown('''
                <div class="card" style="text-align:center; padding: 3rem 1.5rem;">
                    <div style="font-size:2.5rem;">👈</div>
                    <p style="color:#8a8f9c;">Click <b>Predict</b> on the left to see the result here.</p>
                </div>
            ''', unsafe_allow_html=True)
else:
    st.markdown('''
        <div class="card" style="text-align:center; padding: 3rem 1.5rem;">
            <div style="font-size:2.5rem;">👆</div>
            <p style="color:#8a8f9c;">Please upload or drag & drop an MRI image to get started.</p>
        </div>
    ''', unsafe_allow_html=True)
