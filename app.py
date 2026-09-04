import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from streamlit_paste_button import paste_image_button as pbutton

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload, drag & drop, ya paste (Ctrl+V) karke brain MRI scan classify karein: glioma, meningioma, pituitary tumor, ya no tumor.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_classifier.h5")

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

if "img" not in st.session_state:
    st.session_state.img = None

# --- Input options ---
tab1, tab2 = st.tabs(["📁 Upload / Drag & Drop", "📋 Paste Image"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an MRI image (drag & drop bhi kar sakte hain)...",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    if uploaded_file is not None:
        st.session_state.img = Image.open(uploaded_file).convert("RGB")

with tab2:
    paste_result = pbutton(
        label="📋 Clipboard se Paste karein",
        key="paste_button"
    )
    if paste_result.image_data is not None:
        st.session_state.img = paste_result.image_data.convert("RGB")

col_clear1, col_clear2 = st.columns([1, 5])
with col_clear1:
    if st.button("🔄 Clear"):
        st.session_state.img = None
        st.rerun()

st.divider()

# --- Side by side layout: image left, prediction right ---
if st.session_state.img is not None:
    img = st.session_state.img
    left_col, right_col = st.columns(2)

    with left_col:
        st.image(img, caption="Uploaded MRI Scan", use_container_width=True)
        predict_clicked = st.button("🔍 Predict", use_container_width=True)

    with right_col:
        if predict_clicked:
            with st.spinner("Analyzing scan..."):
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction))

            st.success(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.subheader("All class probabilities:")
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {prediction[0][i]:.2%}")
        else:
            st.info("Left side pe **Predict** click karein result dekhne ke liye.")
else:
    st.info("Upload, drag & drop, ya paste karke shuru karein.")
