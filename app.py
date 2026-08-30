import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan to classify it as glioma, meningioma, pituitary tumor, or no tumor.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_classifier.h5")

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Initialize session state so we can reset/reupload easily
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader(
    "Choose an MRI image...",
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)

with col2:
    clear_clicked = st.button("🔄 Clear / Upload New", use_container_width=True)

if clear_clicked:
    st.rerun()

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

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
        st.info("Click **Predict** to classify this scan.")
else:
    st.info("Please upload an MRI image to get started.")
