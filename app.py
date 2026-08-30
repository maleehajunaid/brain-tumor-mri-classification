import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan to classify it as glioma, meningioma, pituitary tumor, or no tumor.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_classifier.h5")

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")
