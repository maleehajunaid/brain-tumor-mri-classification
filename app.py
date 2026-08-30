import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🧠 Brain Tumor MRI Classifier")

st.markdown(
    """
    **MobileNetV2 Transfer Learning + Grad-CAM Explainability**

    Upload a brain MRI image to get the predicted tumor class,
    confidence score, and Grad-CAM visualization.
    """
)

# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_PATH = "brain_tumor_model.h5"

# IMPORTANT:
# This is the common 4-class ordering used in the project.
# If your original training notebook used a different order,
# change the order here.

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMG_SIZE = (224, 224)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_model()

if model is None:

    st.error(
        "❌ Model file not found. "
        "Please upload 'brain_tumor_model.h5' to the Hugging Face Space."
    )

    st.stop()

# =========================================================
# FIND LAST CONVOLUTIONAL LAYER
# =========================================================

def find_last_conv_layer(model):

    # First search top-level layers
    for layer in reversed(model.layers):

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D
            )
        ):
            return layer.name

    # Search inside nested models
    for layer in reversed(model.layers):

        if hasattr(layer, "layers"):

            for sublayer in reversed(layer.layers):

                if isinstance(
                    sublayer,
                    (
                        tf.keras.layers.Conv2D,
                        tf.keras.layers.DepthwiseConv2D,
                        tf.keras.layers.SeparableConv2D
                    )
                ):
                    return sublayer.name

    return None


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index
):

    try:

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(
                    last_conv_layer_name
                ).output,
                model.output
            ]
        )

    except Exception:

        return None

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_heatmap = tf.reduce_max(
        heatmap
    )

    if max_heatmap > 0:

        heatmap /= max_heatmap

    return heatmap.numpy()


# =========================================================
# CREATE GRAD-CAM IMAGE
# =========================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    original = np.array(
        original_image
    )

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original_bgr.shape[1],
            original_bgr.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay


# =========================================================
# UPLOAD MRI
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded MRI")

    st.image(
        image,
        caption="Input MRI",
        use_container_width=True
    )

    # =====================================================
    # PREPROCESSING
    # =====================================================

    resized_image = image.resize(
        IMG_SIZE
    )

    img_array = np.array(
        resized_image
    ).astype(
        np.float32
    )

    # MobileNetV2 preprocessing
    img_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(img_array)
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    with st.spinner("🔍 Analyzing MRI..."):

        predictions = model.predict(
            img_array,
            verbose=0
        )

    predictions = np.asarray(
        predictions
    )

    # =====================================================
    # CONVERT OUTPUT TO PROBABILITIES
    # =====================================================

    if predictions.shape[-1] != len(CLASS_NAMES):

        st.error(
            f"Model returned {predictions.shape[-1]} outputs, "
            f"but {len(CLASS_NAMES)} classes were expected."
        )

        st.stop()

    # If model output is logits
    if (
        np.min(predictions) < 0
        or np.max(predictions) > 1
        or not np.isclose(
            np.sum(predictions[0]),
            1.0,
            atol=0.01
        )
    ):

        probabilities = tf.nn.softmax(
            predictions[0]
        ).numpy()

    else:

        probabilities = predictions[0]

    # =====================================================
    # GET PREDICTION
    # =====================================================

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        probabilities[predicted_index] * 100
    )

    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader("🎯 Prediction")

    st.success(
        f"### {predicted_class}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.subheader("📊 Class Probabilities")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        percentage = (
            probability * 100
        )

        st.write(
            f"**{class_name}: {percentage:.2f}%**"
        )

        st.progress(
            float(probability)
        )

    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.divider()

    st.subheader(
        "🔥 Grad-CAM Explainability"
    )

    last_conv_layer = find_last_conv_layer(
        model
    )

    if last_conv_layer is None:

        st.warning(
            "Could not automatically find a convolutional layer "
            "for Grad-CAM."
        )

    else:

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            last_conv_layer,
            predicted_index
        )

        if heatmap is not None:

            gradcam_image = create_gradcam_overlay(
                image,
                heatmap
            )

            st.image(
                gradcam_image,
                caption=(
                    "Grad-CAM — regions contributing "
                    "to the model prediction"
                ),
                use_container_width=True
            )

            st.info(
                "Brighter regions indicate areas that contributed "
                "more strongly to the model's prediction."
            )

        else:

            st.warning(
                "Prediction was successful, but Grad-CAM "
                "could not be generated for this model."
            )

# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application is for educational and research purposes only. "
    "It is not intended to replace professional medical diagnosis."
)
