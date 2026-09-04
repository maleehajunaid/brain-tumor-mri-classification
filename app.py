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
