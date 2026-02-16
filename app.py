import streamlit as st
import os
import tensorflow as tf
from inference import inference_generation
import numpy as np
from PIL import Image

# Configuration
STATIC_DIR = "./static"
CHECKPOINT_DIR = "./models"

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

st.set_page_config(page_title="Pix2Pix Image Translator", layout="wide")

st.title("## Pix2Pix Image-to-Image Translator")
st.write("Upload a side-by-side image (from the facades dataset) to generate a predicted facade ...")

# Initialize Session State
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def reset_session():
    st.session_state.generated = False
    st.session_state.uploaded_file_path = None
    st.session_state.prediction = None
    # Optional: Clear static folder
    for f in os.listdir(STATIC_DIR):
        os.remove(os.path.join(STATIC_DIR, f))

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("### Reset App"):
        reset_session()
        st.rerun()

# Main UI
if not st.session_state.generated:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_path = os.path.join(STATIC_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = file_path
        
        st.image(file_path, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Generate Prediction"):
            with st.spinner("Generating... this may take a few seconds."):
                try:
                    # Run inference
                    # inference_generation returns (input, target, prediction) normalized to [-1, 1]
                    inp, tar, pred = inference_generation(file_path, CHECKPOINT_DIR)
                    
                    # Convert to [0, 255] uint8 for display
                    st.session_state.prediction = (pred.numpy() * 0.5 + 0.5) * 255
                    st.session_state.prediction = st.session_state.prediction.astype(np.uint8)
                    st.session_state.generated = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during generation: {e}")

else:
    # Display Results
    st.success("Generation Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(st.session_state.uploaded_file_path, use_container_width=True)
        
    with col2:
        st.subheader("Predicted Output")
        st.image(st.session_state.prediction, use_container_width=True)

    if st.button("New Test"):
        reset_session()
        st.rerun()
