# app/streamlit_app.py
"""Streamlit dashboard for NeuroScan AI.
Features:
- Dark‑mode glass‑morphism UI with custom CSS and Google Font Inter.
- Sidebar for file upload and patient ID input.
- Displays uploaded MRI, prediction bar chart, Grad‑CAM overlay.
- Generates PDF report via utils.report_generator.
- Stores prediction history in SQLite (utils.database).
"""

import os
import io
import base64
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

# Ensure project root is on PYTHONPATH
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.append(PROJECT_ROOT)

from utils.preprocessing import load_image
from utils.prediction import load_trained_model, predict_image
from utils.gradcam import compute_gradcam, overlay_heatmap
from utils.report_generator import generate_report
from utils.database import insert_prediction, fetch_history

# --------------------- UI STYLE ---------------------
st.set_page_config(page_title="NeuroScan AI", layout="wide")

# Custom CSS for dark glass‑morphism
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0d0d0d, #1a1a2e);
        color: #f0f0f0;
    }
    .glass {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
        color: hsl(210,70%,65%);
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background: rgba(0,0,0,0.4);
        backdrop-filter: blur(8px);
    }
    .stButton>button {
        background-color: hsl(210,70%,55%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: hsl(210,70%,65%);
        transform: translateY(-2px);
    }
    .metric {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("<div class='title'>NeuroScan AI – Brain Tumor Detection</div>", unsafe_allow_html=True)

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Choose an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    patient_id = st.text_input("Patient ID", value="PATIENT_001")
    if st.button("Run Prediction") and uploaded_file:
        # Save uploaded file temporarily
        tmp_path = os.path.join("tmp", uploaded_file.name)
        os.makedirs("tmp", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['image_path'] = tmp_path
        st.session_state['patient_id'] = patient_id
        st.session_state['run'] = True

# --------------------- MAIN ---------------------
if st.session_state.get('run'):
    img_path = st.session_state['image_path']
    patient_id = st.session_state['patient_id']
    # Load and preprocess image
    img_array = load_image(img_path)
    # Load model (assumes model has been trained and saved)
    model_path = os.path.join(PROJECT_ROOT, "model", "brain_tumor_model.h5")
    try:
        model = load_trained_model(model_path)
    except FileNotFoundError:
        st.error("Trained model not found. Please run `model_training.py` first.")
        st.stop()
    # Prediction
    pred_idx, confidences = predict_image(img_array, model)
    pred_label = confidences.get(next(iter(confidences)))  # placeholder not needed
    pred_label = list(confidences.keys())[list(confidences.values()).index(max(confidences.values()))]
    # Grad‑CAM
    gradcam_heat = compute_gradcam(img_array, model, class_idx=None)
    overlay_img = overlay_heatmap(img_array, gradcam_heat)
    # Save overlay to temp file for display / PDF
    overlay_path = os.path.join("tmp", f"gradcam_{uploaded_file.name}")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    # Store prediction in DB
    insert_prediction(filename=uploaded_file.name, patient_id=patient_id, predicted_class=pred_label, confidence=max(confidences.values()))

    # ---- Display ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original MRI")
        st.image(img_path, use_column_width=True)
    with col2:
        st.subheader("Grad‑CAM Explanation")
        st.image(overlay_path, use_column_width=True)

    st.subheader("Prediction Results")
    # Bar chart of confidences
    sorted_labels = list(confidences.keys())
    sorted_probs = [confidences[l] for l in sorted_labels]
    df_conf = pd.DataFrame({"Confidence": sorted_probs}, index=sorted_labels)
    st.bar_chart(df_conf)
    st.markdown(f"**Predicted Class:** {pred_label} (Confidence: {max(confidences.values())*100:.2f}%)")

    # PDF generation button
    if st.button("Generate PDF Report"):
        pdf_path = generate_report(
            patient_id=patient_id,
            image_path=img_path,
            prediction_label=pred_label,
            confidences=confidences,
            gradcam_path=overlay_path,
            output_dir=os.path.join(PROJECT_ROOT, "reports"),
        )
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
        )

    # History table
    st.subheader("Prediction History")
    rows = fetch_history(limit=20)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["ID", "Timestamp", "Filename", "PatientID", "Class", "Confidence"])  # noqa: E501
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(df)
    else:
        st.info("No history yet.")
else:
    st.info("Upload an MRI scan and click **Run Prediction** to begin.")
