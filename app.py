import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import torch
import torchxrayvision as xrv
import cv2
from fpdf import FPDF
import datetime

# --- Page Config ---
# --- Premium Medical UI ---
st.markdown("""
<style>
    .stApp { background-color: #0a192f; }
    [data-testid="stSidebar"] { background-color: #112240; }
    h1, h2, h3 { color: #64ffda !important; }
    .stButton>button {
        background-color: #64ffda;
        color: #0a192f;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #0a192f;
    }
</style>
<div style="text-align:center; padding: 20px 0;">
    <h1 style="font-size: 42px; margin-bottom:0;">🫁 ThoraxInsight Pro</h1>
    <p style="color:#8892b0; font-size:18px;">AI-Powered Clinical Decision Support | V3.2 Premium</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ThoraxInsight Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Clinical Decision Support Platform for Chest Radiography | V3.2 Final - Stable</div>', unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

model = load_model()

# --- Sidebar ---
with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID", "P-2026-001")
    patient_age = st.number_input("Age", 1, 100, 35)
    patient_gender = st.selectbox("Gender", ["Male", "Female"])
    st.divider()
    st.info("Model: densenet121-res224-all\nTrained on NIH, CheXpert, MIMIC")
    st.warning("For Research & Educational Use Only. Not a final diagnosis.")

uploaded_file = st.file_uploader("Upload Chest X-Ray (DICOM / JPG / PNG)", type=["dcm", "jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.name.lower().endswith(".dcm"):
        ds = pydicom.dcmread(uploaded_file)
        orig_img = ds.pixel_array
    else:
        orig_img = np.array(Image.open(uploaded_file).convert("L"))

    img_resized_224 = np.array(Image.fromarray(orig_img).resize((224, 224)))
    img_norm = xrv.datasets.normalize(img_resized_224, 255)
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0) # 4D Fix

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Radiograph")
        st.image(orig_img, use_container_width=True, clamp=True)

    with col2:
        st.subheader("AI Analysis")
        if st.button("🔬 Analyze & Localize Pathology", type="primary", use_container_width=True):
            with st.spinner("Analyzing with ThoraxInsight AI..."):
                with torch.no_grad():
                    pred = model(img_tensor)
                probs = pred[0].numpy()
                pathologies = dict(zip(model.pathologies, probs))
                sorted_path = sorted(pathologies.items(), key=lambda x: x[1], reverse=True)
                top_disease, top_prob = sorted_path[0]
                top_idx = model.pathologies.index(top_disease)

                with torch.no_grad():
                    features = model.features(img_tensor)
                weights = model.classifier.weight[top_idx]
                cam = torch.einsum('c, b c h w -> b h w', weights, features).squeeze(0)
                cam = torch.relu(cam)
                cam = cam.detach().numpy()
                cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
                cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

                heatmap_colored = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                orig_bgr = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)

            st.success(f"Primary Finding: {top_disease} ({top_prob*100:.1f}%)")
            st.image(overlay, caption=f"Localization Heatmap - {top_disease}", use_container_width=True)

            st.write("**Full Probability Report:**")
            for name, p in sorted_path[:12]:
                st.progress(float(p), text=f"{name}: {p*100:.1f}%")

            # --- PDF FIXED ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "ThoraxInsight Pro - Radiology Report", ln=True, align='C')
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 10, f"Date: {datetime.date.today()} | Patient: {patient_id} | Age: {patient_age} | Gender: {patient_gender}", ln=True)
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Primary Finding: {top_disease} - {top_prob*100:.1f}%", ln=True)
            pdf.set_font("Arial", "", 11)
            for name, p in sorted_path:
                pdf.cell(0, 7, f"{name}: {p*100:.1f}%", ln=True)
            pdf.ln(5)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(0, 5, "Disclaimer: This is an AI-assisted preliminary analysis and not a final medical diagnosis. Must be reviewed by a certified radiologist.")

            # الحل النهائي لإيرور الـ PDF
            pdf_output = pdf.output(dest='S')
            if isinstance(pdf_output, str):
                pdf_bytes = pdf_output.encode('latin-1')
            else:
                pdf_bytes = bytes(pdf_output)

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Report_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    st.info("Awaiting X-Ray upload. System ready.")
