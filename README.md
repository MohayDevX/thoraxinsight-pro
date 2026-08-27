# 🩻 ThoraxInsight Pro - AI-Powered Chest X-Ray Analysis Platform

> Clinical Decision Support System for Automated Chest Radiograph Interpretation with Explainable AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![AI Model](https://img.shields.io/badge/Model-DenseNet121-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

### 🔴 Live Demo: [Your Streamlit Link Will Be Here Soon]

---

## 📋 Overview
**ThoraxInsight Pro** is an end-to-end AI platform designed to assist radiologists in detecting thoracic pathologies from chest X-rays. 
In many hospitals, especially in low-resource areas, there is a shortage of expert radiologists and a high workload. This system provides an instant second opinion, localizes the pathology, and generates a professional PDF report.

**The Problem It Solves:**
*   Shortage of radiologists
*   Human error due to fatigue
*   No explainability in black-box AI (We solve this with Heatmaps)
*   Lack of fast preliminary reports

## ✨ Key Features

1.  **Multi-Format Support:** Accepts both `DICOM (.dcm)` and standard images `JPG/PNG`.
2.  **State-of-the-Art AI:** Uses `densenet121-res224-all` from `torchxrayvision`, trained on 3 major datasets: **NIH, CheXpert, and MIMIC**.
3.  **Explainable AI (XAI):** Generates **Grad-CAM Heatmaps** to show *where* the model is looking. Red/Yellow areas = High suspicion.
4.  **Comprehensive Analysis:** Detects 18 pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Pneumonia, Pneumothorax, etc.
5.  **Professional PDF Reporting:** One-click downloadable clinical report with patient data and findings.
6.  **Clinical UI:** Built with Streamlit for a clean, hospital-ready interface.

## 🧠 How It Works

The system takes X-Ray -> Preprocesses it -> Extracts features with DenseNet121 -> Classifies 18 diseases and generates a heatmap -> Creates final PDF report

## 🛠️ Tech Stack
*   **Frontend:** Streamlit
*   **AI / Deep Learning:** PyTorch, TorchXrayVision
*   **Image Processing:** Pydicom, OpenCV, Pillow
*   **Visualization:** Grad-CAM Implementation
*   **Reporting:** FPDF2

## 🚀 How to Run Locally

```bash
git clone https://github.com/MohayDevX/thoraxinsight-pro.git
cd thoraxinsight-pro
pip install -r requirements.txt
streamlit run app.py


⚠️ Disclaimer
For Research & Educational Purposes Only. This system is a Clinical Decision Support tool and does not replace a certified radiologist's diagnosis.

Developed with ❤️ by Mohamed H Younis (MohayDevX) | AI Engineer
