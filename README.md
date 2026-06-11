# README.md

## NeuroScan AI

Intelligent brain tumor detection and classification system built with TensorFlow/Keras and Streamlit.

### Features
- Load MRI scans (PNG/JPG) and preprocess them.
- CNN model (trained from scratch) classifies into **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**.
- Grad‑CAM visual explanations.
- Streamlit dashboard with dark‑mode glass‑morphism UI.
- SQLite history of predictions.
- PDF report generation with download button (ReportLab + Streamlit `st.download_button`).

### Setup
```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (first time only)
python model_training.py
```

### Run the app
```bash
streamlit run app/streamlit_app.py
```

### Folder Structure
```
NeuroScan AI/
│
├─ dataset/                # MRI images (already provided)
├─ model/                  # Trained model (brain_tumor_model.h5)
├─ utils/                  # Helper modules
│   ├─ __init__.py
│   ├─ preprocessing.py
│   ├─ prediction.py
│   ├─ gradcam.py
│   ├─ report_generator.py
│   └─ database.py
├─ app/                    # Streamlit app
│   └─ streamlit_app.py
├─ reports/                # Generated PDF reports
├─ database/               # SQLite DB (history.db)
├─ model_training.py       # Script to train the CNN
├─ requirements.txt
├─ .gitignore
└─ README.md
```
