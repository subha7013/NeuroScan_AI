# utils/report_generator.py
"""Generate a PDF report for a prediction using ReportLab.
The report includes patient ID, timestamp, the original MRI thumbnail,
the predicted class with confidence scores, and the Grad‑CAM heatmap.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def _draw_image(c: canvas.Canvas, img_path: str, x: int, y: int, width: int, height: int):
    if os.path.exists(img_path):
        c.drawImage(ImageReader(img_path), x, y, width=width, height=height, preserveAspectRatio=True, mask='auto')
    else:
        c.setFillColor(colors.red)
        c.rect(x, y, width, height, fill=1)
        c.setFillColor(colors.white)
        c.drawString(x + 5, y + height / 2, "Image not found")

def generate_report(
    patient_id: str,
    image_path: str,
    prediction_label: str,
    confidences: dict,
    gradcam_path: str,
    output_dir: str = "reports",
) -> str:
    """Create a PDF report and return the absolute path to the file.

    Args:
        patient_id: Identifier for the patient/scenario.
        image_path: Path to the original MRI image.
        prediction_label: Human‑readable class name.
        confidences: Mapping from class name to probability (0‑1).
        gradcam_path: Path to the Grad‑CAM overlay image.
        output_dir: Directory where the PDF will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{patient_id}_{timestamp}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 50, "NeuroScan AI Report")

    # Patient info
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Patient ID: {patient_id}")
    c.drawString(50, height - 120, f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Original image
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 150, "MRI Scan:")
    _draw_image(c, image_path, 50, height - 450, 200, 200)

    # Grad‑CAM image
    c.drawString(300, height - 150, "Grad‑CAM:")
    _draw_image(c, gradcam_path, 300, height - 450, 200, 200)

    # Prediction table
    c.drawString(50, height - 470, "Prediction Results:")
    start_y = height - 490
    c.setFont("Helvetica", 11)
    for i, (cls, prob) in enumerate(confidences.items()):
        text = f"{cls}: {prob * 100: .2f}%"
        if cls == prediction_label:
            c.setFillColor(colors.green)
        else:
            c.setFillColor(colors.black)
        c.drawString(60, start_y - i * 15, text)
        c.setFillColor(colors.black)

    c.save()
    return pdf_path
