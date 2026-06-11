# utils/prediction.py
"""
Utility functions for loading the trained Keras model and performing inference on MRI images.
"""
import os
import numpy as np
from tensorflow.keras.models import load_model

# Mapping from label index to human‑readable class name
LABELS = {0: "Glioma", 1: "Meningioma", 2: "Pituitary", 3: "No Tumor"}

def load_trained_model(model_path: str):
    """Load a Keras model from the given path.

    Args:
        model_path: Path to the `.h5` model file.
    Returns:
        A compiled Keras model ready for inference.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    return model

def predict_image(img_array: np.ndarray, model) -> tuple[int, dict]:
    """Run prediction on a pre‑processed image array.

    Args:
        img_array: Numpy array of shape (224, 224, 3) already normalized.
        model: Loaded Keras model.
    Returns:
        pred_index: Integer class index with highest probability.
        confidences: Mapping from class name to probability (0‑1).
    """
    # Ensure batch dimension
    img_batch = np.expand_dims(img_array, axis=0)  # shape (1, h, w, 3)
    probs = model.predict(img_batch, verbose=0)[0]  # shape (4,)
    pred_index = int(np.argmax(probs))
    confidences = {LABELS[i]: float(probs[i]) for i in range(len(probs))}
    return pred_index, confidences

def get_prediction_label(pred_index: int) -> str:
    """Return the class name for a given index."""
    return LABELS.get(pred_index, "Unknown")
