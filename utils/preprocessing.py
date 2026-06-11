# utils/preprocessing.py
"""
Utility functions for loading and preprocessing MRI images.

- Images are read with OpenCV, converted to RGB, resized to 224x224.
- Pixel values are scaled to [0, 1].
- Labels are encoded as integers:
    Glioma      -> 0
    Meningioma  -> 1
    Pituitary   -> 2
    No Tumor    -> 3
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Mapping from folder name to label index
LABEL_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    # Dataset contains a folder named 'notumor' (without underscore)
    "no_tumor": 3,
    "notumor": 3,
}

# ------------------------------------------------------------------
def load_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """Load a single image and apply the same preprocessing used for training.

    Parameters
    ----------
    image_path: str
        Path to the image file.
    img_size: int, optional
        Desired width and height after resizing (default 224).

    Returns
    -------
    np.ndarray
        A ``(img_size, img_size, 3)`` array of type ``float32`` ready for model inference.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return img


def load_images_and_labels(dataset_dir: str, img_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Load all images from the dataset directory.

    Args:
        dataset_dir: Root path containing sub‑folders for each class.
        img_size: Desired width and height after resizing (square).

    Returns:
        X: NumPy array of shape (N, img_size, img_size, 3) with dtype float32.
        y: NumPy array of shape (N,) with integer class indices.
    """
    images = []
    labels = []
    for class_name, label_idx in LABEL_MAP.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            # Accept common image extensions
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            
            # Robust reading
            img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize
            img = cv2.resize(img, (img_size, img_size))
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label_idx)
    # If no images were loaded from class subfolders, attempt to load any images directly in the dataset directory.
    if not images:
        for fname in os.listdir(dataset_dir):
            fpath = os.path.join(dataset_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            # Assign a default label (e.g., 0) for unlabeled images
            labels.append(0)
    if not images:
        raise FileNotFoundError(
            f"No images found in dataset directory {dataset_dir}. "
            "Ensure images are placed in class subfolders (glioma, meningioma, pituitary, no_tumor) "
            "or directly in the dataset folder."
        )
    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y

def load_data(dataset_dir: str, img_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper that forwards to load_images_and_labels.
    Returns the same (X, y) as load_images_and_labels.
    """
    return load_images_and_labels(dataset_dir, img_size)

def get_data_splits(dataset_dir: str, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """Split the dataset into train / validation / test sets.

    The function first loads all data, then performs a stratified split.
    """
    X, y = load_images_and_labels(dataset_dir)
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Remaining part is further split into train/val
    val_fraction = val_size / (1 - test_size)  # proportion of the remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
