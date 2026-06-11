# model_training.py
"""Train the CNN model for brain tumor classification.
The script expects the MRI dataset in `dataset/` following the structure:
    dataset/<class_name>/image.jpg
It uses the preprocessing utilities to load data, builds a simple CNN,
trains with early stopping, and saves the trained model to `model/brain_tumor_model.h5`.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Ensure project root is in PYTHONPATH for utils imports
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
# The training script resides in the project root, so use CURRENT_DIR directly.
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.append(PROJECT_ROOT)

from utils.preprocessing import load_data

# ------------------------------------------------------------
# Configuration
# Determine dataset path (project folder, fallback to Desktop)
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
if not os.path.isdir(DATASET_PATH):
    # Create the expected subfolders if they don't exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    for sub in ["glioma", "meningioma", "pituitary", "no_tumor"]:
        os.makedirs(os.path.join(DATASET_PATH, sub), exist_ok=True)
    print(f"Created dataset directory with subfolders at {DATASET_PATH}. Populate it with images.")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.h5")
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (224, 224, 3)
# Train/val/test split ratios (70/15/15)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# ------------------------------------------------------------

def build_cnn(input_shape=IMG_SIZE, num_classes=4):
    """Return a Keras Sequential model matching the spec.
    Architecture: 3 Conv‑MaxPool blocks → Flatten → Dense‑512 → Dropout → Dense‑128 → Softmax.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def main():
    # Dataset directory existence is ensured above; no need for explicit check.
    print(f"Dataset path resolved to: {DATASET_PATH}")
    X, y = load_data(DATASET_PATH)
    # One‑hot encode labels
    num_classes = len(np.unique(y))
    y_cat = tf.keras.utils.to_categorical(y, num_classes)

    # Split data
    total = X.shape[0]
    indices = np.random.permutation(total)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    X_train, y_train = X[indices[:train_end]], y_cat[indices[:train_end]]
    X_val, y_val = X[indices[train_end:val_end]], y_cat[indices[train_end:val_end]]
    X_test, y_test = X[indices[val_end:]], y_cat[indices[val_end:]]

    model = build_cnn(input_shape=IMG_SIZE, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print("Starting training…")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2,
    )

    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc * 100:.2f}% (loss: {loss:.4f})")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
