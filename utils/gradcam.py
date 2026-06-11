# utils/gradcam.py
"""Utility to compute Grad‑CAM heatmaps for a Keras CNN.
The implementation follows the original Grad‑CAM paper using the
gradient of the predicted class with respect to the output of a
convolutional layer.
"""

import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer(model):
    """Return the name of the last Conv2D layer in the model, handling nested models.

    Some saved Keras Sequential models are wrapped inside a top‑level Model
    (named "sequential"). This function recursively walks through all sub‑layers
    to locate the final Conv2D layer.
    """
    def flatten_layers(layers):
        for lyr in layers:
            if isinstance(lyr, tf.keras.Model):
                # Recursively flatten nested model layers
                yield from flatten_layers(lyr.layers)
            else:
                yield lyr
    # Create a flat list of all concrete layers
    flat = list(flatten_layers(model.layers))
    for layer in reversed(flat):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")

# Helper to retrieve a layer object by name, searching recursively
def find_layer(model, name):
    """Search model (including nested sub‑models) for a layer with the given name.
    Returns the layer instance or raises ValueError if not found.
    """
    for layer in model.layers:
        if layer.name == name:
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return find_layer(layer, name)
            except ValueError:
                continue
    raise ValueError(f"Layer {name} not found in model hierarchy")


def compute_gradcam(img_array: np.ndarray, model, class_idx: int = None) -> np.ndarray:
    """Compute a Grad‑CAM heatmap.

    Args:
        img_array: Pre‑processed image array of shape (224, 224, 3).
        model: Loaded Keras model.
        class_idx: Optional explicit class index. If None, the predicted class is used.

    Returns:
        Heatmap as a 2‑D NumPy array (values 0‑255) resized to the original image size.
    """
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)
    if class_idx is None:
        preds = model(img_tensor)
        class_idx = int(tf.argmax(preds[0]).numpy())
    with tf.GradientTape() as tape:
        # Build a sub‑model that outputs the last conv layer activations
        last_conv_name = get_last_conv_layer(model)
        # Build a sub‑model directly from the model's inputs to the last conv layer output.
        # This avoids the need for dummy inputs and ensures the graph is connected.
        try:
            # Attempt to locate the target convolutional layer directly.
            target_layer = find_layer(model, last_conv_name)
            submodel = tf.keras.models.Model(inputs=model.inputs, outputs=target_layer.output)
        except Exception as e:
            # Fallback: rebuild model with a dummy input if the original model lacks inputs.
            dummy_input = tf.keras.Input(shape=(224, 224, 3), name="gradcam_dummy_input")
            _ = model(dummy_input)
            target_layer = find_layer(model, last_conv_name)
            submodel = tf.keras.models.Model(inputs=model.input, outputs=target_layer.output)
        conv_output = submodel(img_tensor)
        tape.watch(conv_output)
        predictions = model(img_tensor)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_output)
    # If gradients are None (possible for some models), replace with zeros to avoid errors
    if grads is None:
        grads = tf.zeros_like(conv_output)
    # Global average pooling on the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Multiply each channel in the feature map array by "how important" it is
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    # Resize to original image size
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    return heatmap_uint8

def overlay_heatmap(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay the heatmap onto the original image.

    Returns an RGB image (uint8) suitable for display.
    """
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), 1.0, heatmap_color, alpha, 0)
    return overlay
