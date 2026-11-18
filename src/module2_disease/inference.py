"""
Module 2 — Disease Detection: Inference / Prediction.

Loads the trained MobileNetV2 model and class-name mapping to predict
crop diseases from leaf images.  Includes Grad-CAM visualisation for
explainability.

Usage::

    python -m src.module2_disease.inference --image path/to/leaf.jpg
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.utils.config import MODELS_DIR
from src.utils.common import setup_logging

logger = setup_logging("module2_disease.inference")

IMAGE_SIZE = (224, 224)


# ──────────────────────────────────────────────────────────────────────
# Disease predictor
# ──────────────────────────────────────────────────────────────────────


class DiseasePredictor:
    """Plant disease classifier using a trained MobileNetV2 model."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        class_names_path: Optional[Path] = None,
    ):
        import tensorflow as tf

        model_path = model_path or MODELS_DIR / "disease_model.h5"
        class_names_path = class_names_path or MODELS_DIR / "class_names.json"

        self.model = tf.keras.models.load_model(str(model_path))
        with open(class_names_path) as f:
            self.class_names: List[str] = json.load(f)

        logger.info("DiseasePredictor loaded — %d classes", len(self.class_names))

    @staticmethod
    def _load_and_preprocess(
        image_input: Union[str, Path, bytes],
        target_size: tuple = IMAGE_SIZE,
    ) -> np.ndarray:
        """Load image from path or bytes and preprocess for model input."""
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        if isinstance(image_input, (str, Path)):
            img = load_img(str(image_input), target_size=target_size)
        elif isinstance(image_input, bytes):
            from PIL import Image
            img = Image.open(io.BytesIO(image_input)).resize(target_size)
        else:
            raise TypeError(f"Expected str, Path, or bytes; got {type(image_input)}")

        arr = img_to_array(img) / 255.0
        return np.expand_dims(arr, axis=0)

    @staticmethod
    def _parse_class_name(class_name: str) -> Dict[str, str]:
        """Parse PlantVillage class name into crop and disease."""
        # Format: "Crop___Disease" or "Crop___healthy"
        parts = class_name.replace("___", "__").split("__")
        if len(parts) >= 2:
            crop = parts[0].replace("_", " ")
            disease = parts[1].replace("_", " ")
        else:
            crop = class_name
            disease = "Unknown"

        if disease.lower() == "healthy":
            disease = "Healthy (No Disease)"

        return {"crop": crop, "disease": disease}

    def predict(
        self,
        image_input: Union[str, Path, bytes],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict disease from a leaf image.

        Parameters
        ----------
        image_input : str, Path, or bytes
            Path to image file or raw image bytes.
        top_n : int
            Number of top predictions to return.

        Returns
        -------
        dict with keys: crop, disease, confidence, all_predictions.
        """
        X = self._load_and_preprocess(image_input)
        probas = self.model.predict(X, verbose=0)[0]

        # Sort by probability
        top_indices = np.argsort(probas)[::-1][:top_n]

        all_preds = []
        for idx in top_indices:
            parsed = self._parse_class_name(self.class_names[idx])
            all_preds.append({
                "class_name": self.class_names[idx],
                "crop": parsed["crop"],
                "disease": parsed["disease"],
                "confidence": round(float(probas[idx]), 4),
            })

        top1 = all_preds[0]
        return {
            "crop": top1["crop"],
            "disease": top1["disease"],
            "confidence": top1["confidence"],
            "class_name": top1["class_name"],
            "all_predictions": all_preds,
        }

    def grad_cam(
        self,
        image_input: Union[str, Path, bytes],
        save_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate Grad-CAM heatmap to highlight disease-relevant regions.

        Returns the heatmap array and optionally saves the visualisation.
        """
        import tensorflow as tf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X = self._load_and_preprocess(image_input)

        # Find the last convolutional layer
        last_conv = None
        for layer in reversed(self.model.layers):
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                last_conv = layer
                break

        if last_conv is None:
            # Try to find conv layer in the base model
            for layer in reversed(self.model.layers):
                if hasattr(layer, "layers"):
                    for sub_layer in reversed(layer.layers):
                        if hasattr(sub_layer, "output") and len(sub_layer.output.shape) == 4:
                            last_conv = sub_layer
                            break
                    if last_conv:
                        break

        if last_conv is None:
            logger.warning("Could not find convolutional layer for Grad-CAM.")
            return None

        grad_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[last_conv.output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(X)
            top_class = tf.argmax(predictions[0])
            loss = predictions[:, top_class]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        if save_path:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            import cv2

            if isinstance(image_input, (str, Path)):
                orig = img_to_array(load_img(str(image_input), target_size=IMAGE_SIZE))
            else:
                from PIL import Image
                orig = np.array(Image.open(io.BytesIO(image_input)).resize(IMAGE_SIZE))

            heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
            heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3] * 255
            superimposed = (orig * 0.6 + heatmap_color * 0.4).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(orig.astype(np.uint8))
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")
            axes[2].imshow(superimposed)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
            plt.suptitle("Grad-CAM — Disease Region Visualisation", fontsize=12)
            plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info("Grad-CAM saved → %s", save_path)

        return heatmap


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────

_predictor: Optional[DiseasePredictor] = None


def predict_disease(
    image_input: Union[str, Path, bytes],
) -> Dict[str, Any]:
    """Quick convenience function — loads model on first call."""
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor.predict(image_input)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Disease Detection Inference")
    parser.add_argument("--image", type=Path, required=True, help="Path to leaf image")
    parser.add_argument("--grad-cam", type=Path, default=None, help="Save Grad-CAM to this path")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    predictor = DiseasePredictor()
    result = predictor.predict(args.image, top_n=args.top_n)

    print(f"\n🔬 Disease Detection Results")
    print("=" * 50)
    print(f"  Crop     : {result['crop']}")
    print(f"  Disease  : {result['disease']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print("-" * 50)
    print("  Top predictions:")
    for p in result["all_predictions"]:
        print(f"    {p['crop']:12s} — {p['disease']:30s}  {p['confidence']*100:5.1f}%")
    print("=" * 50)

    if args.grad_cam:
        predictor.grad_cam(args.image, save_path=args.grad_cam)


if __name__ == "__main__":
    main()
