"""
Module 2 — Disease Detection: Model Evaluation.

Loads the trained CNN and test data, computes per-class precision/recall/F1,
generates a confusion matrix heatmap, identifies worst-performing classes,
and saves all metrics to JSON.

Usage::

    python -m src.module2_disease.evaluate [--model-path PATH] [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
)
from src.utils.common import setup_logging, Timer

logger = setup_logging("module2_disease.evaluate")


def evaluate(
    model_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline for disease detection CNN."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    data_dir = data_dir or DATA_PROCESSED / "module2_disease"
    model_path = model_path or MODELS_DIR / "disease_model.h5"
    output_dir = output_dir or FIGURES_DIR / "module2_disease"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    with Timer("Loading model"):
        model = tf.keras.models.load_model(str(model_path))

    # ── Load class names ─────────────────────────────────────────────
    class_names_path = MODELS_DIR / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            class_names = json.load(f)
    else:
        class_names = None

    # ── Test data generator ──────────────────────────────────────────
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        data_dir / "test",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    if class_names is None:
        class_names = list(test_gen.class_indices.keys())

    # ── Predict ──────────────────────────────────────────────────────
    with Timer("Generating predictions"):
        y_proba = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_proba, axis=1)
        y_true = test_gen.classes

    # ── Metrics ──────────────────────────────────────────────────────
    with Timer("Computing metrics"):
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        report_dict = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        )

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS — Disease Detection")
    logger.info("=" * 60)
    logger.info("Accuracy : %.4f", accuracy)
    logger.info("Macro F1 : %.4f", macro_f1)
    logger.info("-" * 60)
    print(report_text)

    # ── Identify worst-performing classes ─────────────────────────────
    per_class_f1 = {
        name: report_dict[name]["f1-score"]
        for name in class_names
        if name in report_dict
    }
    worst_classes = sorted(per_class_f1.items(), key=lambda x: x[1])[:5]
    logger.info("Worst performing classes:")
    for cls_name, f1_val in worst_classes:
        logger.info("  %-40s  F1=%.3f", cls_name, f1_val)

    # ── Confusion matrix heatmap ─────────────────────────────────────
    with Timer("Plotting confusion matrix"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Disease Detection — Confusion Matrix", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", cm_path)

    # ── Save metrics ─────────────────────────────────────────────────
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": report_dict,
        "worst_classes": worst_classes,
        "num_test_samples": int(len(y_true)),
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved → %s", metrics_path)

    # SageMaker metric logging
    print(f"test_accuracy={accuracy:.4f};")
    print(f"test_macro_f1={macro_f1:.4f};")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate disease detection model")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    evaluate(model_path=args.model_path, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
