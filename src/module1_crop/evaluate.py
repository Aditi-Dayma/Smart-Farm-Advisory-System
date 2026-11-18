"""
Module 1 — Crop Recommendation: Model Evaluation.

Loads the held-out test set and the best trained model, computes
comprehensive classification metrics, generates confusion-matrix and
feature-importance visualisations, and persists everything to disk.

Usage::

    python -m src.module1_crop.evaluate [--model-path PATH] [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

from src.utils.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.utils.common import setup_logging, Timer

logger = setup_logging("module1_crop.evaluate")

# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────


def load_test_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load test features, labels, and class names."""
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    # Load label encoder for class names
    le = joblib.load(data_dir / "label_encoder.pkl")
    class_names = list(le.classes_)
    return X_test, y_test, class_names


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Top-3 accuracy (if probabilities available)
    if y_proba is not None and y_proba.shape[1] > 1:
        try:
            metrics["top_3_accuracy"] = float(
                top_k_accuracy_score(y_true, y_proba, k=3)
            )
        except Exception:
            metrics["top_3_accuracy"] = None

    # Per-class report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["per_class"] = report

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Crop Recommendation — Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    save_path: Path,
    top_n: int = 15,
) -> None:
    """Plot feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_; skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="forestgreen",
        edgecolor="darkgreen",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances — Crop Recommendation")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ──────────────────────────────────────────────────────────────────────


def evaluate(
    model_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline and return metrics dict."""
    data_dir = data_dir or DATA_PROCESSED / "module1_crop"
    model_path = model_path or MODELS_DIR / "crop_best_model.pkl"
    output_dir = output_dir or FIGURES_DIR / "module1_crop"
    output_dir.mkdir(parents=True, exist_ok=True)

    with Timer("Loading model and data"):
        model = joblib.load(model_path)
        X_test, y_test, class_names = load_test_data(data_dir)
        feature_names = list(X_test.columns)

    with Timer("Generating predictions"):
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)
            if hasattr(model, "predict_proba")
            else None
        )

    with Timer("Computing metrics"):
        metrics = compute_metrics(y_test.values, y_pred, y_proba, class_names)

    # ── Print summary ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS — Crop Recommendation")
    logger.info("=" * 60)
    logger.info("Accuracy        : %.4f", metrics["accuracy"])
    logger.info("Macro F1        : %.4f", metrics["macro_f1"])
    logger.info("Weighted F1     : %.4f", metrics["weighted_f1"])
    if metrics.get("top_3_accuracy"):
        logger.info("Top-3 Accuracy  : %.4f", metrics["top_3_accuracy"])
    logger.info("-" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # ── Visualisations ───────────────────────────────────────────────
    with Timer("Generating plots"):
        plot_confusion_matrix(
            y_test.values, y_pred, class_names,
            save_path=output_dir / "confusion_matrix.png",
        )
        plot_feature_importance(
            model, feature_names,
            save_path=output_dir / "feature_importance.png",
        )

    # ── Save metrics JSON ────────────────────────────────────────────
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics JSON saved → %s", metrics_path)

    return metrics


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crop recommendation model")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    evaluate(model_path=args.model_path, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
