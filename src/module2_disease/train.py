"""
Module 2 — Disease Detection: MobileNetV2 Transfer Learning Training.

Two-phase training:
  Phase 1 — Train only the new classifier head (10 epochs).
  Phase 2 — Unfreeze the last 30 layers and fine-tune (10 epochs, lower LR).

Supports SageMaker Training entry-point format.

Usage::

    python -m src.module2_disease.train [--data-dir PATH] [--epochs 20]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    HYPERPARAMS,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.utils.common import setup_logging, Timer

logger = setup_logging("module2_disease.train")

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 10
PHASE1_LR = 1e-4
PHASE2_LR = 1e-5
DROPOUT = 0.3
FINE_TUNE_AT = -30  # unfreeze last 30 layers


# ──────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────


def build_model(num_classes: int, image_size: tuple = IMAGE_SIZE):
    """
    Build MobileNetV2 with a custom classification head.

    Returns compiled model with base layers frozen.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*image_size, 3),
    )
    base_model.trainable = False  # Phase 1: freeze all base layers

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(
        "Model built — %d total params, %d trainable",
        model.count_params(),
        sum(int(w.numpy().size) for w in model.trainable_weights),
    )
    return model, base_model


# ──────────────────────────────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────────────────────────────


def train(
    data_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    phase1_epochs: int = PHASE1_EPOCHS,
    phase2_epochs: int = PHASE2_EPOCHS,
) -> dict:
    """Run the two-phase training pipeline."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    # SageMaker overrides
    data_dir = data_dir or Path(
        os.getenv("SM_CHANNEL_TRAIN", str(DATA_PROCESSED / "module2_disease"))
    )
    model_dir = model_dir or Path(os.getenv("SM_MODEL_DIR", str(MODELS_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = FIGURES_DIR / "module2_disease"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Data generators ──────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        data_dir / "train", target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True,
        seed=RANDOM_SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        data_dir / "val", target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )

    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())
    logger.info("Classes (%d): %s", num_classes, class_names)

    # Save class names
    with open(model_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # ── Build model ──────────────────────────────────────────────────
    model, base_model = build_model(num_classes)

    callbacks_phase1 = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(
            str(model_dir / "disease_best_phase1.h5"),
            monitor="val_accuracy", save_best_only=True,
        ),
    ]

    # ── Phase 1: Train classifier head only ──────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — Training classifier head (%d epochs)", phase1_epochs)
    logger.info("=" * 60)

    with Timer("Phase 1 training"):
        history1 = model.fit(
            train_gen,
            epochs=phase1_epochs,
            validation_data=val_gen,
            callbacks=callbacks_phase1,
        )

    # ── Phase 2: Fine-tune last N layers ─────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — Fine-tuning last %d layers (%d epochs)", abs(FINE_TUNE_AT), phase2_epochs)
    logger.info("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8),
        ModelCheckpoint(
            str(model_dir / "disease_best_phase2.h5"),
            monitor="val_accuracy", save_best_only=True,
        ),
    ]

    with Timer("Phase 2 fine-tuning"):
        history2 = model.fit(
            train_gen,
            epochs=phase2_epochs,
            validation_data=val_gen,
            callbacks=callbacks_phase2,
        )

    # ── Save model ───────────────────────────────────────────────────
    model.save(str(model_dir / "disease_model_saved"), save_format="tf")
    model.save(str(model_dir / "disease_model.h5"))
    logger.info("Models saved → %s", model_dir)

    # ── Training curves ──────────────────────────────────────────────
    _plot_training_curves(history1, history2, fig_dir / "training_curves.png")

    # ── SageMaker metric logging ─────────────────────────────────────
    final_val_acc = history2.history["val_accuracy"][-1]
    final_val_loss = history2.history["val_loss"][-1]
    print(f"val_accuracy={final_val_acc:.4f};")  # SageMaker HPO objective metric
    print(f"val_loss={final_val_loss:.4f};")

    return {
        "val_accuracy": float(final_val_acc),
        "val_loss": float(final_val_loss),
        "num_classes": num_classes,
        "class_names": class_names,
    }


def _plot_training_curves(history1, history2, save_path: Path) -> None:
    """Plot combined training/validation accuracy and loss curves."""
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]
    phase1_len = len(history1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(acc, label="Train Accuracy")
    ax1.plot(val_acc, label="Val Accuracy")
    ax1.axvline(x=phase1_len - 0.5, color="red", linestyle="--", alpha=0.5, label="Fine-tune start")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training & Validation Accuracy")
    ax1.legend()

    ax2.plot(loss, label="Train Loss")
    ax2.plot(val_loss, label="Val Loss")
    ax2.axvline(x=phase1_len - 0.5, color="red", linestyle="--", alpha=0.5, label="Fine-tune start")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.legend()

    plt.suptitle("MobileNetV2 — Disease Detection Training Curves", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease detection CNN")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    args = parser.parse_args()
    train(data_dir=args.data_dir, model_dir=args.model_dir,
          phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs)


if __name__ == "__main__":
    main()
