"""
Module 2 — Disease Detection: Image Preprocessing Pipeline.

Organises PlantVillage images into train/val/test splits, resizes,
normalises, applies augmentation, and creates Keras data generators.
Supports running as a SageMaker Processing job.

Usage::

    python -m src.module2_disease.preprocess [--input-dir PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import (
    DATA_RAW,
    DATA_PROCESSED,
    FIGURES_DIR,
    HYPERPARAMS,
    RANDOM_SEED,
)
from src.utils.common import setup_logging, Timer

logger = setup_logging("module2_disease.preprocess")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Crops and their PlantVillage class folder prefixes
TARGET_CROPS: Dict[str, str] = {
    "Tomato": "Tomato",
    "Potato": "Potato",
    "Corn": "Corn",
    "Grape": "Grape",
    "Apple": "Apple",
}

IMAGE_SIZE: Tuple[int, int] = (224, 224)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
BATCH_SIZE: int = 32


# ──────────────────────────────────────────────────────────────────────
# Split and organise images
# ──────────────────────────────────────────────────────────────────────


def discover_classes(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover PlantVillage class directories matching target crops.

    Returns mapping of class_name → list of image Paths.
    """
    class_map: Dict[str, List[Path]] = {}
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return class_map

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        # Match if the folder name starts with any target crop prefix
        for crop, prefix in TARGET_CROPS.items():
            if class_dir.name.startswith(prefix):
                images = [
                    p for p in class_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                ]
                if images:
                    class_map[class_dir.name] = images
                    logger.info("  Found class: %-45s — %d images", class_dir.name, len(images))
                break

    logger.info("Discovered %d classes across %d target crops", len(class_map), len(TARGET_CROPS))
    return class_map


def split_dataset(
    class_map: Dict[str, List[Path]],
    output_dir: Path,
    seed: int = RANDOM_SEED,
) -> Dict[str, int]:
    """Split images into train/val/test directories."""
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "test": 0}

    for class_name, image_paths in class_map.items():
        rng.shuffle(image_paths)
        n = len(image_paths)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": image_paths[:n_train],
            "val": image_paths[n_train:n_train + n_val],
            "test": image_paths[n_train + n_val:],
        }

        for split_name, paths in splits.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                shutil.copy2(src, dest_dir / src.name)
            counts[split_name] += len(paths)

    logger.info("Split counts — train: %d, val: %d, test: %d", counts["train"], counts["val"], counts["test"])
    return counts


# ──────────────────────────────────────────────────────────────────────
# Data generators (Keras)
# ──────────────────────────────────────────────────────────────────────


def create_data_generators(
    data_dir: Path,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Create Keras ImageDataGenerators with augmentation for training.

    Returns dict with keys 'train_gen', 'val_gen', 'test_gen', 'class_names'.
    """
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        logger.warning("TensorFlow not installed — returning None generators.")
        return {"train_gen": None, "val_gen": None, "test_gen": None, "class_names": []}

    # Training generator WITH augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    # Validation & test generators WITHOUT augmentation
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        data_dir / "train",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        data_dir / "val",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        data_dir / "test",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_gen.class_indices.keys())
    return {
        "train_gen": train_gen,
        "val_gen": val_gen,
        "test_gen": test_gen,
        "class_names": class_names,
    }


# ──────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────


def plot_class_distribution(
    class_map: Dict[str, List[Path]],
    save_path: Path,
) -> None:
    """Bar chart of images per class."""
    names = list(class_map.keys())
    counts = [len(v) for v in class_map.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, counts, color="mediumseagreen", edgecolor="darkgreen")
    ax.set_xlabel("Number of Images")
    ax.set_title("PlantVillage — Class Distribution (5 Crops)")
    ax.bar_label(bars, padding=3, fontsize=7)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Class distribution plot saved → %s", save_path)


def plot_sample_augmented(
    data_dir: Path,
    save_path: Path,
    n_samples: int = 5,
) -> None:
    """Show original vs augmented image samples."""
    try:
        from tensorflow.keras.preprocessing.image import (
            ImageDataGenerator,
            load_img,
            img_to_array,
        )
    except ImportError:
        logger.warning("TensorFlow not available — skipping augmentation plot.")
        return

    train_dir = data_dir / "train"
    if not train_dir.exists():
        return

    # Pick first class, first image
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        return

    sample_dir = class_dirs[0]
    images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.JPG")) + list(sample_dir.glob("*.png"))
    if not images:
        return

    img_path = images[0]
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, 0)

    aug = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    fig, axes = plt.subplots(1, n_samples + 1, figsize=(3 * (n_samples + 1), 3))
    axes[0].imshow(img_arr[0])
    axes[0].set_title("Original")
    axes[0].axis("off")

    gen = aug.flow(img_arr, batch_size=1)
    for i in range(1, n_samples + 1):
        augmented = next(gen)[0]
        axes[i].imshow(np.clip(augmented, 0, 1))
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis("off")

    plt.suptitle(f"Augmentation Samples — {sample_dir.name}", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Augmentation samples saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────


def preprocess(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Run the full image preprocessing pipeline."""
    # SageMaker Processing environment overrides
    input_dir = input_dir or Path(os.getenv("SM_CHANNEL_INPUT", str(DATA_RAW / "plantvillage")))
    output_dir = output_dir or Path(os.getenv("SM_OUTPUT_DATA_DIR", str(DATA_PROCESSED / "module2_disease")))
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = FIGURES_DIR / "module2_disease"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with Timer("Discovering classes"):
        class_map = discover_classes(input_dir)

    if not class_map:
        logger.error("No classes found in %s — aborting.", input_dir)
        return {}

    # Save class names
    class_names = sorted(class_map.keys())
    class_names_path = output_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Saved %d class names → %s", len(class_names), class_names_path)

    with Timer("Plotting class distribution"):
        plot_class_distribution(class_map, fig_dir / "class_distribution.png")

    with Timer("Splitting dataset"):
        counts = split_dataset(class_map, output_dir)

    with Timer("Plotting augmentation samples"):
        plot_sample_augmented(output_dir, fig_dir / "augmentation_samples.png")

    # Save metadata
    metadata = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "split_counts": counts,
        "image_size": list(IMAGE_SIZE),
        "target_crops": list(TARGET_CROPS.keys()),
    }
    with open(output_dir / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Preprocessing complete — %d classes, %d total images.", len(class_names), sum(counts.values()))
    return counts


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PlantVillage images")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    preprocess(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
