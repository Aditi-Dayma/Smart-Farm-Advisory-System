"""
AWS SageMaker Processing Job: Image Data Preprocessing for Module 2 (Disease Detection).

Executed inside a TensorFlow/Python container on SageMaker Processing.
Reads raw PlantVillage images, filters target crops (Tomato, Potato, Corn, Grape, Apple),
generates train/validation/test partitions, and writes normalized manifests and metadata to S3.
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil

TARGET_PREFIXES = ["Tomato", "Potato", "Corn", "Grape", "Apple"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    rng = random.Random(args.seed)

    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    discovered_classes = []
    class_stats = {}

    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue
        # Check target crop match
        if any(folder.name.startswith(p) for p in TARGET_PREFIXES):
            images = [p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            if not images:
                continue
            discovered_classes.append(folder.name)
            rng.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * args.train_ratio)
            n_val = int(n_total * args.val_ratio)

            train_imgs = images[:n_train]
            val_imgs = images[n_train:n_train + n_val]
            test_imgs = images[n_train + n_val:]

            class_stats[folder.name] = {
                "total": n_total,
                "train": len(train_imgs),
                "val": len(val_imgs),
                "test": len(test_imgs)
            }

            for split_name, img_subset in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                dest_class_dir = output_path / split_name / folder.name
                dest_class_dir.mkdir(parents=True, exist_ok=True)
                for img_p in img_subset:
                    shutil.copy2(img_p, dest_class_dir / img_p.name)

    # Save class indices and processing manifest
    with open(output_path / "class_names.json", "w") as f:
        json.dump(discovered_classes, f, indent=2)

    with open(output_path / "split_stats.json", "w") as f:
        json.dump(class_stats, f, indent=2)

    print(f"Image preprocessing finished: {len(discovered_classes)} classes organized.")


if __name__ == "__main__":
    main()
