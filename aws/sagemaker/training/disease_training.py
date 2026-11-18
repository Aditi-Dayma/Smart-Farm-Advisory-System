"""
AWS SageMaker Training Entry Point: MobileNetV2 Disease Detection (Module 2).

Reads image splits from SM_CHANNEL_TRAIN and SM_CHANNEL_VALIDATION,
performs transfer learning with fine-tuning, and saves SavedModel / H5 artifacts to SM_MODEL_DIR.
"""

import argparse
import json
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-phase1", type=int, default=10)
    parser.add_argument("--epochs-phase2", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate-phase1", type=float, default=1e-4)
    parser.add_argument("--learning-rate-phase2", type=float, default=1e-5)
    parser.add_argument("--img-size", type=int, default=224)

    # Channels
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--val", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/val"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    img_size = (args.img_size, args.img_size)

    # Augmentation for train channel
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        args.train,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        args.val,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    with open(model_dir / "class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)

    # Transfer Learning Architecture: MobileNetV2
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate_phase1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Phase 1: Train classification head
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    print("Beginning Phase 1 training (head only)...")
    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs_phase1, callbacks=callbacks)

    # Phase 2: Fine-tune top 30 layers
    print("Unfreezing last 30 layers for Phase 2 fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate_phase2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs_phase2, callbacks=callbacks)

    val_acc = history2.history["val_accuracy"][-1]
    val_loss = history2.history["val_loss"][-1]
    print(f"val_accuracy: {val_acc:.4f}")
    print(f"val_loss: {val_loss:.4f}")

    # Save format for SageMaker TensorFlow Serving
    model.save(str(model_dir / "1"))
    print(f"Model exported to {model_dir / '1'}")


if __name__ == "__main__":
    main()
