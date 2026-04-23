"""
TARA ADAS — Pothole Detection Training Script
Train a binary classifier (MobileNetV2 α=0.35, 128x128) for pothole/no-pothole.

Run on a PC with GPU or Google Colab — NOT on the Raspberry Pi.

Usage:
  python train_pothole.py
  python train_pothole.py --epochs 30
  python train_pothole.py --input-size 128   # For classifier
  python train_pothole.py --input-size 300 --ssd  # For SSD detector
"""
import os
import argparse
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    )
    print(f"TensorFlow {tf.__version__} — GPU: {tf.config.list_physical_devices('GPU')}")
except ImportError:
    print("ERROR: TensorFlow not found. Run on PC/Colab, not Raspberry Pi.")
    exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_SIZE = 128         # Classifier model input
NUM_CLASSES = 2          # Binary: pothole vs clear road
ALPHA = 0.35             # MobileNetV2 width multiplier
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
DATASET_DIR = "datasets/pothole"
MODEL_SAVE_DIR = "saved_models"


def prepare_dataset():
    """
    Prepare the pothole dataset.

    Expected directory structure:
    datasets/pothole/
        pothole/      ← images of potholes
        normal/       ← images of clear road

    If using a Kaggle/Roboflow dataset, organize into these two folders.
    """
    if not os.path.exists(DATASET_DIR):
        print(f"""
╔══════════════════════════════════════════════════════════╗
║  Please download a pothole dataset and organize as:     ║
║                                                         ║
║  {DATASET_DIR}/                                       ║
║    pothole/    ← images of potholes                     ║
║    normal/     ← images of clear road                   ║
║                                                         ║
║  Recommended datasets:                                  ║
║  • Kaggle: pothole-image-dataset (by sachinpatel21)     ║
║  • Roboflow: search "pothole detection"                 ║
║  • Combine multiple sources for better accuracy         ║
╚══════════════════════════════════════════════════════════╝
""")
        exit(1)

    # Verify structure
    classes = os.listdir(DATASET_DIR)
    dirs = [c for c in classes if os.path.isdir(os.path.join(DATASET_DIR, c))]
    if len(dirs) < 2:
        print(f"ERROR: Expected at least 2 class folders in {DATASET_DIR}")
        print(f"Found: {dirs}")
        exit(1)

    total_images = sum(
        len(os.listdir(os.path.join(DATASET_DIR, d)))
        for d in dirs
    )
    print(f"Dataset: {len(dirs)} classes, {total_images} total images")
    for d in dirs:
        count = len(os.listdir(os.path.join(DATASET_DIR, d)))
        print(f"  {d}: {count} images")


def load_dataset(input_size):
    """
    Load pothole dataset with augmentation.

    Args:
        input_size: Image dimension for model input

    Returns:
        (train_ds, val_ds) — tf.data.Dataset objects
    """
    # Augmentation pipeline
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.3),
        layers.RandomContrast(0.3),
        layers.RandomTranslation(0.1, 0.1),
    ])

    # Training set
    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(input_size, input_size),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Validation set
    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(input_size, input_size),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Apply augmentation
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_classifier(input_size):
    """
    Build MobileNetV2 binary classifier for pothole detection.

    Args:
        input_size: Input image dimension

    Returns:
        Compiled Keras model
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        alpha=ALPHA,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = keras.Sequential([
        layers.Rescaling(1.0 / 255.0, input_shape=(input_size, input_size, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model, base_model


def main():
    parser = argparse.ArgumentParser(description="Train Pothole Detection Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--fine-tune", action="store_true")
    args = parser.parse_args()

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Prepare and load dataset
    prepare_dataset()
    train_ds, val_ds = load_dataset(args.input_size)

    # Build model
    model, base_model = build_classifier(args.input_size)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, "pothole_best.h5"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        EarlyStopping(patience=8, monitor="val_accuracy", restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    # Phase 1: Train head
    print("\n" + "=" * 50)
    print("Phase 1: Training classification head")
    print("=" * 50)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune
    if args.fine_tune:
        print("\n" + "=" * 50)
        print("Phase 2: Fine-tuning")
        print("=" * 50)

        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        ft_callbacks = [
            ModelCheckpoint(
                os.path.join(MODEL_SAVE_DIR, "pothole_finetuned_best.h5"),
                save_best_only=True,
                monitor="val_accuracy",
            ),
            EarlyStopping(patience=5, restore_best_weights=True),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + 15,
            initial_epoch=args.epochs,
            callbacks=ft_callbacks,
        )

    # Save final model
    final_path = os.path.join(MODEL_SAVE_DIR, "pothole_mobilenetv2_final")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    # Evaluate
    loss, accuracy = model.evaluate(val_ds)
    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")

    print(f"\n✅ Training complete! Next step:")
    print(f"   python convert_to_tflite.py --model {final_path} --output pothole_mobilenetv2_int8.tflite")


if __name__ == "__main__":
    main()
