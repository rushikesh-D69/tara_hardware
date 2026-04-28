"""
TARA ADAS — Traffic Sign Recognition Training Script
Train MobileNetV2 (α=0.35, 96x96) on GTSRB dataset.

Run on a PC with GPU or Google Colab — NOT on the Raspberry Pi.

Usage:
  python train_tsr.py                         # Train from scratch
  python train_tsr.py --epochs 30             # Custom epochs
  python train_tsr.py --resume checkpoint.h5  # Resume training
"""
import os
import argparse
import numpy as np

# Check TensorFlow is available (not expected on RPi)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    )
    print(f"TensorFlow {tf.__version__} — GPU: {tf.config.list_physical_devices('GPU')}")
except ImportError:
    print("ERROR: TensorFlow not found. Install with: pip install tensorflow")
    print("This script must run on a PC/Colab with GPU, NOT on the Raspberry Pi.")
    exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_SIZE = 96          # Model input dimension (96x96)
NUM_CLASSES = 43         # GTSRB has 43 sign classes
ALPHA = 0.35             # MobileNetV2 width multiplier (smaller = faster)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_DIR = "datasets/GTSRB"
MODEL_SAVE_DIR = "saved_models"


def download_gtsrb():
    """Download and extract the GTSRB dataset if not present."""
    if os.path.exists(DATASET_DIR):
        print(f"Dataset found at {DATASET_DIR}")
        return

    print("Downloading GTSRB dataset...")
    import urllib.request
    import zipfile

    url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    zip_path = "gtsrb_train.zip"

    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATASET_DIR)

    os.remove(zip_path)
    print("Dataset ready!")


def load_dataset():
    """
    Load GTSRB dataset using tf.keras.utils.image_dataset_from_directory
    or manual loading.

    Returns:
        (train_ds, val_ds) — tf.data.Dataset objects
    """
    # Try to load with image_dataset_from_directory if organized by class folders
    data_dir = DATASET_DIR

    # Check if the data is in the standard structure
    # GTSRB/Final_Training/Images/00000/, /00001/, etc.
    possible_paths = [
        os.path.join(data_dir, "GTSRB", "Final_Training", "Images"),
        os.path.join(data_dir, "Final_Training", "Images"),
        os.path.join(data_dir, "train"),
        data_dir,
    ]

    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subdirs) >= 10:  # At least 10 class folders
                actual_path = p
                break

    if actual_path is None:
        print("ERROR: Could not find GTSRB image directories.")
        print(f"Please download GTSRB and place class folders in: {DATASET_DIR}")
        print("Expected structure: GTSRB/00000/, GTSRB/00001/, ..., GTSRB/00042/")
        exit(1)

    print(f"Loading dataset from: {actual_path}")

    # Data augmentation for training
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    # Load training set
    train_ds = keras.utils.image_dataset_from_directory(
        actual_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Load validation set
    val_ds = keras.utils.image_dataset_from_directory(
        actual_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # Apply augmentation to training data
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Optimize pipeline performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model():
    """
    Build MobileNetV2 (α=0.35) for traffic sign classification.

    Returns:
        Compiled Keras model
    """
    # Use MobileNetV2 as feature extractor (pretrained on ImageNet)
    base_model = keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=ALPHA,           # Width multiplier — 0.35 = smallest variant
        include_top=False,     # Remove ImageNet classification head
        weights="imagenet",
    )

    # Freeze base model initially
    base_model.trainable = False

    # Build classification head
    model = keras.Sequential([
        # Normalization: scale pixel values from [0, 255] to [0, 1]
        layers.Rescaling(1.0 / 255.0, input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),

        # Feature extraction backbone
        base_model,

        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum(p.numpy().size for p in model.trainable_weights):,}")

    return model, base_model


def fine_tune(model, base_model, train_ds, val_ds, initial_epochs):
    """
    Unfreeze top layers of MobileNetV2 and fine-tune.

    Args:
        model: Trained Keras model
        base_model: MobileNetV2 base
        train_ds: Training dataset
        val_ds: Validation dataset
        initial_epochs: Number of initial training epochs completed
    """
    print("\n" + "=" * 50)
    print("Phase 2: Fine-tuning top layers")
    print("=" * 50)

    # Unfreeze the top 30 layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_epochs = 20
    total_epochs = initial_epochs + fine_tune_epochs

    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, "tsr_finetuned_best.h5"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        EarlyStopping(patience=5, monitor="val_accuracy", restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        callbacks=callbacks,
    )

    return history


def main():
    parser = argparse.ArgumentParser(description="Train TSR model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--fine-tune", action="store_true", help="Enable fine-tuning phase")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Download dataset
    if not args.no_download:
        download_gtsrb()

    # Load data
    train_ds, val_ds = load_dataset()

    # Build or resume model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = keras.models.load_model(args.resume)
        base_model = None
    else:
        model, base_model = build_model()

    # Training callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, "tsr_best.h5"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        EarlyStopping(patience=8, monitor="val_accuracy", restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    # Phase 1: Train classification head
    print("\n" + "=" * 50)
    print("Phase 1: Training classification head")
    print("=" * 50)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune (optional)
    if args.fine_tune and base_model is not None:
        fine_tune(model, base_model, train_ds, val_ds, args.epochs)

    # Save final model
    final_path = os.path.join(MODEL_SAVE_DIR, "tsr_mobilenetv2_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    # Also save as .h5
    h5_path = os.path.join(MODEL_SAVE_DIR, "tsr_mobilenetv2_final.h5")
    model.save(h5_path)
    print(f"H5 model saved to: {h5_path}")

    # Evaluate
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")

    print("\n✅ Training complete! Next step:")
    print(f"   python convert_to_tflite.py --model {final_path} --output tsr_mobilenetv2_int8.tflite")


if __name__ == "__main__":
    main()
