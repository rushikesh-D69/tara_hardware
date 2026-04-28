import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

"""
TARA ADAS — Model Training Script
Trains a custom MobileNetV2 classifier on your collected dataset.
Run this on your Laptop/PC.
"""

# Settings (match traffic_sign.py)
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 15      # Should be enough for 10 classes
DATASET_PATH = "dataset"

def train():
    print("--- TARA ADAS — Starting Model Training ---")

    # [1] Load and split the dataset
    # This automatically uses the folder names (e.g. '14', '1') as class labels
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected Classes: {class_names}")

    # [2] Preprocessing: Scale to [-1, 1] for MobileNetV2
    normalization_layer = layers.Rescaling(1./127.5, offset=-1)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # [3] Data Augmentation: Makes model robust to low light / tilt
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # [4] Build the Model: MobileNetV2 Alpha=0.35 (Extra Lightweight)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35
    )
    base_model.trainable = False  # Freeze the base for now

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # [5] Train (Phase 1: Head only)
    print("\nTraining classifier head...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # [6] Fine-tuning (Phase 2: Unfreeze last layers for better accuracy)
    base_model.trainable = True
    # Freeze all layers EXCEPT the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nFine-tuning model accuracy...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    # [7] Save the model
    model.save("tsr_model_custom.h5")
    print("\nTraining complete! Saved as tsr_model_custom.h5")

if __name__ == "__main__":
    train()
