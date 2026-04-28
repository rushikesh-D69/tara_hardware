import tensorflow as tf
import numpy as np
import os
import argparse

"""
TARA ADAS — TFLite Export & INT8 Quantization
Converts your .h5 Keras model to a high-speed TFLite model for RPi.
Run this on your Laptop/PC.
"""

def representative_data_gen(dataset_path, num_samples=100):
    """
    Generator for calibration data.
    Takes a small subset of your real dataset to calibrate the INT8 weights.
    """
    # Find all image paths in the dataset
    all_imgs = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".jpg") or f.endswith(".png"):
                all_imgs.append(os.path.join(root, f))
    
    # Shuffle and pick a small number for calibration
    np.random.shuffle(all_imgs)
    samples = all_imgs[:num_samples]
    
    for path in samples:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [96, 96])
        img = (img / 127.5) - 1.0  # MobileNetV2 Preprocessing [-1, 1]
        img = tf.expand_dims(img, axis=0)
        yield [img]

def export():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tsr_model_custom.h5")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--output", type=str, default="tsr_mobilenetv2_int8.tflite")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Could not find model {args.model}")
        return

    print(f"--- Converting {args.model} to TFLite (INT8) ---")

    # Load the Keras model
    model = tf.keras.models.load_model(args.model)
    
    # [1] Set up Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # [2] Apply Full Integer Quantization (INT8)
    # This is what makes the model run at 15-20ms on RPi 4B.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(args.dataset)
    
    # Ensure inputs and outputs are also strictly integer (INT8)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # [3] Convert
    tflite_model = converter.convert()

    # [4] Save the file
    with open(args.output, "wb") as f:
        f.write(tflite_model)
    
    print(f"\nSuccess! Saved quantized model to: {args.output}")
    print("Next: Copy this .tflite file to your Raspberry Pi folder: models/")

if __name__ == "__main__":
    export()
