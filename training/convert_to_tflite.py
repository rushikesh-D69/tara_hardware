"""
TARA ADAS — TFLite Model Conversion + INT8 Quantization
Converts a trained Keras/SavedModel to optimized INT8 TFLite format.

This is the critical optimization step for Raspberry Pi 4B deployment:
  - 4× smaller model size
  - 2-3× faster inference on ARM CPU
  - Compatible with XNNPACK and ARM NN delegates

Usage:
  python convert_to_tflite.py --model saved_models/tsr_mobilenetv2_final --output tsr_int8.tflite
  python convert_to_tflite.py --model saved_models/pothole_final --output pothole_int8.tflite
  python convert_to_tflite.py --model model.h5 --output model_int8.tflite --input-size 96
"""
import os
import argparse
import numpy as np

try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow required for conversion.")
    exit(1)


def create_representative_dataset(input_size, num_samples=200):
    """
    Create a representative dataset generator for quantization calibration.

    For best results, use actual training images. This function generates
    random images as a fallback when training data isn't available.

    Args:
        input_size: Model input dimension
        num_samples: Number of calibration samples

    Returns:
        Generator function for TFLiteConverter
    """
    def representative_dataset():
        for _ in range(num_samples):
            # Random images as calibration data (replace with real images for production)
            data = np.random.rand(1, input_size, input_size, 3).astype(np.float32)
            yield [data]

    return representative_dataset


def create_representative_dataset_from_dir(data_dir, input_size, num_samples=200):
    """
    Create representative dataset from actual images.

    Args:
        data_dir: Directory containing sample images
        input_size: Model input dimension
        num_samples: Number of calibration samples

    Returns:
        Generator function
    """
    import cv2
    import glob

    image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    image_paths += glob.glob(os.path.join(data_dir, "**", "*.ppm"), recursive=True)

    if not image_paths:
        print(f"WARNING: No images found in {data_dir}, using random data")
        return create_representative_dataset(input_size, num_samples)

    # Limit to num_samples
    np.random.shuffle(image_paths)
    image_paths = image_paths[:num_samples]

    def representative_dataset():
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_size, input_size))
            img = img.astype(np.float32) / 255.0
            yield [np.expand_dims(img, axis=0)]

    return representative_dataset


def convert_to_tflite(model_path, output_path, input_size=96,
                      quantize="int8", data_dir=None):
    """
    Convert a Keras/SavedModel to optimized TFLite format.

    Args:
        model_path: Path to Keras model (.h5 or SavedModel dir)
        output_path: Output .tflite file path
        input_size: Model input dimension
        quantize: Quantization mode ("int8", "float16", "dynamic", "none")
        data_dir: Directory with calibration images (for INT8)

    Returns:
        Size of the output TFLite model in bytes
    """
    # Load model
    print(f"Loading model from: {model_path}")
    if model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # Get converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "int8":
        print("Applying INT8 quantization (best for RPi 4B)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for calibration
        if data_dir and os.path.exists(data_dir):
            rep_dataset = create_representative_dataset_from_dir(
                data_dir, input_size)
        else:
            print("Using random calibration data (provide --data-dir for better accuracy)")
            rep_dataset = create_representative_dataset(input_size)

        converter.representative_dataset = rep_dataset

        # Force full integer quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    elif quantize == "float16":
        print("Applying Float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantize == "dynamic":
        print("Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    else:
        print("No quantization (float32)")

    # Convert
    print("Converting...")
    tflite_model = converter.convert()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n✅ TFLite model saved: {output_path}")
    print(f"   Size: {size_mb:.2f} MB ({len(tflite_model):,} bytes)")

    return len(tflite_model)


def validate_tflite(tflite_path, input_size):
    """
    Quick validation of the converted TFLite model.

    Args:
        tflite_path: Path to .tflite file
        input_size: Expected input dimension
    """
    print(f"\nValidating: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input:  shape={input_details[0]['shape']}, "
          f"dtype={input_details[0]['dtype']}")
    print(f"  Output: shape={output_details[0]['shape']}, "
          f"dtype={output_details[0]['dtype']}")

    # Test with dummy input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    if input_dtype == np.uint8:
        test_input = np.random.randint(0, 255, input_shape).astype(np.uint8)
    else:
        test_input = np.random.rand(*input_shape).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_input)

    # Benchmark inference time
    import time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000)

    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"  Output sample: {output[0][:5]}...")
    print(f"  Inference time: {np.mean(times):.1f}ms avg "
          f"(min: {np.min(times):.1f}ms, max: {np.max(times):.1f}ms)")
    print("  ✅ Validation passed!")


def main():
    parser = argparse.ArgumentParser(description="Convert model to TFLite INT8")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to Keras model (.h5 or SavedModel)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .tflite file path")
    parser.add_argument("--input-size", type=int, default=96,
                        help="Model input dimension")
    parser.add_argument("--quantize", type=str, default="int8",
                        choices=["int8", "float16", "dynamic", "none"],
                        help="Quantization mode")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with calibration images")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after conversion")
    args = parser.parse_args()

    # Convert
    convert_to_tflite(
        model_path=args.model,
        output_path=args.output,
        input_size=args.input_size,
        quantize=args.quantize,
        data_dir=args.data_dir,
    )

    # Validate
    if args.validate:
        validate_tflite(args.output, args.input_size)

    print("\n" + "=" * 50)
    print("Next steps:")
    print(f"1. Copy {args.output} to your Raspberry Pi:")
    print(f"   scp {args.output} pi@raspberrypi:~/TARA/rpi/models/")
    print("2. Run the ADAS pipeline:")
    print("   python3 main.py --debug")
    print("=" * 50)


if __name__ == "__main__":
    main()
