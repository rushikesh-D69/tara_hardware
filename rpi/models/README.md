# TARA ADAS — TFLite Models Directory

Place your trained and converted TFLite models here:

- `tsr_mobilenetv2_int8.tflite` — Traffic Sign Recognition
- `pothole_mobilenetv2_int8.tflite` — Pothole Binary Classifier
- `pothole_ssd_mobilenetv2_int8.tflite` — Pothole SSD Detector (optional)

## How to get these models:

1. Train on PC/Colab using scripts in `training/`
2. Convert to TFLite INT8 using `training/convert_to_tflite.py`
3. Copy the .tflite files to this directory
