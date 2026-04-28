# TARA ADAS — Dataset Guide

## Traffic Sign Recognition (TSR)

### GTSRB Dataset
- **Source:** [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- **Size:** ~50,000 images, 43 classes
- **Format:** PPM images organized by class ID (00000 - 00042)

**Download:**
```bash
# Option 1: Direct download
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip -d GTSRB/

# Option 2: TensorFlow Datasets
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('german_traffic_sign')"
```

**After download, organize as:**
```
training/datasets/GTSRB/
    00000/          ← Speed limit 20 km/h
    00001/          ← Speed limit 30 km/h
    ...
    00042/          ← End of no passing >3.5t
```

---

## Pothole Detection

### Recommended Datasets

1. **Kaggle Pothole Dataset (sachinpatel21)**
   - URL: https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset
   - ~5,000 images

2. **Roboflow Pothole Datasets**
   - URL: https://universe.roboflow.com/search?q=pothole
   - Multiple datasets available, export as classification format

3. **Custom Collection**
   - Capture your own images of your test track surface
   - Include both pothole/obstacle images and clear road images
   - Aim for 500+ images per class minimum

**Organize as:**
```
training/datasets/pothole/
    pothole/        ← Images containing potholes
    normal/         ← Images of clear road
```

---

## Data Augmentation Tips

Both training scripts include automatic augmentation, but for best results:

- Capture images under **varied lighting** (indoor/outdoor, different times)
- Include **multiple angles** and **distances**
- Balance classes (equal number of images per class)
- For pothole detection on an indoor track: photograph your specific track surface obstacles

## Using Google Colab

1. Mount Google Drive
2. Upload datasets to Drive
3. Upload training scripts to Drive
4. Run with GPU runtime:
   ```python
   !python train_tsr.py --epochs 50 --fine-tune
   !python convert_to_tflite.py --model saved_models/tsr_mobilenetv2_final \
       --output tsr_mobilenetv2_int8.tflite --validate
   ```
5. Download the `.tflite` files and copy to your RPi
