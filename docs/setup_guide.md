# TARA ADAS — Raspberry Pi 4B Setup Guide

## Hardware Requirements

| Component | Purpose | Notes |
|---|---|---|
| Raspberry Pi 4B (4GB) | ML inference brain | **64-bit OS required** |
| Pi Camera V2 or USB webcam | Vision input | Wide-angle (160°) recommended |
| MicroSD card (32GB+) | OS + models | Class 10 or faster |
| Heatsink + Fan | Thermal management | **Essential for continuous ML** |
| USB cable | RPi ↔ ESP32 communication | Micro-USB or USB-C |
| 5V 3A power supply | RPi power | Dedicated, not shared with motors |

---

## Step 1: Install Raspberry Pi OS (64-bit)

1. Download [Raspberry Pi OS (64-bit, Lite)](https://www.raspberrypi.com/software/)
2. Flash to microSD using Raspberry Pi Imager
3. Enable SSH and set WiFi during flashing
4. Boot and SSH into the Pi:
   ```bash
   ssh pi@raspberrypi.local
   ```

---

## Step 2: System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    libatlas-base-dev \
    libjasper-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    v4l-utils

# (Optional) Install camera tools
sudo apt install -y libcamera-tools
```

---

## Step 3: Python Environment

```bash
# Create virtual environment
python3 -m venv ~/tara-venv
source ~/tara-venv/bin/activate

# Install Python dependencies
cd ~/TARA/rpi
pip install -r requirements.txt

# Verify OpenCV
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Verify TFLite
python3 -c "import tflite_runtime.interpreter as tflite; print('TFLite OK')"
```

> **Note:** If `tflite-runtime` install fails, try:
> ```bash
> pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
> ```

---

## Step 4: Camera Setup

### USB Webcam
```bash
# Check if camera is detected
v4l2-ctl --list-devices

# Test capture
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f'Camera OK: {frame.shape}' if ret else 'FAILED')
cap.release()
"
```

### Pi Camera Module (CSI)
```bash
# Enable camera in config
sudo raspi-config
# → Interface Options → Camera → Enable

# Test with libcamera
libcamera-still -o test.jpg
```

---

## Step 5: Transfer Models

From your training PC, copy the TFLite models to the Pi:

```bash
# On your PC:
scp tsr_mobilenetv2_int8.tflite pi@raspberrypi:~/TARA/rpi/models/
scp pothole_mobilenetv2_int8.tflite pi@raspberrypi:~/TARA/rpi/models/
```

---

## Step 6: Connect ESP32

1. Plug ESP32 into Pi USB port
2. Check connection:
   ```bash
   ls /dev/ttyUSB*
   # Should show /dev/ttyUSB0
   
   # Grant serial permissions
   sudo usermod -a -G dialout $USER
   # Log out and back in after this
   ```

---

## Step 7: Run TARA ADAS

```bash
source ~/tara-venv/bin/activate
cd ~/TARA/rpi

# Test without ESP32 first (vision only)
python3 main.py --no-serial --debug

# Full pipeline with ESP32
python3 main.py --debug

# Production mode (no GUI)
python3 main.py

# With video file for testing
python3 main.py --video test_drive.mp4 --no-serial --debug
```

---

## Step 8: Thermal Management

**Critical:** The RPi 4B will thermal-throttle under continuous ML inference.

```bash
# Monitor temperature
watch -n 1 vcgencmd measure_temp

# Should stay below 80°C
# If hitting 80°C+, improve cooling:
#   1. Add heatsink on CPU
#   2. Add active fan (PWM controlled)
#   3. Consider an aluminum case with passive cooling
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Camera not found | Check `v4l2-ctl --list-devices`, try index 0 or 1 |
| Serial permission denied | `sudo usermod -a -G dialout $USER`, then re-login |
| TFLite import error | Install correct wheel for aarch64: `pip install tflite-runtime` |
| Low FPS (<5) | Reduce `PROC_WIDTH/HEIGHT` in config.py, or disable pothole SSD |
| Thermal throttling | Add heatsink + fan, reduce to 3 active modules |
| ESP32 not responding | Check baud rate matches (115200), try different USB port |

---

## Auto-Start on Boot (Optional)

Create a systemd service to start TARA on boot:

```bash
sudo nano /etc/systemd/system/tara-adas.service
```

```ini
[Unit]
Description=TARA ADAS Pipeline
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/TARA/rpi
ExecStart=/home/pi/tara-venv/bin/python3 main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable tara-adas
sudo systemctl start tara-adas
```
