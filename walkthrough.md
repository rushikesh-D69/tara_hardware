# TARA — The Complete Baby-Steps Guide 🍼

> **Read this top to bottom. Don't skip anything. Each step depends on the one before it.**

---

## Part 1: What Are We Building?

Your car prototype has **two brains**:

```
┌─────────────────────┐         USB Cable         ┌──────────────────┐
│                     │  ───────────────────────→  │                  │
│   RASPBERRY PI 4B   │    "steer left, speed 150" │     ESP32        │
│   (the THINKER)     │                            │   (the DOER)     │
│                     │  ←───────────────────────  │                  │
│   "sees" the road   │   "distance is 30cm"       │  controls motors │
│   "thinks" what to  │                            │  reads sensors   │
│    do next          │                            │                  │
└─────────────────────┘                            └──────────────────┘
        │                                                   │
    USB Webcam                                    TB6612FNG Motor Driver
    (its eyes)                                    HC-SR04 Ultrasonic
                                                  MPU-6050 IMU
                                                  Wheel Encoders
                                                  Motors
```

**RPi thinks. ESP32 does.** They talk through a USB cable.

---

## Part 2: What Does Each File Do?

### 🔵 Files that run on Raspberry Pi (`rpi/` folder)

| File | What it does | Analogy |
|---|---|---|
| `main.py` | **The boss.** Starts everything, coordinates all modules. | The driver's brain |
| `config.py` | **Settings.** Camera resolution, speed limits, thresholds. | The car's settings menu |
| `camera/capture.py` | Reads frames from your USB webcam. | The eyes |
| `adas/lane_detection.py` | Finds lane lines in the camera image. | "Am I still in my lane?" |
| `adas/traffic_sign.py` | Recognizes traffic signs (stop, speed limit, etc.) | "What does that sign say?" |
| `adas/pothole_detection.py` | Detects potholes on the road. | "Is there a hole ahead?" |
| `adas/adaptive_cruise.py` | Uses ultrasonic sensor to keep distance from obstacles. | "Am I too close to something?" |
| `adas/decision_manager.py` | Takes ALL the above outputs and decides: steer how much? speed how much? | "Okay, based on everything, here's what we do" |
| `comms/serial_bridge.py` | Sends the final decision to ESP32 via USB cable. | The mouth (tells ESP32 what to do) |
| `cloud/firebase_logger.py` | Saves data to cloud (optional, not required). | A dashcam recorder |
| `models/` | **YOU PUT TRAINED MODEL FILES HERE.** Empty right now. | The knowledge bank |

### 🟢 Files that run on your PC (for training only)

| File | What it does | When to use |
|---|---|---|
| `training/train_tsr.py` | Trains the traffic sign recognition model | Once, before running the car |
| `training/train_pothole.py` | Trains the pothole detection model | Once, before running the car |
| `training/convert_to_tflite.py` | Converts trained model to a tiny format RPi can run | Once, after training |

### 🟡 File that runs on ESP32

| File | What it does |
|---|---|
| `esp32/tara_controller/tara_controller.ino` | Receives commands from RPi, controls motors, reads sensors, sends sensor data back |

---

## Part 3: Which Features Need Training?

> [!IMPORTANT]
> **Only 2 out of 5 features need training. The other 3 work immediately with NO training.**

| Feature | Needs Training? | Why? |
|---|---|---|
| Lane Detection | ❌ **NO** | Uses math (OpenCV). Looks for white/yellow lines with color filters. |
| Adaptive Cruise Control | ❌ **NO** | Uses math (PID). Reads distance sensor, slows down if too close. |
| Traffic Sign Recognition | ✅ **YES** | Needs a neural network to recognize 43 types of signs. |
| Pothole Detection | ✅ **YES** | Needs a neural network to tell potholes apart from normal road. |
| Decision Manager | ❌ **NO** | Pure if/else logic. Combines everything and picks the safest action. |

---

## Part 4: The 5 Phases (Do Them In Order)

```
Phase 1          Phase 2          Phase 3          Phase 4          Phase 5
TRAIN            CONVERT          SETUP            CONNECT          RUN
(on your PC)     (on your PC)     (on RPi)         (hardware)       (on RPi)
   │                │                │                │                │
   ▼                ▼                ▼                ▼                ▼
Train models  →  Make them    →  Install       →  Wire it     →  python3
on GPU           tiny for Pi     software         all up         main.py
```

---

### 🟢 Phase 1: Train the Models (On Your PC or Google Colab)

> **Where:** Your Windows PC or [Google Colab](https://colab.research.google.com) (free GPU)
> **Time:** 1-3 hours
> **You only do this ONCE.**

#### Step 1.1: Get the Datasets

**For Traffic Signs:**
1. Go to https://benchmark.ini.rub.de/gtsrb_dataset.html
2. Download "GTSRB Final Training Images"
3. Unzip it
4. Put the class folders (00000, 00001, ..., 00042) inside:
   ```
   d:\Projects\TARA\training\datasets\GTSRB\
   ```

**For Potholes:**
1. Go to https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset
2. Download the dataset
3. Organize images into two folders:
   ```
   d:\Projects\TARA\training\datasets\pothole\
       pothole\     ← photos of potholes
       normal\      ← photos of normal road
   ```

#### Step 1.2: Train Traffic Sign Model

Open a terminal on your PC:
```bash
cd d:\Projects\TARA\training
pip install tensorflow numpy opencv-python
python train_tsr.py --epochs 50 --fine-tune
```

Wait... it will print progress like:
```
Epoch 1/50 - accuracy: 0.61 - val_accuracy: 0.72
Epoch 2/50 - accuracy: 0.78 - val_accuracy: 0.85
...
Epoch 50/50 - accuracy: 0.97 - val_accuracy: 0.95
```

When it finishes, you'll have: `saved_models/tsr_mobilenetv2_final/`

#### Step 1.3: Train Pothole Model

```bash
python train_pothole.py --epochs 40 --fine-tune
```

When it finishes, you'll have: `saved_models/pothole_mobilenetv2_final/`

---

### 🟢 Phase 2: Convert Models to TFLite (Still On Your PC)

> **Why?** The RPi can't run full TensorFlow models. TFLite is a tiny, fast format.

```bash
# Convert traffic sign model
python convert_to_tflite.py \
    --model saved_models/tsr_mobilenetv2_final \
    --output tsr_mobilenetv2_int8.tflite \
    --input-size 96 \
    --validate

# Convert pothole model
python convert_to_tflite.py \
    --model saved_models/pothole_mobilenetv2_final \
    --output pothole_mobilenetv2_int8.tflite \
    --input-size 128 \
    --validate
```

You now have two tiny files:
```
tsr_mobilenetv2_int8.tflite       (~1.5 MB)
pothole_mobilenetv2_int8.tflite   (~2 MB)
```

**Copy these to a USB drive or remember where they are.** You'll need them in Phase 4.

---

### 🟡 Phase 3: Flash the ESP32

> **Where:** Your PC with Arduino IDE
> **Time:** 10 minutes

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Add ESP32 board support:
   - File → Preferences → Additional Board Manager URLs
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Board Manager → Search "ESP32" → Install
3. Open file: `d:\Projects\TARA\esp32\tara_controller\tara_controller.ino`
4. Select: Tools → Board → **ESP32 Dev Module**
5. Select: Tools → Port → (your ESP32's COM port)
6. Click **Upload** (→ button)
7. Wait for "Done uploading"

**Your ESP32 is now ready.** It will:
- Listen for commands from the RPi
- Control the motors via TB6612FNG
- Read the ultrasonic sensor, encoders, and IMU
- Send sensor data back to the RPi

---

### 🔵 Phase 4: Setup the Raspberry Pi

> **Where:** On the Raspberry Pi (SSH in or use keyboard+monitor)
> **Time:** 30 minutes

#### Step 4.1: Install the OS

1. Download [Raspberry Pi OS 64-bit Lite](https://www.raspberrypi.com/software/)
2. Flash to micro SD card using Raspberry Pi Imager
3. Enable SSH during flashing
4. Put the SD card in the RPi, power it on

#### Step 4.2: Install Software

SSH into your RPi and run:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-venv python3-pip libatlas-base-dev libhdf5-dev

# Create a virtual environment
python3 -m venv ~/tara-venv
source ~/tara-venv/bin/activate

# Copy the TARA project to the RPi (from your PC)
# Option A: USB drive
# Option B: scp -r d:\Projects\TARA\rpi pi@raspberrypi:~/TARA/rpi

# Install Python packages
cd ~/TARA/rpi
pip install -r requirements.txt
```

#### Step 4.3: Copy Model Files to RPi

```bash
# From your PC (replace IP with your RPi's IP):
scp tsr_mobilenetv2_int8.tflite pi@192.168.1.XX:~/TARA/rpi/models/
scp pothole_mobilenetv2_int8.tflite pi@192.168.1.XX:~/TARA/rpi/models/
```

Or copy via USB drive into `~/TARA/rpi/models/`

#### Step 4.4: Connect Everything

```
                  ┌─────────────────────────┐
                  │     RASPBERRY PI 4B      │
                  │                          │
  USB Webcam ────→│ USB Port 1              │
                  │                          │
  ESP32 (USB) ───→│ USB Port 2              │
                  │                          │
  5V 3A Power ───→│ USB-C Power             │
                  └─────────────────────────┘

On the ESP32 side (already wired from your schematic):
  ESP32 → TB6612FNG → Motors
  ESP32 → HC-SR04
  ESP32 → MPU-6050
  ESP32 → Wheel Encoders
```

> [!IMPORTANT]
> The RPi connects to ESP32 via **USB cable** (the same one you used to flash it).
> The RPi connects to the webcam via **another USB port**.
> Power the RPi with a **dedicated 5V 3A supply**, NOT from the motor battery.

#### Step 4.5: Quick Checks

```bash
# Check webcam is detected
ls /dev/video*
# Should show: /dev/video0

# Check ESP32 is detected  
ls /dev/ttyUSB*
# Should show: /dev/ttyUSB0

# Fix serial permissions (one time only)
sudo usermod -a -G dialout $USER
# Log out and back in after this
```

---

### 🚀 Phase 5: Run!

```bash
# Activate the virtual environment
source ~/tara-venv/bin/activate
cd ~/TARA/rpi

# ── TEST 1: Camera only (no ESP32 needed) ──
# This tests lane detection + TSR + pothole with just the webcam
python3 main.py --no-serial --no-cloud --debug

# What you should see:
# - A window showing the camera feed
# - Green lines on detected lanes
# - FPS counter
# - Press 'q' to quit

# ── TEST 2: Camera + ESP32 ──
# Plug in the ESP32 via USB, then:
python3 main.py --no-cloud --debug

# Now the car should actually move!
# Lane keeping, obstacle avoidance, everything is live.

# ── TEST 3: Full run (production) ──
python3 main.py

# No window, no debug — just runs headlessly.
# Ctrl+C to stop.
```

---

## Part 5: What Happens When You Run `main.py`?

Here's what happens **every single frame** (20+ times per second):

```
1. 📷 Camera grabs a frame (640x480 image)
           │
           ▼
2. 🛣️  Lane Detection looks for lane lines
   │      (OpenCV, ~10ms, every frame)
   │
   ├──→ Frame 1, 5, 9...: 🚦 Traffic Sign Recognition runs
   │      (TFLite model, ~20ms)
   │
   ├──→ Frame 3, 7, 11...: 🕳️  Pothole Detection runs
   │      (TFLite model, ~20ms)
   │
   └──→ Frame 0, 2, 4...: 📏 ACC reads distance sensor
          (serial read, ~2ms)
           │
           ▼
3. 🧠 Decision Manager combines everything:
   "Lane says steer +15, ACC says speed 150,
    no potholes, no signs → CMD:15,150,0"
           │
           ▼
4. 📡 Serial Bridge sends to ESP32:
   "CMD:15,150,0\n"
           │
           ▼
5. ⚡ ESP32 controls motors:
   Left motor: 165 PWM, Right motor: 135 PWM
   (differential steering)
           │
           ▼
6. 📡 ESP32 sends back sensor data:
   "SEN:45.2,1250,1248,0.02,-0.01,9.81\n"
           │
           ▼
   (repeat from step 1)
```

---

## Part 6: If Something Goes Wrong

| Problem | Solution |
|---|---|
| `Camera not found` | Run `ls /dev/video*`. Try CAMERA_INDEX = 1 in config.py |
| `Serial permission denied` | Run `sudo usermod -a -G dialout $USER` then reboot |
| `TSR model not loaded` | You haven't copied the `.tflite` file to `rpi/models/` yet |
| `ESP32 not responding` | Check USB cable. Run `ls /dev/ttyUSB*`. Try unplugging and replugging |
| `Low FPS (< 5)` | Add a heatsink + fan to RPi. Or reduce PROC_WIDTH to 240 in config.py |
| `Car drives erratically` | Tune PID values in config.py. Start with LKA_PID_KP = 0.3 |
| `Lane not detected` | Adjust HSV thresholds in config.py for your track's lane color |
| `Car doesn't stop for obstacles` | Check HC-SR04 wiring. Test with: `python3 -c "from comms.serial_bridge import ..."` |

---

## Part 7: Files You Can Ignore For Now

These are "nice to have" but **not required** for a working prototype:

| File | Why you can skip it |
|---|---|
| `cloud/firebase_logger.py` | Optional data logging. Use `--no-cloud` flag |
| `docs/cloud_setup.md` | Only needed if you want Firebase |
| `utils/fps_counter.py` | Auto-used by main.py, don't touch it |
| `utils/logger.py` | Auto-used by main.py, don't touch it |
| `.gitignore` | Only matters if you use git |

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│                    TARA QUICK REFERENCE                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  TO TRAIN MODELS (on PC):                                │
│    cd training/                                          │
│    python train_tsr.py --epochs 50 --fine-tune           │
│    python train_pothole.py --epochs 40 --fine-tune       │
│    python convert_to_tflite.py --model ... --output ...  │
│                                                          │
│  TO FLASH ESP32:                                         │
│    Open tara_controller.ino in Arduino IDE → Upload      │
│                                                          │
│  TO RUN ON RPi:                                          │
│    source ~/tara-venv/bin/activate                       │
│    cd ~/TARA/rpi                                         │
│    python3 main.py --debug            (with screen)      │
│    python3 main.py                    (headless)         │
│    python3 main.py --no-serial        (camera only)      │
│                                                          │
│  TO CHANGE SETTINGS:                                     │
│    Edit rpi/config.py                                    │
│                                                          │
│  TO STOP:                                                │
│    Press Ctrl+C  (or press 'q' in debug window)          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```
