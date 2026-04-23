# TARA ADAS — Firebase Cloud Setup Guide

## Why Firebase?

| Feature | Benefit |
|---|---|
| **Realtime Database** | Live telemetry from your car (FPS, steering, speed, distance) |
| **Cloud Storage** | Stores detection event snapshots (pothole photos, sign photos) |
| **Free tier** | 1GB DB storage, 5GB file storage, 1GB/day upload — more than enough |
| **Simple SDK** | Single `pip install firebase-admin` — no server setup |
| **Works offline** | If WiFi drops, prototype keeps running — cloud logging just pauses |

---

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click **"Add project"**
3. Name it: `tara-adas` (or any name)
4. Disable Google Analytics (not needed)
5. Click **"Create project"**

---

## Step 2: Enable Realtime Database

1. In the Firebase Console sidebar → **Build** → **Realtime Database**
2. Click **"Create Database"**
3. Choose location closest to you
4. Start in **"Test mode"** (allows read/write for 30 days)
5. Copy the database URL — looks like:
   ```
   https://tara-adas-default-rtdb.firebaseio.com
   ```

---

## Step 3: Enable Cloud Storage

1. Sidebar → **Build** → **Storage**
2. Click **"Get started"**
3. Start in **"Test mode"**
4. Copy the bucket name — looks like:
   ```
   tara-adas.appspot.com
   ```

---

## Step 4: Download Service Account Key

1. Click the **gear icon** (top-left) → **Project Settings**
2. Go to **"Service accounts"** tab
3. Click **"Generate new private key"**
4. Save the downloaded JSON file as:
   ```
   rpi/firebase_credentials.json
   ```

> ⚠️ **IMPORTANT:** Never commit this file to git! Add it to `.gitignore`.

---

## Step 5: Configure TARA

Edit `rpi/config.py` and fill in your Firebase details:

```python
# Firebase Realtime Database URL
FIREBASE_DB_URL = "https://tara-adas-default-rtdb.firebaseio.com"

# Firebase Storage bucket
FIREBASE_STORAGE_BUCKET = "tara-adas.appspot.com"
```

Make sure `firebase_credentials.json` is in the `rpi/` directory.

---

## Step 6: Install Dependencies on RPi

```bash
source ~/tara-venv/bin/activate
pip install firebase-admin
```

---

## Step 7: Test Connection

```bash
cd ~/TARA/rpi
python3 -c "
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate('firebase_credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'YOUR_DB_URL_HERE'
})

ref = db.reference('/tara/test')
ref.set({'status': 'connected', 'device': 'rpi4b'})
print('Firebase connection successful!')
print('Check Firebase Console → Realtime Database to see the test data.')
"
```

---

## What Gets Uploaded

| Data | Frequency | Size | Destination |
|---|---|---|---|
| **Telemetry** (FPS, steering, speed, distance) | Every 2 seconds | ~200 bytes | Realtime DB |
| **Pothole detection** frame | On detection only | ~15 KB JPEG | Cloud Storage |
| **Traffic sign** frame | On detection only | ~15 KB JPEG | Cloud Storage |
| **Lane departure** frame | Every 15th warning frame | ~15 KB JPEG | Cloud Storage |
| **Emergency stop** frame | On trigger | ~15 KB JPEG | Cloud Storage |
| **Session summary** | On shutdown | ~500 bytes | Realtime DB |

**Estimated daily usage** (1-hour test session):
- Database: ~1 MB
- Storage: ~50-100 MB (depends on detections)
- Bandwidth: < 200 MB upload

All well within Firebase free tier limits.

---

## Viewing Your Data

### Realtime Database
Go to Firebase Console → Realtime Database. You'll see:
```
tara/
  sessions/
    20260401_143022/
      started_at: "2026-04-01T14:30:22"
      status: "completed"
      runtime_seconds: 3600
      telemetry/
        -Nabc123/
          timestamp: "2026-04-01T14:30:24"
          fps: 22.3
          steering: 15
          speed: 150
          distance_cm: 45.2
          lane_detected: true
          ...
      events/
        -Ndef456/
          type: "pothole"
          timestamp: "2026-04-01T14:45:10"
          image_path: "tara/20260401_143022/events/pothole_144510_123.jpg"
          position: "left"
          confidence: 0.87
```

### Cloud Storage
Go to Firebase Console → Storage. Browse:
```
tara/
  20260401_143022/
    events/
      pothole_144510_123.jpg
      sign_145230_456.jpg
      departure_150100_789.jpg
```

---

## Running Without Cloud

The prototype works perfectly without Firebase:

```bash
# No cloud — fully offline
python3 main.py --no-cloud

# Or set in config.py:
CLOUD_ENABLED = False
```

Local CSV recordings still work in `rpi/recordings/` as a fallback.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `firebase-admin` install fails | Try: `pip install firebase-admin --no-cache-dir` |
| Connection timeout | Check WiFi connectivity: `ping google.com` |
| Permission denied | Ensure DB/Storage are in "Test mode" in Firebase Console |
| Credentials not found | Verify `firebase_credentials.json` path in `config.py` |
| Slowing down pipeline | Should never happen (async queue), but try `--no-cloud` to verify |
