# adas/ — ADAS Perception & Decision Modules

Mathematics, algorithms, and design rationale for every module in the pipeline.

---

## 1. Lane Detection (`lane_detection.py`)

### 1.1 Preprocessing — Adaptive Thresholding

Rather than a global brightness threshold (which fails under uneven indoor lighting), each pixel is compared to its **local neighbourhood mean**:

```
Binary(x, y) = 255   if  I(x, y) > mean_{B}(x, y) - C
               0     otherwise
```

Where:
- `I(x, y)` — grayscale intensity at pixel (x, y)
- `mean_B(x, y)` — Gaussian-weighted mean over a B x B neighbourhood (`LANE_ADAPTIVE_BLOCK_SIZE = 51`)
- `C` — constant offset (`LANE_ADAPTIVE_C = -15`); negative means "detect pixels brighter than local mean"

This makes the detector invariant to global lighting gradients and glossy floor reflections.

### 1.2 Perspective Transform — Bird's-Eye View (BEV)

A perspective warp maps the camera's trapezoidal road view to a top-down rectangle. Given source quad points `src` and destination quad points `dst` (both defined as ratios of frame dimensions in `config.py`):

```
M = getPerspectiveTransform(src, dst)

BEV(x', y') = M * [x, y, 1]^T   (homogeneous coordinates)
```

The inverse matrix `M_inv = getPerspectiveTransform(dst, src)` is precomputed to project detected lane points back to camera space for debug overlay.

### 1.3 Sliding-Window Lane Search

Starting from the histogram peak of the bottom half of the BEV mask, N = 12 windows of height `h / N` slide upward. At each level:

```
win_y_low  = H - (i+1) * window_height
win_y_high = H - i     * window_height

Good pixels: { (x, y) | win_y_low <= y < win_y_high
                        AND  x_center - margin <= x < x_center + margin }
```

If `|good pixels| > min_pix_recenter (= 30)`, the window recenters on their mean x. This adapts to curved lanes without assuming straight lines.

### 1.4 Polynomial Fit

Collected lane pixels `(x_i, y_i)` are fit to a **2nd-order polynomial in BEV space** (x as a function of y, because lanes are more vertical than horizontal):

```
x = a * y^2 + b * y + c
```

Solved via least squares: `[a, b, c] = polyfit(y_array, x_array, 2)`

A history deque of depth 5 stores past fits. When the current frame yields too few pixels, the **mean of past fits** is used as a fallback:

```
fit_smooth = mean([fit_{t}, fit_{t-1}, ..., fit_{t-4}])
```

### 1.5 Lane-Center Offset and Steering

With both fits available, the lane center at the bottom of the BEV frame (`y = H`) is:

```
x_left  = a_L * H^2 + b_L * H + c_L
x_right = a_R * H^2 + b_R * H + c_R

lane_center_x = (x_left + x_right) / 2
offset = lane_center_x - frame_center_x        (pixels, + = drifting right)
```

If only one lane is visible, the missing lane is estimated by adding/subtracting an assumed lane width (45% of BEV frame width):

```
x_right_est = x_left + 0.45 * W       (if only left lane visible)
x_left_est  = x_right - 0.45 * W      (if only right lane visible)
```

### 1.6 Steering Correction (LKA)

A proportional controller with a deadband converts the pixel offset to a normalised steering signal:

```
if |offset| <= deadband:
    steering = 0.0
else:
    effective_offset = offset - sign(offset) * deadband
    steering = clamp(effective_offset / (W/2), -1.0, 1.0)
```

`W/2` normalises the pixel offset so that an offset equal to half the frame width gives full steering (`1.0`). The deadband (default 0) eliminates jitter on a centered car.

**Steer gain** in `DecisionManager` scales this further:

```
steer_x = steering * steer_gain * min(1.0, confidence + 0.3)
```

where `confidence = min(1.0, (left_pixels + right_pixels) / 1000)`.

### 1.7 Lane Departure Warning (LDW)

```
departure_warning = True   if |offset| > LANE_DEPARTURE_THRESHOLD (30 px)
```

---

## 2. Traffic Sign Recognition (`traffic_sign.py`)

### 2.1 MobileNetV2 Preprocessing

The model was trained with Keras `preprocess_input`, which maps `[0, 255]` to `[-1, 1]`:

```
x_preprocessed = (x_uint8 / 127.5) - 1.0
```

For INT8 quantised models, the raw `uint8` tensor is passed directly; the quantisation parameters perform the equivalent scaling internally:

```
x_float = (x_int8 - zero_point) * scale
```

### 2.2 Softmax (applied if model outputs logits)

If the output does not sum to ~1 or contains negatives, softmax is applied:

```
sigma(z)_i = exp(z_i - max(z)) / sum_j(exp(z_j - max(z)))
```

Subtracting `max(z)` before exponentiation prevents numerical overflow (log-sum-exp trick).

### 2.3 INT8 Dequantisation

```
x_float = (x_int8 - zero_point) * scale
```

`scale` and `zero_point` are read from the TFLite `quantization_parameters` dict. A legacy fallback reads from the older `quantization` tuple field.

### 2.4 Temporal Majority Voting

A deque of depth 5 stores the last 5 class predictions. The output is:

```
voted_class = argmax(count(class_id) for class_id in buffer)
```

Output is gated: accepted only if `voted_count >= 2` (sign must appear in at least 2 of 5 recent frames). This eliminates single-frame false positives without introducing significant latency.

### 2.5 ROI Crop

The centre 60% x 75% of the frame is cropped before resize to avoid wasting model capacity on floor, ceiling, and walls:

```
y1, y2 = int(H * 0.20), int(H * 0.95)
x1, x2 = int(W * 0.20), int(W * 0.80)
roi = frame[y1:y2, x1:x2]
```

### 2.6 Dark Frame Rejection

```
if roi.mean() < 30:   # 0-255 scale
    skip inference
```

Saves ~15 ms per frame when the scene is too dark to contain a readable sign.

---

## 3. Pothole Detection (`pothole_detection.py`)

### 3.1 Road ROI Extraction

Only the lower half of the frame is analysed (road surface, not sky):

```
road_roi = frame[int(H * 0.5) : H, :]
```

### 3.2 Binary Classifier Output

The model outputs a 2-element probability vector:

```
[p_clear, p_pothole]   where p_clear + p_pothole ~= 1
```

Detection fires when:

```
p_pothole >= POTHOLE_CONFIDENCE_THRESHOLD (default 0.60)
```

### 3.3 Position Estimation — Edge Density

Potholes create higher edge density than smooth road surface. The road ROI is split into thirds and Canny edge counts are compared:

```
edges = Canny(gray, low=30, high=100)

left_edges   = sum(edges[:, :W/3]     > 0)
center_edges = sum(edges[:, W/3:2W/3] > 0)
right_edges  = sum(edges[:, 2W/3:]    > 0)

position = argmax(left_edges, center_edges, right_edges)
```

### 3.4 Avoidance Steer

```
if position == "left":   avoidance_steer = +POTHOLE_STEER_MAGNITUDE   (steer right)
if position == "right":  avoidance_steer = -POTHOLE_STEER_MAGNITUDE   (steer left)
if position == "center": avoidance_steer = +POTHOLE_STEER_MAGNITUDE   (default: dodge right)
```

Passed to `DecisionManager` as PWM offset; divided by 100 for normalisation:

```
steer_x_pothole = avoidance_steer / 100.0   -> [-1.0, 1.0]
```

---

## 4. Adaptive Cruise Control (`adaptive_cruise.py`)

Vision-only (no distance sensor). Outputs a normalised cruise speed setpoint:

```
cruise = ACC_DEFAULT_SPEED / 255.0   (normalised 0-1)

if TSR speed cap is active:
    cruise = min(cruise, tsr_speed_cap)

speed_norm = clamp(cruise, 0.0, ACC_MAX_SPEED / 255.0)
```

The `tsr_speed_cap` is read directly from `config.TSR_SPEED_LIMITS` — already normalised, so no further division occurs.

---

## 5. Traffic Light Recognition (`traffic_light.py`)

### 5.1 Sky ROI

Only the top 30% of the frame is analysed (where overhead traffic lights appear):

```
sky_roi = frame[0 : int(H * 0.3), :]
```

### 5.2 HSV Colour Masking

Red hue wraps around H = 0/180 in OpenCV HSV, requiring two masks:

```
mask_red = inRange(hsv, (0, 100, 100), (10, 255, 255))
         | inRange(hsv, (170, 100, 100), (180, 255, 255))

mask_green  = inRange(hsv, (40, 50, 50), (90, 255, 255))
mask_yellow = inRange(hsv, (15, 100, 100), (35, 255, 255))
```

### 5.3 Circularity Validation

For each contour in a colour mask:

```
circularity = (4 * pi * area) / perimeter^2
```

A perfect circle gives circularity = 1. Traffic lights are accepted when `circularity >= 0.5`. This rejects rectangular signs, clothing, and other coloured objects.

Confidence score:

```
score = area / 1000.0    (larger blob = more confident)
```

### 5.4 Temporal Smoothing

State is confirmed only when 2 consecutive frames agree:

```
if len(history) >= 2 and all states in history are equal:
    output stable_state
```

---

## 6. OpenCV Sign Detector (`sign_detector_cv.py`)

### 6.1 Blue Circle Detection

```
mask_blue = inRange(HSV, (100, 100, 50), (140, 255, 255))
```

Morphological open (5x5) removes noise; dilate closes gaps from reflections.

### 6.2 Arrow Direction — Centroid Analysis

The white arrow inside the blue circle is masked:

```
mask_white = inRange(HSV, (0, 0, 180), (180, 50, 255))
```

The centroid (centre of mass) of white pixels is computed using image moments:

```
M = moments(mask_white)
cx = M["m10"] / M["m00"]     (x centroid)
cy = M["m01"] / M["m00"]     (y centroid)
```

Normalised offset from the bounding-box centre:

```
offset_x = (cx - box_width / 2) / box_width    in [-0.5, 0.5]
```

Decision rule:

```
if offset_x >  0.05: return "RIGHT"
if offset_x < -0.05: return "LEFT"
```

The centroid of an arrow pointing right lies to the right of the geometric centre, and vice-versa.

---

## 7. Decision Manager (`decision_manager.py`)

### 7.1 Exponential Moving Average (Smoothing)

Raw steering and speed signals are smoothed before actuation to prevent jitter:

```
smooth_steer_{t} = alpha_s * smooth_steer_{t-1} + (1 - alpha_s) * raw_steer_{t}
smooth_speed_{t} = alpha_v * smooth_speed_{t-1} + (1 - alpha_v) * raw_speed_{t}
```

Parameters: `alpha_s = 0.10`, `alpha_v = 0.40`

Small alpha = faster response (less history weight). Steering uses lower alpha because the lane detector's polynomial history already provides smoothing; the EMA just removes sensor noise.

### 7.2 Dynamic Speed Reduction on Turns

```
steer_abs = |steer_x|
if steer_abs > 0.3:
    speed_factor = 1.0 - (steer_abs - 0.3) * 0.8
    raw_speed   *= max(0.4, speed_factor)
```

At `steer_abs = 0.3`, `speed_factor = 1.0` (no reduction).
At `steer_abs = 1.0`, `speed_factor = 1.0 - 0.56 = 0.44`, floored to 0.4.

### 7.3 Pothole Avoidance Blend-Back

After the fixed hold period (`pothole_hold_sec = 0.8 s`), steering is linearly interpolated back to lane-keep steering:

```
blend_progress = (elapsed - hold_sec) / blend_sec    in [0, 1]

steer_x = (1 - blend_progress) * pothole_steer
        +      blend_progress  * smooth_steer
```

This prevents an abrupt snap back to LKA steering.

### 7.4 Lane-Loss Fail-Safe

```
if lane_lost_frames > 0:
    raw_speed *= 0.7                        (slow down immediately)

if lane_lost_frames < lane_lost_threshold:
    raw_steer = last_known_steer            (hold last steering for curves)
else:
    raw_steer = 0.0                         (go straight after 8 frames lost)
```

### 7.5 TSR Speed Cap with Auto-Expiry

The speed cap is cleared if no sign is detected for `tsr_expiry_sec = 10 s`:

```
if now - tsr_last_seen_time > tsr_expiry_sec:
    tsr_speed_cap = None
```

When active:

```
speed_y = min(acc_speed_norm, tsr_speed_cap)
```

### 7.6 Command Serialisation

The final command is serialised as JSON for the ESP32:

```json
{"type": "auto_cmd", "x": steer_x, "y": speed_y}
```

Where:
- `x = clamp(steer_x, -1.0, 1.0)` — steering
- `y = clamp(speed_y,  0.0, 1.0)` — throttle

ESP32 motor mapping:
```
targetVL = jd.y + jd.x    (left wheels)
targetVR = jd.y - jd.x    (right wheels)
```

---

## 8. Pipeline Scheduling

```
Frame index mod 4:

  cycle 0: Lane + ACC
  cycle 1: Lane + TSR
  cycle 2: Lane + ACC + TLR
  cycle 3: Lane + Pothole
  even:     SignCV (every 2nd frame)
  all:      DecisionManager
```

At 25 fps, each module is serviced every ~100–160 ms.

---

## 9. Decision Priority (highest to lowest)

| Level | Module | Bypass smoothing? |
|-------|--------|-----------------|
| 1 | Traffic Light RED — full stop | Yes |
| 2 | Traffic Light YELLOW — 30% speed | Yes |
| 3 | Pothole avoidance — override steer | No (time-blended) |
| 4 | TSR speed cap | No |
| 5 | ACC throttle | No |
| 6 | Lane Keeping Assist — steering | No (EMA smoothed) |
| 7 | LDW warning flag | Flag only |

---

## 10. Adding a New Module

1. Create `adas/my_module.py` with a `detect(frame) -> MyResult` method.
2. Instantiate in `main.py` → `TARAAdas.__init__`.
3. Call on the appropriate frame index in `_process_frame`.
4. Pass result to `DecisionManager.update()` and handle priority in `decision_manager.py`.
