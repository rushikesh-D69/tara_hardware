# TÁRA Autonomous Car: Technical Architecture & Mathematics
*Notes for PowerPoint Presentation*

---

## 1. Drive Mechanics & Control Architecture
**Title:** Differential Drive System
*   **Concept:** TÁRA uses a differential drive chassis, meaning movement and steering are achieved by varying the independent speeds of the left and right wheels.
*   **Inputs:** The user's joystick provides continuous `X` (steering) and `Y` (throttle) coordinates in the range `[-1.0, 1.0]`.
*   **Target Velocity Math:** 
    *   The joystick combines with the `baseSpeed` slider (max absolute velocity in m/s) to generate independent target speeds:
    *   `Target V_Left  = (Y + X) * baseSpeed`
    *   `Target V_Right = (Y - X) * baseSpeed`
*   **Result:** Pushing the joystick diagonally forward-right increases the left wheel speed while decreasing the right, executing a smooth arc turn.

---

## 2. Speed Estimation (Encoders)
**Title:** Real-Time Velocity Estimation
*   **Sensors:** Optical/Slit Encoders mounted on the yellow TT motors.
*   **Constants:**
    *   `Wheel Radius (r) = 3.0 cm (0.03 m)`
    *   `Pulses Per Revolution (PPR) = (Motor PPR * Gear Ratio)`
    *   `Meters Per Pulse = (2 * π * r) / PPR`
*   **Delta Time (dt):** Calculated continuously at 50Hz (every 20ms).
*   **Speed Equation:** 
    *   The ESP32 hardware interrupts track the pulse count:
    *   `v_actual = ((Current Pulses - Previous Pulses) * Meters_Per_Pulse) / dt`
*   **Result:** We capture real-time linear speed for each wheel independently in **meters per second (m/s)**.

---

## 3. Robot Kinematics
**Title:** Forward Kinematics (How the Robot Moves)
*   **Concept:** Converting independent wheel speeds into the total motion of the robot chassis.
*   **Constants:** `Wheel Base (L) = 12.5 cm (0.125 m)` — The distance between the left and right wheels.
*   **Linear Velocity (m/s):** The average speed of the two wheels.
    *   `v_linear = (v_Left + v_Right) / 2`
*   **Angular Velocity (rad/s):** The rotation speed of the chassis.
    *   `v_angular = (v_Right - v_Left) / L`

---

## 4. Pose Estimation (Odometry & Dead-Reckoning)
**Title:** Odometry & Path Tracking
*   **Concept:** "Dead-Reckoning" is used to integrate the vehicle's speed and heading over time to estimate its exact [(X, Y)](file:///d:/UG/B.TECH/6th/openlab2/TARA-Tracking_Adaptive_Road_Autonomous_Car-main/Dashboard/ESP32-Web/ESP32-Web.ino#426-467) position on a 2D plane.
*   **Sensor Fusion (IMU):** While encoder differences *can* estimate heading, wheel slip causes drift. TÁRA fuses data from the **MPU6050 IMU's DMP (Digital Motion Processor)** to get highly accurate Yaw (Heading).
*   **Integration Equations (runs at 50Hz):**
    *   `Heading (θ) = IMU_Yaw * (π / 180)`
    *   `Position X = Position X + (v_linear * cos(θ) * dt)`
    *   `Position Y = Position Y + (v_linear * sin(θ) * dt)`
    *   `Distance Traveled = Distance Traveled + |v_linear * dt|`
*   **Dashboard Features:** The UI auto-scales to `cm` resolution and actively monitors the vector dot-product to dynamically erase the drawn path if the car reverses (`v_linear < 0`).

---

## 5. Closed-Loop Control System (PID)
**Title:** Error Correction & PID Tuning
*   **Concept:** A Proportional-Integral-Derivative (PID) controller eliminates the difference between the user's *Requested Speed* and the encoder's *Actual Speed*.
*   **Error Calculation:** 
    *   `Error = Target Velocity (m/s) - Actual Velocity (m/s)`
*   **Equations:**
    *   `P_term = Kp * Error` (Pushes hard when error is large)
    *   `I_term = Ki * ∫(Error * dt)` (Builds up to overcome physical friction/deadzones)
    *   `D_term = Kd * (dError / dt)` (Dampens the response to prevent overshoot)
    *   `Output PWM = P_term + I_term + D_term`
*   **Hardcoded Configuration:** Tuned specifically for yellow TT motors at 50Hz: `Kp=250`, `Ki=40`, `Kd=5`.

---

## 6. Power Management & Safety Mechanisms
**Title:** Electrical Integrity & AEB
*   **PWM Dead-Zone Clamping:** TT Motors stall below `PWM 160`. The controller automatically clamps any non-zero PID output to `[160, 225]` to prevent motor whining and stall currents.
*   **Brownout Prevention:** Prevents the ESP32 `5V` logic regulator from crashing due to instant 2A stall-current spikes by separating the `VM` (Motor Voltage) line to an external 7.4V LiPo battery. 
*   **Automatic Emergency Braking (AEB):** 
    *   The front Ultrasonic sensor calculates Time-To-Collision (`TTC = Distance / v_linear`).
    *   If `Distance < 15cm`, the hardware interrupts the motor queue, forcing `PWM = 0` instantly, bypassing user input.
*   **Watchdog:** If the WebSocket heartbeat is lost for `> 1000ms`, the car safely halts.
