/* =============================================
   TÁRA DASHBOARD — COMPLETE JAVASCRIPT
   WebSocket Communication + Telemetry Parsing
   ============================================= */

/* =========================================
   1. CONFIGURATION & DOM ELEMENTS
   ========================================= */

// CHANGE THIS TO YOUR ESP32's IP ADDRESS
const TARA_IP = "192.168.43.40";
const WEBSOCKET_URL = `ws://${TARA_IP}/ws`;

// --- Header ---
const statusBadge = document.getElementById('connection-status');
const btnEstop = document.getElementById('btn-estop');

// --- Panel 1: Joystick & Speed ---
const zone = document.getElementById('joystick-zone');
const stick = document.getElementById('joystick-stick');
const joyXText = document.getElementById('joy-x');
const joyYText = document.getElementById('joy-y');
const speedSlider = document.getElementById('speed-slider');
const speedValue = document.getElementById('speed-value');

// --- Panel 2: Orientation ---
const dataYaw = document.getElementById('data-yaw');
const dataRate = document.getElementById('data-rate');
const dataAccel = document.getElementById('data-accel');
const carModel = document.getElementById('car-model');

// --- Panel 3: Telemetry & ADAS ---
const dataSpeed = document.getElementById('data-speed');
const dataVtarget = document.getElementById('data-vtarget');
const dataYaw2 = document.getElementById('data-yaw2');
const dataAccel2 = document.getElementById('data-accel2');
const statusAcc = document.getElementById('status-acc');
const statusAeb = document.getElementById('status-aeb');
const dataLead = document.getElementById('data-lead');
const dataTtc = document.getElementById('data-ttc');

// --- Panel 4: Chart ---
const speedChartCanvas = document.getElementById('speed-chart');

// --- Panel 5: Camera Stream & Encoder ---
const piCameraStream = document.getElementById('pi-camera-stream');

const encLeft = document.getElementById('enc-left');
const encRight = document.getElementById('enc-right');
const encDistance = document.getElementById('enc-distance');

// --- Panel 6: Mini Map & Status ---
const minimapCanvas = document.getElementById('minimap-canvas');
const btnClearMap = document.getElementById('btn-clear-map');
const statusWifi = document.getElementById('status-wifi');
const statusWifiIcon = document.getElementById('status-wifi-icon');
const statusBattery = document.getElementById('status-battery');
const statusMotors = document.getElementById('status-motors');


/* =========================================
   2. WEBSOCKET MANAGEMENT
   ========================================= */
let ws;
let isConnected = false;

function connectWebSocket() {
    ws = new WebSocket(WEBSOCKET_URL);

    ws.onopen = () => {
        isConnected = true;
        statusBadge.innerHTML = '<span class="pulse-dot"></span> SYSTEM ONLINE';
        statusBadge.className = "badge connected";
        statusWifi.textContent = "Online";
        statusWifi.className = "status-val connected-text";
        statusWifiIcon.classList.add('online');
        console.log("[TÁRA] Connected to ESP32");
    };

    ws.onclose = () => {
        isConnected = false;
        statusBadge.innerHTML = '<span class="pulse-dot"></span> DISCONNECTED';
        statusBadge.className = "badge disconnected";
        statusWifi.textContent = "Offline";
        statusWifi.className = "status-val disconnected-text";
        statusWifiIcon.classList.remove('online');
        console.log("[TÁRA] Disconnected. Reconnecting in 2s...");
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (error) => {
        console.error("[TÁRA] WebSocket Error:", error);
        ws.close();
    };

    ws.onmessage = (event) => {
        handleTelemetry(event.data);
    };
}

connectWebSocket();

// Auto-connect to Pi Camera Stream
if (piCameraStream) {
    piCameraStream.src = `http://192.168.43.88:5000/video_feed`;
}


/* =========================================
   3. TELEMETRY PARSING ($TARA CSV PACKET)
   ========================================= */
/*
 * The ESP32 sends a comma-separated packet:
 * $TARA, v, v_target, yaw, yaw_rate, accel,
 *        lm_spd, rm_spd, enc_l, enc_r,
 *        dist_traveled, pos_x, pos_y, heading,
 *        lead, ttc, acc_status, aeb_status, bat
 *
 * Fields (0-indexed after split):
 *  [0]  = "$TARA"        (header tag)
 *  [1]  = v              (vehicle speed, m/s)
 *  [2]  = v_target       (target speed, m/s)
 *  [3]  = yaw            (yaw angle, degrees)
 *  [4]  = yaw_rate       (yaw rate, °/s)
 *  [5]  = accel          (acceleration, m/s²)
 *  [6]  = lm_spd         (left motor speed)
 *  [7]  = rm_spd         (right motor speed)
 *  [8]  = enc_l          (left encoder pulses)
 *  [9]  = enc_r          (right encoder pulses)
 *  [10] = dist_traveled  (odometry distance, m)
 *  [11] = pos_x          (odometry X position)
 *  [12] = pos_y          (odometry Y position)
 *  [13] = heading        (heading in degrees)
 *  [14] = lead           (lead vehicle distance, cm)
 *  [15] = ttc            (time to collision, s)
 *  [16] = acc_status     (0=off, 1=active)
 *  [17] = aeb_status     (0=off, 1=warning, 2=braking)
 *  [18] = bat            (battery voltage, V)
 */

function handleTelemetry(raw) {
    const data = raw.trim().split(',');

    // Validate packet header
    if (data[0] !== "SEN:" || data.length < 19) {
        // Fallback: try legacy "RAW" format for backward compatibility
        if (data[0] === "RAW" && data.length >= 3) {
            const yaw = parseFloat(data[1]).toFixed(2);
            const rate = parseFloat(data[2]).toFixed(2);
            dataYaw.textContent = `${yaw}°`;
            dataRate.textContent = `${rate}°/s`;
            // Apply yaw inversely to correct for visual layout (-yaw)
            carModel.style.transform = `rotate(${-yaw}deg)`;
        }
        return;
    }

    // Parse all fields
    const v = parseFloat(data[1]);
    const vTarget = parseFloat(data[2]);
    const yaw = parseFloat(data[3]);
    const yawRate = parseFloat(data[4]);
    const omega = parseFloat(data[5]);   // angular velocity rad/s (encoder-based)
    const lmSpd = parseFloat(data[6]);
    const rmSpd = parseFloat(data[7]);
    const encL = parseInt(data[8], 10);
    const encR = parseInt(data[9], 10);
    const distTrav = parseFloat(data[10]);
    const posX = parseFloat(data[11]);
    const posY = parseFloat(data[12]);
    const heading = parseFloat(data[13]);
    const lead = parseFloat(data[14]);
    const ttc = parseFloat(data[15]);
    const accStatus = parseInt(data[16], 10);
    const aebStatus = parseInt(data[17], 10);
    const bat = parseFloat(data[18]);

    // --- Update Panel 2: Orientation ---
    dataYaw.textContent = `${yaw.toFixed(2)}°`;
    dataRate.textContent = `${yawRate.toFixed(2)}°/s`;
    dataAccel.textContent = `${omega.toFixed(3)} r/s`;
    // Apply yaw inversely to correct for visual layout (-yaw)
    carModel.style.transform = `rotate(${-yaw}deg)`;

    // --- Update Panel 3: Telemetry ---
    dataSpeed.textContent = `${v.toFixed(2)} m/s`;
    dataVtarget.textContent = `${vTarget.toFixed(2)} m/s`;
    dataYaw2.textContent = `${yaw.toFixed(2)}°`;
    dataAccel2.textContent = `${omega.toFixed(3)} rad/s`;

    // ACC Status
    if (accStatus === 1) {
        statusAcc.textContent = "ACTIVE";
        statusAcc.className = "value badge active";
    } else {
        statusAcc.textContent = "Standby";
        statusAcc.className = "value badge standby";
    }

    // AEB Status
    if (aebStatus === 2) {
        statusAeb.textContent = "BRAKING!";
        statusAeb.className = "value badge active-danger";
    } else if (aebStatus === 1) {
        statusAeb.textContent = "WARNING";
        statusAeb.className = "value badge active";
    } else {
        statusAeb.textContent = "Standby";
        statusAeb.className = "value badge standby";
    }

    // Distance and TTC logic removed
    dataLead.textContent = `— cm`;
    dataTtc.textContent = `— s`;

    // --- Update Panel 4: Speed Chart ---
    pushChartData(lmSpd, rmSpd, v);

    // --- Update Panel 5: Camera Stream & Encoder ---
    // Camera connects automatically; no repetitive telemetry needed for it.

    encLeft.textContent = encL;
    encRight.textContent = encR;
    encDistance.textContent = distTrav.toFixed(2);

    // --- Update Panel 6: Mini Map & Status ---
    addMapPoint(posX, posY, heading, v);

    // Parse nav status (fields 19, 20 — extended packet)
    if (data.length >= 21) {
        const ns = parseInt(data[19], 10);
        const np = parseFloat(data[20]);
        updateNavDisplay(ns, np);
    }

    // Battery
    statusBattery.textContent = `${bat.toFixed(1)} V`;
    if (bat < 6.5) {
        statusBattery.className = "status-val disconnected-text";
    } else if (bat < 7.2) {
        statusBattery.style.color = "var(--warning)";
    } else {
        statusBattery.className = "status-val connected-text";
    }

    // Motor state
    if (Math.abs(lmSpd) > 0 || Math.abs(rmSpd) > 0) {
        statusMotors.textContent = "Running";
        statusMotors.className = "status-val connected-text";
    } else {
        statusMotors.textContent = "Idle";
        statusMotors.className = "status-val";
    }
}


/* =========================================
   4. JOYSTICK LOGIC (PRESERVED & ENHANCED)
   ========================================= */
const stickRadius = 25;
const maxDist = 90 - stickRadius;
const centerX = 90;
const centerY = 90;

let currentX = 0;
let currentY = 0;
let isDragging = false;

function moveStick(e) {
    if (!isDragging) return;
    e.preventDefault();

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const rect = zone.getBoundingClientRect();
    const dx = clientX - rect.left - centerX;
    const dy = clientY - rect.top - centerY;

    const distance = Math.min(Math.sqrt(dx * dx + dy * dy), maxDist);
    const angle = Math.atan2(dy, dx);

    const stickX = distance * Math.cos(angle);
    const stickY = distance * Math.sin(angle);

    stick.style.transform = `translate(${stickX}px, ${stickY}px)`;

    // Map to normalized -1.0 to 1.0 range for JSON commands
    currentX = parseFloat((stickX / maxDist).toFixed(2));
    currentY = parseFloat((-(stickY / maxDist)).toFixed(2));

    joyXText.textContent = currentX;
    joyYText.textContent = currentY;
}

function resetStick() {
    isDragging = false;
    stick.style.transform = 'translate(0px, 0px)';
    stick.style.transition = 'transform 0.15s ease-out';

    currentX = 0;
    currentY = 0;
    joyXText.textContent = "0";
    joyYText.textContent = "0";
}

function startDrag(e) {
    isDragging = true;
    stick.style.transition = 'none';
    moveStick(e);
}

// Mouse
zone.addEventListener('mousedown', startDrag);
document.addEventListener('mousemove', moveStick);
document.addEventListener('mouseup', resetStick);

// Touch
zone.addEventListener('touchstart', startDrag, { passive: false });
document.addEventListener('touchmove', moveStick, { passive: false });
document.addEventListener('touchend', resetStick);


/* =========================================
   5. SPEED SLIDER
   ========================================= */
let currentBaseSpeed = 150;  // PWM

speedSlider.addEventListener('input', (e) => {
    currentBaseSpeed = parseInt(e.target.value, 10);
    speedValue.textContent = currentBaseSpeed;
});

// Send speed change on release to avoid flooding
speedSlider.addEventListener('change', () => {
    sendCommand({ type: "speed", value: currentBaseSpeed });  // value in PWM
});

const pprLInput = document.getElementById('ppr-l-input');
const pprRInput = document.getElementById('ppr-r-input');
const btnUpdatePpr = document.getElementById('btn-update-ppr');

btnUpdatePpr.addEventListener('click', () => {
    let pprL = parseFloat(pprLInput.value);
    let pprR = parseFloat(pprRInput.value);
    if (!isNaN(pprL) && pprL > 0 && !isNaN(pprR) && pprR > 0) {
        sendCommand({ type: "ppr", value_l: pprL, value_r: pprR });
    }
});


/* =========================================
   6. EMERGENCY STOP
   ========================================= */
let eStopEngaged = false;

btnEstop.addEventListener('click', () => {
    eStopEngaged = !eStopEngaged;
    sendCommand({ type: "estop", state: eStopEngaged });

    // Visual feedback
    if (eStopEngaged) {
        btnEstop.classList.add('engaged');
        btnEstop.style.transform = 'scale(0.95)';
        console.warn("[TÁRA] EMERGENCY STOP ENGAGED!");
    } else {
        btnEstop.classList.remove('engaged');
        btnEstop.style.transform = '';
        console.info("[TÁRA] Emergency stop released.");
    }
});


/* =========================================
   7. DATA TRANSMISSION
   ========================================= */
function sendCommand(cmd) {
    if (isConnected && ws.readyState === WebSocket.OPEN) {
        if (typeof cmd === 'string') {
            ws.send(cmd);
        } else {
            ws.send(JSON.stringify(cmd));
        }
    }
}

// Send telemetry requests or drive commands
let heartbeatCounter = 0;
setInterval(() => {
    if (isConnected && ws.readyState === WebSocket.OPEN) {
        if (!autoMode) {
            sendCommand(`CMD:${currentX},${currentY},${eStopEngaged ? 4 : 0}`);
        } else {
            // Heartbeat in auto mode at 2Hz (every 10 ticks)
            heartbeatCounter++;
            if (heartbeatCounter >= 10) {
                sendCommand({ type: "heartbeat" });
                heartbeatCounter = 0;
            }
        }
    }
}, 50);


/* =========================================
   8. CHART.JS — SPEED TIME-SERIES GRAPH
   ========================================= */
const MAX_CHART_POINTS = 60;
const chartLabels = [];
const chartLeftSpeed = [];
const chartRightSpeed = [];
const chartRobotSpeed = [];
let chartTimeCounter = 0;

const speedChart = new Chart(speedChartCanvas, {
    type: 'line',
    data: {
        labels: chartLabels,
        datasets: [
            {
                label: 'Left Motor',
                data: chartLeftSpeed,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.35,
                fill: true,
            },
            {
                label: 'Right Motor',
                data: chartRightSpeed,
                borderColor: '#22d3ee',
                backgroundColor: 'rgba(34, 211, 238, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.35,
                fill: true,
            },
            {
                label: 'Robot Speed',
                data: chartRobotSpeed,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.08)',
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.35,
                fill: true,
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        interaction: {
            intersect: false,
            mode: 'index',
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#8b97b0',
                    font: { family: "'Inter', sans-serif", size: 11 },
                    boxWidth: 12,
                    boxHeight: 3,
                    padding: 15,
                    usePointStyle: false,
                }
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                titleColor: '#f1f5f9',
                bodyColor: '#8b97b0',
                borderColor: 'rgba(56, 189, 248, 0.2)',
                borderWidth: 1,
                padding: 10,
                bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            }
        },
        scales: {
            x: {
                display: true,
                ticks: { color: '#4b5672', font: { size: 9 }, maxTicksLimit: 8 },
                grid: { color: 'rgba(56, 189, 248, 0.04)' }
            },
            y: {
                display: true,
                ticks: { color: '#4b5672', font: { size: 10 } },
                grid: { color: 'rgba(56, 189, 248, 0.04)' },
                beginAtZero: true,
            }
        }
    }
});

function pushChartData(leftSpd, rightSpd, robotSpd) {
    chartTimeCounter++;
    chartLabels.push(chartTimeCounter + 's');
    chartLeftSpeed.push(leftSpd);
    chartRightSpeed.push(rightSpd);
    chartRobotSpeed.push(robotSpd);

    // Sliding window
    if (chartLabels.length > MAX_CHART_POINTS) {
        chartLabels.shift();
        chartLeftSpeed.shift();
        chartRightSpeed.shift();
        chartRobotSpeed.shift();
    }

    speedChart.update();
}


/* =========================================
   9. (ULTRASONIC DISTANCE REMOVED)
   ========================================= */


/* =========================================
   10. MINI MAP (ODOMETRY PATH CANVAS)
   ========================================= */
const mapCtx = minimapCanvas.getContext('2d');
const MAP_SIZE = 280;
const MAP_SCALE = 4; // pixels per real-world unit (adjust based on your odometry scale)
const pathHistory = [];

// Center of the canvas in pixel coords
const mapCenterX = MAP_SIZE / 2;
const mapCenterY = MAP_SIZE / 2;

function drawMapGrid() {
    mapCtx.fillStyle = '#060810';
    mapCtx.fillRect(0, 0, MAP_SIZE, MAP_SIZE);

    // Grid lines
    mapCtx.strokeStyle = 'rgba(56, 189, 248, 0.06)';
    mapCtx.lineWidth = 0.5;
    const gridSpacing = 20;
    for (let i = 0; i <= MAP_SIZE; i += gridSpacing) {
        mapCtx.beginPath();
        mapCtx.moveTo(i, 0);
        mapCtx.lineTo(i, MAP_SIZE);
        mapCtx.stroke();
        mapCtx.beginPath();
        mapCtx.moveTo(0, i);
        mapCtx.lineTo(MAP_SIZE, i);
        mapCtx.stroke();
    }

    // Origin crosshair
    mapCtx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
    mapCtx.lineWidth = 1;
    mapCtx.beginPath();
    mapCtx.moveTo(mapCenterX, 0);
    mapCtx.lineTo(mapCenterX, MAP_SIZE);
    mapCtx.stroke();
    mapCtx.beginPath();
    mapCtx.moveTo(0, mapCenterY);
    mapCtx.lineTo(MAP_SIZE, mapCenterY);
    mapCtx.stroke();
}

function addMapPoint(posX, posY, headingDeg, v_linear) {
    if (isNaN(posX) || isNaN(posY)) return;

    if (pathHistory.length === 0) {
        pathHistory.push({ x: posX, y: posY, heading: headingDeg });
        renderMap();
        return;
    }

    const last = pathHistory[pathHistory.length - 1];
    const dx = posX - last.x;
    const dy = posY - last.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    // If reversing (> 1cm/s back), erase the line by popping points
    if (v_linear < -0.01) {
        if (dist > 0.01 && pathHistory.length > 1) { // 1 cm threshold for popping
            pathHistory.pop();
            renderMap();
        }
        return;
    }

    // Moving forward: only add point if moved > 1cm (0.01m)
    if (dist > 0.01) {
        // Neglect small angle changes for map straightness (< 5 degrees)
        let finalHeading = headingDeg;
        let dHeading = Math.abs((headingDeg - last.heading + 180) % 360 - 180);
        if (dHeading < 5.0) {
            finalHeading = last.heading; // Lock angle to previous
        }

        pathHistory.push({ x: posX, y: posY, heading: finalHeading });
        if (pathHistory.length > 500) pathHistory.shift();
        renderMap();
    }
}

function renderMap() {
    drawMapGrid();

    if (pathHistory.length === 0) return;

    // Auto-center: compute bounding box of the path
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const pt of pathHistory) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const midPtX = (minX + maxX) / 2;
    const midPtY = (minY + maxY) / 2;

    // Dynamic scale to fit path within canvas with padding
    const padding = 40;
    const scale = Math.min(
        (MAP_SIZE - padding * 2) / rangeX,
        (MAP_SIZE - padding * 2) / rangeY,
        4000 // Huge max zoom so 5cm moves fill the screen
    );

    function toCanvasX(px) { return mapCenterX + (px - midPtX) * scale; }
    function toCanvasY(py) { return mapCenterY - (py - midPtY) * scale; } // Invert Y

    // Draw path trail
    mapCtx.beginPath();
    mapCtx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
    mapCtx.lineWidth = 2;
    mapCtx.lineJoin = 'round';
    mapCtx.lineCap = 'round';

    for (let i = 0; i < pathHistory.length; i++) {
        const cx = toCanvasX(pathHistory[i].x);
        const cy = toCanvasY(pathHistory[i].y);
        if (i === 0) {
            mapCtx.moveTo(cx, cy);
        } else {
            mapCtx.lineTo(cx, cy);
        }
    }
    mapCtx.stroke();

    // Draw path points (fading older ones)
    for (let i = 0; i < pathHistory.length; i++) {
        const cx = toCanvasX(pathHistory[i].x);
        const cy = toCanvasY(pathHistory[i].y);
        const alpha = 0.15 + 0.85 * (i / pathHistory.length);
        mapCtx.fillStyle = `rgba(56, 189, 248, ${alpha})`;
        mapCtx.beginPath();
        mapCtx.arc(cx, cy, 1.5, 0, Math.PI * 2);
        mapCtx.fill();
    }

    // Start marker
    const startCx = toCanvasX(pathHistory[0].x);
    const startCy = toCanvasY(pathHistory[0].y);
    mapCtx.fillStyle = 'rgba(34, 211, 238, 0.5)';
    mapCtx.beginPath();
    mapCtx.arc(startCx, startCy, 4, 0, Math.PI * 2);
    mapCtx.fill();

    // Draw current position with heading arrow
    const last = pathHistory[pathHistory.length - 1];
    const lx = toCanvasX(last.x);
    const ly = toCanvasY(last.y);

    // Current position dot
    mapCtx.fillStyle = '#10b981';
    mapCtx.beginPath();
    mapCtx.arc(lx, ly, 5, 0, Math.PI * 2);
    mapCtx.fill();
    mapCtx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
    mapCtx.lineWidth = 1;
    mapCtx.beginPath();
    mapCtx.arc(lx, ly, 9, 0, Math.PI * 2);
    mapCtx.stroke();

    // Heading arrow
    // CSS rotate is CW+; yaw is CCW+. Negate so arrow points in actual heading direction.
    const headingRad = (last.heading || 0) * (Math.PI / 180) + Math.PI / 2;
    const arrowLen = 18;
    const ax = lx + arrowLen * Math.cos(headingRad);
    const ay = ly - arrowLen * Math.sin(headingRad);

    mapCtx.strokeStyle = '#10b981';
    mapCtx.lineWidth = 2.5;
    mapCtx.beginPath();
    mapCtx.moveTo(lx, ly);
    mapCtx.lineTo(ax, ay);
    mapCtx.stroke();

    // Arrowhead
    const headSize = 6;
    const angle1 = headingRad + Math.PI + Math.PI / 6;
    const angle2 = headingRad + Math.PI - Math.PI / 6;
    mapCtx.fillStyle = '#10b981';
    mapCtx.beginPath();
    mapCtx.moveTo(ax, ay);
    mapCtx.lineTo(ax + headSize * Math.cos(angle1), ay - headSize * Math.sin(angle1));
    mapCtx.lineTo(ax + headSize * Math.cos(angle2), ay - headSize * Math.sin(angle2));
    mapCtx.closePath();
    mapCtx.fill();
}

// Clear map button
btnClearMap.addEventListener('click', () => {
    pathHistory.length = 0;
    drawMapGrid();
});

// Initial render
drawMapGrid();


/* =========================================
   11. DEMO / SIMULATION MODE
   ========================================= */
/*
 * Uncomment the block below to generate synthetic telemetry
 * for testing the dashboard without a real ESP32 connection.
 *
 * let simAngle = 0;
 * let simDist = 0;
 * let simX = 0, simY = 0;
 * setInterval(() => {
 *     simAngle += 0.8;
 *     simDist += 0.02;
 *     simX += Math.cos(simAngle * Math.PI / 180) * 0.1;
 *     simY += Math.sin(simAngle * Math.PI / 180) * 0.1;
 *     const lm = 120 + Math.sin(simAngle * 0.05) * 50;
 *     const rm = 130 + Math.cos(simAngle * 0.04) * 45;
 *     const v = ((lm + rm) / 2 / 255).toFixed(2);
 *     const lead = 80 + Math.sin(simAngle * 0.02) * 70;
 *     const aeb = lead < 20 ? 2 : lead < 50 ? 1 : 0;
 *     const packet = `$TARA,${v},0.50,${(simAngle % 360).toFixed(1)},1.20,0.15,${lm.toFixed(0)},${rm.toFixed(0)},${Math.floor(simDist*100)},${Math.floor(simDist*98)},${simDist.toFixed(2)},${simX.toFixed(3)},${simY.toFixed(3)},${(simAngle % 360).toFixed(1)},${lead.toFixed(1)},2.50,${aeb > 0 ? 1 : 0},${aeb},7.40`;
 *     handleTelemetry(packet);
 * }, 200);
 */


/* =========================================
   12. AUTONOMOUS MODE
   ========================================= */
let autoMode = false;

// DOM refs
const btnModeToggle   = document.getElementById('btn-mode-toggle');
const modeLabelManual = document.getElementById('mode-label-manual');
const modeLabelAuto   = document.getElementById('mode-label-auto');
const autoPanel       = document.getElementById('panel-autonomous');
const autoControlsEl  = document.getElementById('auto-controls');
const navStatusBadge  = document.getElementById('nav-status-badge');
const navStatusDetail = document.getElementById('nav-status-detail');
const navProgressFill = document.getElementById('nav-progress-fill');
const navProgressPct  = document.getElementById('nav-progress-pct');

// --- Mode toggle ---
btnModeToggle.addEventListener('change', () => {
    autoMode = btnModeToggle.checked;

    if (autoMode) {
        // Switch to AUTO
        autoPanel.classList.add('auto-active');
        autoControlsEl.classList.remove('disabled');
        zone.classList.add('joystick-disabled');
        currentX = 0; currentY = 0;
        sendCommand({ type: 'nav_stop' });
        navStatusBadge.textContent = 'IDLE';
        navStatusBadge.className = 'nav-status-badge';
        navStatusDetail.textContent = 'Auto mode — waiting for command';
    } else {
        // Switch to MANUAL
        autoPanel.classList.remove('auto-active');
        autoControlsEl.classList.add('disabled');
        zone.classList.remove('joystick-disabled');
        sendCommand({ type: 'nav_stop' });
        navStatusBadge.textContent = 'IDLE';
        navStatusBadge.className = 'nav-status-badge';
        navStatusDetail.textContent = 'Manual joystick active';
        navProgressFill.style.width = '0%';
        navProgressPct.textContent = '0%';
    }
});

// Start in manual: controls grid dimmed
autoControlsEl.classList.add('disabled');

// --- GoTo speed slider ---
const navGotoSpeedSlider = document.getElementById('nav-goto-speed');
const navGotoSpeedVal    = document.getElementById('nav-goto-speed-val');
navGotoSpeedSlider.addEventListener('input', () => {
    navGotoSpeedVal.textContent = (parseInt(navGotoSpeedSlider.value) / 100).toFixed(2);
});

// --- GoTo button ---
document.getElementById('btn-goto').addEventListener('click', () => {
    if (!isConnected || !autoMode) return;
    const cm    = parseFloat(document.getElementById('nav-dist-cm').value);
    const speed = parseInt(navGotoSpeedSlider.value) / 100;
    sendCommand({ type: 'goto', dist: cm / 100.0, speed });
    navStatusBadge.textContent  = 'GOTO';
    navStatusBadge.className    = 'nav-status-badge badge-goto';
    navStatusDetail.textContent = `Driving ${cm} cm forward…`;
});

// --- Turn buttons ---
document.getElementById('btn-turn-left').addEventListener('click', () => {
    if (!isConnected || !autoMode) return;
    const deg = parseFloat(document.getElementById('nav-turn-deg').value);
    sendCommand({ type: 'turn', angle: deg, speed: 0.20 });
    navStatusBadge.textContent  = 'TURN';
    navStatusBadge.className    = 'nav-status-badge badge-turn';
    navStatusDetail.textContent = `Turning left ${deg}°…`;
});

document.getElementById('btn-turn-right').addEventListener('click', () => {
    if (!isConnected || !autoMode) return;
    const deg = parseFloat(document.getElementById('nav-turn-deg').value);
    sendCommand({ type: 'turn', angle: -deg, speed: 0.20 });
    navStatusBadge.textContent  = 'TURN';
    navStatusBadge.className    = 'nav-status-badge badge-turn';
    navStatusDetail.textContent = `Turning right ${deg}°…`;
});

// --- Abort button ---
document.getElementById('btn-nav-abort').addEventListener('click', () => {
    sendCommand({ type: 'nav_stop' });
    navStatusBadge.textContent  = 'IDLE';
    navStatusBadge.className    = 'nav-status-badge';
    navStatusDetail.textContent = 'Navigation aborted';
    navProgressFill.style.width = '0%';
    navProgressPct.textContent  = '0%';
});

// --- Sequence Builder ---
const seqSteps       = [];
const seqStepType    = document.getElementById('seq-step-type');
const seqTurnDir     = document.getElementById('seq-turn-dir');
const seqStepValue   = document.getElementById('seq-step-value');
const seqStepUnit    = document.getElementById('seq-step-unit');
const seqStepsList   = document.getElementById('seq-steps-list');
const seqEmptyEl     = document.getElementById('seq-empty');
const btnSeqAdd      = document.getElementById('btn-seq-add');
const btnSeqExecute  = document.getElementById('btn-seq-execute');
const btnSeqClear    = document.getElementById('btn-seq-clear');

seqStepType.addEventListener('change', () => {
    const isGoto = seqStepType.value === 'goto';
    seqStepUnit.textContent = isGoto ? 'cm' : '°';
    seqStepValue.value      = isGoto ? '20' : '90';
    seqTurnDir.style.display = isGoto ? 'none' : 'inline-block';
});

function renderSeqSteps() {
    // Remove existing step items
    seqStepsList.querySelectorAll('.seq-step-item').forEach(el => el.remove());
    seqEmptyEl.style.display  = seqSteps.length === 0 ? 'block' : 'none';
    btnSeqExecute.disabled    = seqSteps.length === 0;

    seqSteps.forEach((step, i) => {
        const el    = document.createElement('div');
        el.className = 'seq-step-item';
        const label = step.type === 'goto'
            ? `▶ GoTo ${(step.dist * 100).toFixed(0)} cm @ ${step.speed.toFixed(2)} m/s`
            : `↺ Turn ${step.angle > 0 ? 'L' : 'R'} ${Math.abs(step.angle)}°`;
        el.innerHTML = `
            <span class="seq-step-icon">${i + 1}</span>
            <span class="seq-step-label">${label}</span>
            <button class="seq-step-remove" onclick="removeSeqStep(${i})">✕</button>
        `;
        seqStepsList.insertBefore(el, seqEmptyEl);
    });
}

btnSeqAdd.addEventListener('click', () => {
    if (seqSteps.length >= 10) return;
    const type  = seqStepType.value;
    const val   = parseFloat(seqStepValue.value);
    const speed = parseInt(navGotoSpeedSlider.value) / 100;

    if (type === 'goto') {
        seqSteps.push({ type: 'goto', dist: val / 100.0, speed });
    } else {
        const sign = seqTurnDir.value === 'right' ? -1 : 1;
        seqSteps.push({ type: 'turn', angle: sign * val, speed: 0.15 });
    }
    renderSeqSteps();
});

window.removeSeqStep = function(i) {
    seqSteps.splice(i, 1);
    renderSeqSteps();
};

btnSeqExecute.addEventListener('click', () => {
    if (!isConnected || !autoMode || seqSteps.length === 0) return;
    sendCommand({ type: 'sequence', steps: seqSteps });
    navStatusBadge.textContent  = 'SEQ';
    navStatusBadge.className    = 'nav-status-badge badge-seq';
    navStatusDetail.textContent = `Running ${seqSteps.length}-step sequence…`;
});

btnSeqClear.addEventListener('click', () => {
    seqSteps.length = 0;
    renderSeqSteps();
});

// --- Nav display (called from telemetry) ---
function updateNavDisplay(status, progress) {
    const pct = Math.round(progress * 100);
    navProgressFill.style.width = pct + '%';
    navProgressPct.textContent  = pct + '%';

    if (status === 0) {
        navStatusBadge.textContent  = 'IDLE';
        navStatusBadge.className    = 'nav-status-badge';
        navStatusDetail.textContent = autoMode ? 'Waiting for command' : 'Manual joystick active';
    } else if (status === 1) {
        navStatusBadge.textContent  = 'GOTO';
        navStatusBadge.className    = 'nav-status-badge badge-goto';
        navStatusDetail.textContent = `Driving… ${pct}% complete`;
    } else if (status === 2) {
        navStatusBadge.textContent  = 'TURNING';
        navStatusBadge.className    = 'nav-status-badge badge-turn';
        navStatusDetail.textContent = `Turning… ${pct}% complete`;
    }
}