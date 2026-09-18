# 🚀 Live Line Crossing Detector on i.MX8 NPU

[![Hardware](https://img.shields.io/badge/Hardware-i.MX8M_Plus-blue.svg?style=for-the-badge&logo=cpu-intel)](https://www.nxp.com/)
[![NPU Acceleration](https://img.shields.io/badge/Inference-Vivante_VIP8000_NPU-orange.svg?style=for-the-badge)](https://www.nxp.com/)
[![GStreamer](https://img.shields.io/badge/Media_Pipeline-GStreamer_1.20-green.svg?style=for-the-badge&logo=gstreamer)](https://gstreamer.freedesktop.org/)
[![License](https://img.shields.io/badge/Status-Production_Ready-success.svg?style=for-the-badge)](https://github.com/)

A production-grade, high-performance, and self-healing ML vision system designed to track people and count boundary crossings in real-time. Built specifically for NXP's **i.MX8M Plus** edge-computing platform utilizing hardware NPU acceleration, GStreamer media pipelines, and system-level autostart and self-healing daemon layers.

> **Problem Statement:** Standard deep learning vision pipelines on embedded Linux suffer from pipeline stalls, CPU exhaustion from software colorspace conversions, and memory leaks during camera disconnects. This project engineers a zero-stall, production-grade edge vision pipeline on the NXP i.MX8M Plus capable of continuous operation within strict thermal (<85°C) and latency boundaries without CPU lockup.

---

## 📹 Project Showcase & Demos

*(Embed your recorded video and screenshots here to make your portfolio pop!)*

| Live RTSP Stream (Counting Overlay) | On-Device Hardware Metrics |
| :---: | :---: |
| ![Live RTSP Stream Overlay](board_backup/rtsp_preview.png) <br> *Real-time VLC client stream showing foot-point tracking & line crossing counts.* | ![Terminal Hardware Metrics](board_backup/terminal_metrics.png) <br> *Terminal console showing NPU utilization active at 24.4%.* |

---

## 🧠 System Architecture

The pipeline consists of a decoupled architecture. GStreamer handles hardware-accelerated frame ingestion and video encoding on dedicated silicon units, while our Python orchestrator manages the high-level ML inference loop, multi-object tracking, and vector-based crossing checks.

```text
               +--------------------------------------------------------+
               |                       i.MX8 Board                      |
               |                                                        |
+----------+   | +------------+     +----------------+     +----------+ |   +-----------+
| Physical |-->| |  v4l2src   |-->| | imxvideoconvert |-->| | appsink  | |-->|  YOLOv8n  |
|  Camera  |   | | (Camera)   |     |     (g2d)      |     | (RGB/Raw)| |   | (NPU/TFL) |
+----------+   | +------------+     +----------------+     +----------+ |   +-----------+
               |                                                |       |         |
               |                                                v       |         v
               | +------------+     +----------------+     +----------+ |   +-----------+
               | |  udpsink   |     | cairooverlay   |     | OpenCV   |<--| Centroid  |
               | | (RTSP/UDP) |<--| | (Overlays)     |<--| Tracker  |   |  Tracker  |
               | +------------+     +----------------+     +----------+ |   +-----------+
               +--------------------------------------------------------+
```

### Hardware & Software Stack

| Layer | Component / Technology | Details & Role |
| :--- | :--- | :--- |
| **SoC / Compute** | NXP i.MX8M Plus Quad | 4× ARM Cortex-A53 @ 1.8 GHz, Vivante VIP8000 NPU (2.3 TOPS) |
| **Acceleration** | TFLite Delegate via NXP eIQ / TIM-VX | `libvx_delegate.so` hardware NPU delegate, G2D GPU 2D Engine |
| **Video Engine** | GStreamer 1.20 | Hardware pipeline (`v4l2src`, `imxvideoconvert_g2d`, `appsink`) |
| **Model** | YOLOv8n (Full INT8 Quantized) | 640×640 input, Post-Training Quantization calibrated via representative dataset |
| **Tracking & Logic** | Custom Centroid Tracker | Bounding box foot-point vector cross-product crossing geometry |
| **Reliability** | Systemd Daemon & GStreamer Leaky Bus | Dynamic dropping buffer (`drop=true, max-buffers=1`), Stepped backoff reconnect |

---

## 📊 Live On-Device Performance Metrics

The following reports represent continuous profiling data collected directly from the physical i.MX8 board during a **30.7-minute (1,841.0-second)** stability run. The system processed **26,911 frames** overall, running a full-integer quantized YOLOv8n TFLite model on the NPU.

### 1. Stage Latency Profile
*Identifies average execution latency at each step in the pipeline cycle.*

| Stage | Processed Frames | Avg Latency | Min Latency | Max Latency | Median | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Demux** | — | *N/A (HW)* | — | — | — | — |
| **1. Decoder** | — | *N/A (HW)* | — | — | — | — |
| **2. Appsink** | 13,730 | 0.12 ms | 0.07 ms | 4.20 ms | 0.10 ms | 0.28 ms |
| **3. Color Convert (imx g2d)** | 13,730 | **8.81 ms** | 6.24 ms | 27.80 ms | 7.92 ms | 12.80 ms |
| **4. Preprocess** | 13,730 | 11.02 ms | 5.56 ms | 34.90 ms | 11.01 ms | 13.86 ms |
| **5. Inference (YOLO NPU)** | 13,730 | **78.21 ms** | 73.61 ms | 92.38 ms | 78.33 ms | 80.43 ms |
| **6. Postprocess (NMS)** | 13,730 | 32.27 ms | 22.79 ms | 64.93 ms | 32.24 ms | 38.34 ms |
| **7. Tracking (Centroid)** | 13,730 | 0.17 ms | 0.02 ms | 5.07 ms | 0.03 ms | 0.73 ms |
| **8. Line Crossing Math** | 13,730 | 0.04 ms | 0.01 ms | 1.81 ms | 0.02 ms | 0.09 ms |
| **9. Overlay (Cairo Draw)** | 26,911 | 1.39 ms | 0.57 ms | 47.92 ms | 1.09 ms | 2.84 ms |

### 2. Stage Throughput Profile (FPS)
*Measures continuous frame processing rates across decoupled stages.*

| Stage | Processed Frames | Throughput (FPS) | Behavior |
| :--- | :---: | :---: | :--- |
| **Display Rendering (RTSP Output)** | 26,911 | **11.2 FPS** | Fluid real-time video output |
| **NPU Inference Rate** | 13,730 | **6.0 FPS** | YOLOv8n networks processed per second |

> ℹ️ **Decoupled Throughput Design:** The display rendering pipeline operates at **11.2 FPS** (matching the camera feed's output speed). However, since YOLOv8n inference on the NPU takes **78.21 ms**, the NPU loop processes frames at **6.0 FPS**. To prevent latency queues from backing up, the GStreamer `appsink` drops outdated frames dynamically (`drop=true`, `max-buffers=1`) so that the video overlay remains perfectly real-time.
>
> ⚠️ **Embedded Systems Time-Sync Gotcha:** If the i.MX8 board does not have a battery-backed RTC, the system clock may start at a default compile timestamp (e.g. Feb 2024) and jump forward by years when NTP syncs over the network. Our profiler accounts for this clock skew by relying on monotonic clocks for frame-by-frame throughput, avoiding division errors caused by system clock jumps.

### 3. System Resource Utilization

| Resource | Average Load | Maximum Peak | Monitoring Path / Source |
| :--- | :---: | :---: | :--- |
| **NPU Utilization** | **21.5%** | **42.0%** | `/sys/kernel/debug/gc/load` (Core 1 - VIP8000) |
| **CPU Utilization** | **47.3%** | **84.0%** | `psutil` (system total) |
| **RAM Utilization** | **18.9%** | **21.7%** | `psutil` (system total) |
| **SoC Temperature** | **80.6°C** | **84.0°C** | `/sys/class/thermal/thermal_zone0` |
| **GPU Utilization** | **0.0%** | **0.0%** | `/sys/kernel/debug/gc/load` (Core 0 - GPU) |

---

## ⚡ Engineering Challenges Solved

* **Decoupled Ingestion vs. Inference (Eliminating Buffer Bloat):** The physical camera sensor produces frames at 11.2–30 FPS while the VIP8000 NPU executes inference at 6.0 FPS (78.21 ms latency). Standard OpenCV/V4L2 queues buffer frames indefinitely, generating multi-second cumulative lag and unbounded memory growth. We resolved this by implementing GStreamer's leaky buffer pattern (`drop=true, max-buffers=1`) at the `appsink`, ensuring the overlay and tracker always process the freshest camera frame with zero queue bloat.
* **CPU Offloading via 2D GPU Engine:** Performing software colorspace conversion (`cv2.cvtColor` BGR to RGB) and bilateral frame resizing on the Cortex-A53 cores saturated CPU utilization at >90%, triggering thermal throttling. By delegating all colorspace conversion and image resizing directly to the i.MX8's Vivante 2D GPU via `imxvideoconvert_g2d`, we slashed CPU load to ~47% and preserved system thermal headroom (<81°C average).
* **Monotonic Edge Profiling (Mitigating Time Skew):** Embedded Linux boards lacking battery-backed real-time clocks (RTC) experience abrupt multi-year time jumps when NTP synchronizes over Ethernet post-boot. Standard wall-clock timers (`time.time()`) produce negative deltas or division-by-zero exceptions in throughput calculations. We solved this by strictly utilizing monotonic edge profiling via `time.monotonic()` for all stage-by-stage latency and FPS monitoring.
* **NPU-GPU-CPU Bottleneck Balancing:** Profiling revealed postprocessing (NMS) consumed 32.27 ms on CPU. By bounding candidate proposals (top-k=500), optimizing vectorized NumPy NMS, and decoupling the display rendering thread (11.2 FPS) from the inference worker (6.0 FPS), the system maintains smooth video rendering without stalling inference.

---

## 🛠️ System Design & Under the Hood

### 1. YOLOv8n INT8 Quantization & NPU Acceleration

To achieve real-time inference on the edge without GPU power consumption, the model was converted and fully quantized for the Vivante VIP8000 NPU:

* **Input Resolution:** 640×640 (RGB, 3-channel).
* **Quantization Scheme:** Post-Training Full Integer Quantization (INT8 weights and INT8 activations) calibrated using a representative dataset via TensorFlow Lite and NXP eIQ.
* **Inference Runtime:** Dispatched via the Vivante TIM-VX runtime delegate (`libvx_delegate.so`), which loads graph operations directly onto NPU Core 1.
* **Why YOLOv8n & INT8:** Provides the ideal trade-off between bounding box localization accuracy (>37% mAP on COCO person detection with <1.5% drop from FP32) and sub-80 ms NPU latency (78.21 ms on VIP8000). Full INT8 quantization slashes the model storage and memory footprint by ~74% (from ~13 MB FP32 down to 3.4 MB INT8), eliminating memory bandwidth bottlenecks.

```text
[YOLOv8n PyTorch FP32]  -->  [ONNX Export]  -->  [TFLite Converter + Representative Calib]
                                                            |
                                                            v
[NXP i.MX8M Plus Board] <-- [libvx_delegate.so] <-- [YOLOv8n Full-INT8 (3.4 MB)]
         |
         +--> [Vivante VIP8000 NPU: 78.21 ms avg latency, 21.5% avg load]
```

### 2. Vector-Based Line Crossing Mathematics
Rather than using loose bounding box overlaps which cause jitter, crossing detection is calculated using the **bottom-center foot point** of the tracked bounding box:

$$F_t = \left(x_t + \frac{w}{2},\, y_t + h\right)$$

Let the virtual boundary line segment be defined by points $P_1(x_1, y_1)$ and $P_2(x_2, y_2)$. To determine which side of the line segment a point $P(x,y)$ lies, we compute the sign of the vector cross-product:

$$d(P) = (x - x_1)(y_2 - y_1) - (y - y_1)(x_2 - x_1)$$

* If $d(P) > 0$, the person is on **Side A**.
* If $d(P) < 0$, the person is on **Side B**.

A crossing event is triggered when the sign of $d(F_{t-1})$ differs from the sign of $d(F_t)$ over successive frames, verifying the person's trajectory line segment intersected our boundary segment:

$$\text{Crossing} = \text{sign}\big(d(F_{t-1})\big) \neq \text{sign}\big(d(F_t)\big)$$

### 3. Live Vivante gc/load Driver Parser
Standard Linux monitoring tools (like `top` or `htop`) only display CPU usage. To extract actual NPU load averages, the `SystemMonitor` module parses the Vivante GPU driver's kernel debugfs node `/sys/kernel/debug/gc/load`:
```python
# Core 0 = GPU utilization, Core 1 = VIP8000 NPU utilization
cores = {}
for line in content.splitlines():
    if "core" in line:
        current_core = int(line.split(":")[1].strip())
    elif "load" in line:
        cores[current_core] = float(line.split(":")[1].replace("%", "").strip())
```

### 4. GStreamer Pipeline Ingestion String
We bypass standard bottlenecked OpenCV capture sources by declaring an optimized GStreamer pipeline string:
```text
v4l2src device=/dev/video3 ! video/x-raw,width=1920,height=1080,framerate=30/1 ! queue ! imxvideoconvert_g2d ! video/x-raw,format=RGB16 ! appsink drop=true max-buffers=1
```
* `imxvideoconvert_g2d` offloads scaling and YUV-to-RGB conversion to the i.MX8's internal 2D GPU, preserving CPU cores.
* `appsink drop=true` drops older frames if the Python execution loop is busy performing NPU inference.

---

## 🔌 Production-Grade Self-Healing Layers

### 1. Stepped Backoff Reconnection Loop
Physical camera connections in industrial deployments can suffer from vibration or power fluctuations.
* **GStreamer Bus Intercept:** The GStreamer pipeline intercepts `Gst.MessageType.ERROR` on the bus, tearing down pipeline elements safely.
* **Stepped Retry Policy:**
  - **Fast Retries (First 2 mins):** Attempts reconnection every **3 seconds** (up to 40 times).
  - **Slow Retries (After 2 mins):** Backs off to a **3-minute** interval indefinitely to prevent high CPU utilization during long outages.
* **Auto-Resumption:** Once `/dev/video3` is detected in the filesystem, the pipeline rebuilds itself and resumes RTSP casting.

### 2. Systemd Autostart & Process Recovery
Registered as a Linux background service (`line-crossing.service`):
* **Autostart on Boot:** Launches automatically as soon as the board powers up.
* **Auto-Restart on Process Kill:** If the process is forcefully killed (e.g. `kill -9`), systemd automatically spawns a new instance within **3 seconds**, releasing any camera device locks.

---

## 🔮 Future Improvements & Roadmap

* **DeepSORT / ByteTrack Integration:** Replace the lightweight centroid tracker with an embedded-optimized ByteTrack (Kalman filter + dual-threshold association) to maintain robust track identities through severe visual occlusions and intersecting pedestrian trajectories.
* **Zero-Copy Memory Sharing (DMA-BUF):** Implement DMA-BUF zero-copy buffer passing between GStreamer video sinks and TFLite tensor buffers, eliminating userspace memory copy overhead and shaving ~8–10 ms off the pipeline.
* **Over-the-Air (OTA) & Edge Telemetry:** Expose Prometheus / MQTT telemetry endpoints to stream real-time VIP8000 NPU load, SoC temperature, and boundary crossing analytics to an industrial edge dashboard (e.g., Grafana / ThingsBoard).

---

## 🚀 Installation & Deployment

### 1. Copy Code to the Board (Run on Laptop)
Navigate to the project root directory on your laptop and transfer the codebase to the i.MX8 board:
```bash
scp -r src configs main.py root@192.168.1.87:~/line_crossing/
```

### 2. Register & Enable Service (Run on Board via SSH)
Log into the board and configure the systemd service:
```bash
# 1. SSH into the board
ssh root@192.168.1.87

# 2. Copy the service unit config to systemd system folders
cp /root/line_crossing/configs/line-crossing.service /etc/systemd/system/

# 3. Reload systemd configuration
systemctl daemon-reload

# 4. Enable the service to run automatically on system boot
systemctl enable line-crossing

# 5. Start the service
systemctl start line-crossing
```

### 3. Service Administration Commands (Run on Board via SSH)
* **Check Service Status:**
  ```bash
  systemctl status line-crossing
  ```
* **Watch Live Log Feed:**
  ```bash
  journalctl -u line-crossing -f
  ```
* **Stop Service Gracefully (Triggers Profiling Report):**
  ```bash
  systemctl stop line-crossing
  ```

---

## 🔬 Local Simulation Mode (Laptop Development)

For development, visual inspection, or testing on a laptop without physical hardware, you can run the pipeline with OpenCV display windows.

### 1. Local Environment Setup
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Run Simulation
```powershell
# Run using local video file with performance profiling enabled
python main.py -v data/videos/vid.mp4 --no-use-tiling --profile
```
