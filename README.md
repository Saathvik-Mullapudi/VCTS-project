# 🚀 Live Line Crossing Detector on i.MX8 NPU

[![Hardware](https://img.shields.io/badge/Hardware-i.MX8M_Plus-blue.svg?style=for-the-badge&logo=cpu-intel)](https://www.nxp.com/)
[![NPU Acceleration](https://img.shields.io/badge/Inference-Vivante_VIP8000_NPU-orange.svg?style=for-the-badge)](https://www.nxp.com/)
[![GStreamer](https://img.shields.io/badge/Media_Pipeline-GStreamer_1.24-green.svg?style=for-the-badge&logo=gstreamer)](https://gstreamer.freedesktop.org/)
[![License](https://img.shields.io/badge/Status-Production_Ready-success.svg?style=for-the-badge)](https://github.com/)

A production-grade, high-performance, and self-healing ML vision system designed to track people and count boundary crossings in real-time. Built specifically for **i.MX8 boards** utilizing hardware NPU acceleration, GStreamer media pipelines, and system-level boot autostart and crash recovery.

---

## 📸 Project Showcase & Demos

*(Add your recorded video and screenshots here to make your portfolio pop!)*

| Live RTSP Stream (Counting Overlay) | On-Device Hardware Metrics |
| :---: | :---: |
| ![Live RTSP Stream Overlay](board_backup/rtsp_preview.png) <br> *Film of VLC player showing real-time foot-point tracking & line crossing counts.* | ![Terminal Hardware Metrics](board_backup/terminal_metrics.png) <br> *Terminal console showing NPU utilization active at 24.4%.* |

---

## 🧠 System Architecture

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

---

## 📊 Live On-Device Performance Metrics

The system was profiled continuously on the live i.MX8 board with a full-integer quantized YOLOv8n TFLite graph.

### 1. Stage Latency Profile

| Stage | Processed Frames | Avg Latency | Min Latency | Max Latency | Median | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Demux** | - | *N/A (HW)* | - | - | - | - |
| **1. Decoder** | - | *N/A (HW)* | - | - | - | - |
| **2. Appsink** | 713 | 0.11 ms | 0.07 ms | 1.45 ms | 0.09 ms | 0.24 ms |
| **3. Color Convert (imx g2d)** | 713 | **8.00 ms** | 6.29 ms | 29.34 ms | 7.26 ms | 11.98 ms |
| **4. Preprocess** | 713 | 9.24 ms | 7.20 ms | 19.23 ms | 8.84 ms | 11.92 ms |
| **5. Inference (YOLOv8n NPU)** | 713 | **76.26 ms** | 72.30 ms | 85.54 ms | 75.42 ms | 80.39 ms |
| **6. Postprocess** | 713 | 26.43 ms | 22.71 ms | 37.00 ms | 26.20 ms | 30.52 ms |
| **7. Tracking (Centroid)** | 713 | 0.26 ms | 0.02 ms | 2.61 ms | 0.34 ms | 0.68 ms |
| **8. Line Crossing** | 713 | 0.04 ms | 0.01 ms | 0.82 ms | 0.04 ms | 0.08 ms |
| **9. Overlay (Cairo Draw)** | 1605 | 1.39 ms | 0.56 ms | 23.73 ms | 1.16 ms | 2.60 ms |

### 2. System Resource Utilization

Metrics collected over a continuous **130-second** live camera test:

| Resource | Average Load | Maximum Peak | Monitoring Path / Source |
| :--- | :---: | :---: | :--- |
| **NPU Utilization** | **24.4%** | **42.0%** | `/sys/kernel/debug/gc/load` (Core 1 - VIP8000) |
| **CPU Utilization** | **44.3%** | **46.6%** | `psutil` (system total) |
| **RAM Utilization** | **18.1%** | **18.2%** | `psutil` (system total) |
| **SoC Temperature** | **72.4°C** | **79.0°C** | `/sys/class/thermal/thermal_zone0` |
| **GPU Utilization** | **0.0%** | **0.0%** | `/sys/kernel/debug/gc/load` (Core 0 - GPU) |

---

## 🔌 Self-Healing Architecture & Resilience

### 1. Camera Disconnect Stepped Backoff Reconnection
If the physical camera is unplugged or experiences signal dropouts:
* **GStreamer Bus Intercept:** The GStreamer pipeline catches the bus error and shuts down the state machine safely.
* **Stepped Retry Policy:**
  - **Fast Retries (First 2 minutes):** Attempts reconnection every **3 seconds** (up to 40 attempts).
  - **Slow Retries (After 2 minutes):** Backs off to a **3-minute** interval indefinitely to prevent high CPU utilization during long-term outages.
* **Automatic Recovery:** Once the physical sensor node `/dev/video3` reappears in the filesystem, the pipeline automatically re-initializes, starts up, and continues RTSP casting.

### 2. Systemd Power-On Autostart & Process Recovery
Registered as a Linux background service (`line-crossing.service`):
* **Autostart on Boot:** The application starts automatically as soon as the board receives power, allowing headless deployments in remote locations.
* **Auto-Restart on Process Kill:** If the process is forcefully killed (e.g. `killall -9 python3`), systemd automatically spawns a new instance within **3 seconds**, releasing any dangling socket or camera resource locks.

---

## 🛠️ Installation & Deployment

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
