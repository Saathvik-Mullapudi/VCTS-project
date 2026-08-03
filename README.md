# Live Line Crossing Detector on i.MX8 NPU

A production-grade, high-performance, and self-healing ML vision system designed to track people and count boundary crossings in real-time. Built specifically for **i.MX8 boards** utilizing hardware NPU acceleration, GStreamer media pipelines, and system-level boot autostart and crash recovery.

---

## Key Features

* 🚀 **NPU Accelerated Inference:** Runs YOLOv8n inference on the onboard VeriSilicon VIP8000 NPU using TensorFlow Lite delegates, achieving high framerates with low CPU utilization.
* 📹 **GStreamer Media Pipelines:** Leverages hardware-accelerated GStreamer elements (`imxvideoconvert_g2d`, `vpuenc_h264`) for zero-copy frame capturing, color conversion, and H.264 video encoding.
* 🔌 **Hardware Disconnect Resilience:** Stepped backoff reconnection loop. Intercepts physical camera unplugs, cleanly tears down the pipeline, and retries every 3 seconds for 2 minutes, then backs off to every 3 minutes indefinitely to prevent CPU lockups.
* 🔄 **Systemd Autostart & Recovery:** Configured as a native Linux background service (`line-crossing.service`) that automatically launches at system boot and auto-restarts within 3 seconds if the process is killed or crashes.
* 📊 **Resource & Throughput Profiling:** Logs stage-by-stage latencies (inference vs. preprocess vs. draw) and captures CPU/NPU/RAM/SoC temperature averages.

---

## System Architecture

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

## On-Device Performance & Metrics

The system was profiled on the live i.MX8 board with the following latency and system resource averages.

### Stage Latency Profile

| Stage | Processed Frames | Avg Latency (ms) | p95 Latency (ms) |
| :--- | :---: | :---: | :---: |
| **Inference (YOLOv8n NPU)** | 713 | **74.3 ms** | **76.5 ms** |
| **Cairo Overlay Drawing** | 1605 | 1.6 ms | 3.3 ms |
| **Postprocessing** | 713 | 26.2 ms | 30.7 ms |
| **Color Convert (imx g2d)** | 713 | 7.5 ms | 11.1 ms |
| **Centroid Tracking** | 713 | 0.3 ms | 0.7 ms |

### System Resource Utilization (1–2 Hour Continuous Run)
*Note: A 2.8-minute validation run yielded the following averages; to be updated with long-term metrics.*

| Resource | Average Load | Maximum Peak | Monitoring Source |
| :--- | :---: | :---: | :---: |
| **NPU Utilization** | **22.1%** | **40.0%** | `/sys/kernel/debug/gc/load` (Core 1) |
| **CPU Utilization** | **44.6%** | **50.0%** | `psutil` (system total) |
| **RAM Utilization** | **17.9%** | **18.0%** | `psutil` (system total) |
| **SoC Temperature** | **73.2°C** | **80.0°C** | `/sys/class/thermal/thermal_zone0` |
| **GPU Utilization** | **0.0%** | **0.0%** | `/sys/kernel/debug/gc/load` (Core 0) |

---

## Embedded Board Installation

Follow these steps to deploy, enable autostart, and monitor the application on your i.MX8 device:

### 1. Copy the code to the board (Run on Laptop)
Navigate to the project root directory on your laptop and copy the files via SCP:
```bash
scp -r src configs main.py root@192.168.1.87:~/line_crossing/
```

### 2. Register the Systemd Service (Run on Board via SSH)
Log into the board and copy the service unit configuration to systemd:
```bash
# 1. SSH into the board
ssh root@192.168.1.87

# 2. Copy the service config to systemd system directories
cp /root/line_crossing/configs/line-crossing.service /etc/systemd/system/

# 3. Reload systemd daemon to pick up the new service
systemctl daemon-reload

# 4. Enable the service to launch automatically on system boot
systemctl enable line-crossing

# 5. Start the service
systemctl start line-crossing
```

### 3. Monitoring and Managing the Service (Run on Board via SSH)
Use these systemctl commands to check status, review logs, and stop the service:

* **Check Service Status:**
  ```bash
  systemctl status line-crossing
  ```
* **Watch Live Metrics & Output Logs:**
  ```bash
  journalctl -u line-crossing -f
  ```
* **Gracefully Stop (Triggers Profiling Export):**
  ```bash
  systemctl stop line-crossing
  ```

---

## Laptop Development Setup (Simulation Mode)

For local testing, model adjustments, and simulation without physical hardware, you can run the pipeline on your laptop with OpenCV window popups.

### 1. Python Environment Setup
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Run in Simulation Mode (Run on Laptop)
```powershell
# Run using local video file with performance profiling enabled
python main.py -v data/videos/vid.mp4 --no-use-tiling --profile
```
