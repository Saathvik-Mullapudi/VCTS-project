# 🛠️ Developer Handover Guide: Line Crossing Detection Subsystem

Welcome! This guide is designed to help you quickly understand, run, adapt, and integrate the **Real-Time Line Crossing Detection System** into your own application or codebase.

---

## 🎯 1. Overview & Architecture

This repository implements a **decoupled, hardware-accelerated computer vision pipeline** optimized for edge platforms (specifically NXP i.MX8M Plus) as well as desktop development (Windows/Linux).

### Key Architectural Principles:
1. **Decoupled Video & Inference Streams:** Video ingestion and rendering run in GStreamer / C-space at full camera framerate (e.g. 11.2+ FPS). AI inference runs on its own loop (e.g. 6.0 FPS). GStreamer drops intermediate frames dynamically (`appsink drop=true max-buffers=1`) so video never lags behind real-time.
2. **Orchestrator Pattern (`main.py`):** `main.py` manages lifecycle, GLib event loop, and signal handling. Core logic is modularized inside `src/`.
3. **Thread Safety:** Shared state between the NPU thread and GStreamer rendering thread is synchronized using lightweight `with self.lock:` snapshots.

```
 Physical / RTSP Camera 
        │
        ▼
 GStreamer Pipeline ───(NV12 ➔ BGRx via GPU)───┬──► cairooverlay (_draw_overlay) ──► RTSP / Screen
                                                │
                                                ▼ (Zero-Copy Buffer Map)
                                           appsink (_on_new_sample)
                                                │
                                                ▼
                                         model.py (NPU Inference)
                                                │
                                                ▼
                                       inference.py (NMS)
                                                │
                                                ▼
                                   tracker.py (Centroid & Cross Math)
```

---

## 📂 2. Repository Structure & Key Classes

| File | Primary Class / Functions | Responsibility |
| :--- | :--- | :--- |
| [`main.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py) | `LineCrossingDetector` | Orchestrates pipeline lifecycle, event loops, thread locking, and reconnection logic. |
| [`src/gst_pipeline.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/gst_pipeline.py) | `build_pipeline()`, `extract_frame_from_sample()` | Builds GStreamer element strings (`v4l2src`, `imxvideoconvert_g2d`, `appsink`) and maps zero-copy C memory buffers to NumPy. |
| [`src/model.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/model.py) | `TFLitePersonDetector` | Loads TFLite runtime + NPU delegate (`libvx_delegate.so`), performs ROI Tiling, and handles box re-scaling. |
| [`src/inference.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/inference.py) | `CPUPreprocessor`, `OptimizedPostprocessor` | Resizes/letterboxes input buffers, de-quantizes NPU output tensors $(V_{float} = (q - z) \cdot s)$, and executes NumPy NMS. |
| [`src/tracker.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py) | `CentroidTracker`, `LineCrossingCounter` | Tracks bottom-center foot points $(x + w/2, y_2)$ with EMA smoothing ($\alpha=0.75$), and checks 2D vector cross-product line crossings. |
| [`src/renderer.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/renderer.py) | `draw_overlay_cairo()` | Renders boundary lines, bounding boxes, foot points, and HUD statistics using Cairo context. |
| [`src/system_monitor.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/system_monitor.py)| `SystemMonitor` | Background daemon thread polling CPU, RAM, Temp, and NXP Vivante NPU driver (`/sys/kernel/debug/gc/load`). |
| [`src/metrics.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/metrics.py) | `ThroughputProfiler`, `MetricsRecorder` | High-precision nanosecond profiling (`with profiler.stage(...)`) and CSV/JSON metadata logging. |

---

## ⚙️ 3. How to Configure for Your Own Application

All major parameters are centrally managed in [`configs/settings.py`](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/configs/settings.py) or overridden via JSON configs (e.g. `configs/imx462_camera.json`).

### 🛠️ Common Customizations:

1. **Changing Camera / Video Source:**
   In JSON or `settings.py`:
   ```python
   "video_src": "/dev/video3"           # Physical CSI/USB camera
   "video_src": "data/videos/test.mp4"  # Local video file
   "video_src": "rtsp://192.168.1.50:554/stream" # RTSP IP Camera
   ```

2. **Adjusting Boundary Line Coordinates:**
   Coordinates are defined in **1080p display space** (`1920x1080`):
   ```python
   LINE_START = (691, 496)
   LINE_END = (1593, 648)
   ```
   *Note: `LineCrossingCounter` automatically scales these to model input dimensions when performing vector calculations.*

3. **ROI Tiling (For High-Resolution / Wide-Angle Cameras):**
   If people cross in a specific region of interest (e.g. the right side of the screen):
   ```python
   USE_TILING = True
   TILE_PADDING = 120
   ```
   This crops the 1080p frame around the line before resizing to model dimensions, doubling pixel resolution for distant people.

4. **Tracking Sensitivity & Debouncing:**
   ```python
   BUFFER_PIXELS = 30     # Spatial padding boundary around line
   DEBOUNCE_FRAMES = 2    # Consecutive frames required on new side to confirm cross
   ```

---

## 🔌 4. How to Plug This Subsystem into Your Application

If you are importing this logic into an existing backend or microservice, follow these integration steps:

### Option A: Use `LineCrossingDetector` directly (Standalone Process)
Run `main.py` directly, configuring RTSP output or HTTP callbacks:
```bash
python main.py --config configs/imx462_camera.json --profile
```

### Option B: Modular Import into your Python Backend
If you already have your own frame capture loop or frame queue, import `TFLitePersonDetector`, `CentroidTracker`, and `LineCrossingCounter` directly:

```python
from src.model import TFLitePersonDetector
from src.tracker import CentroidTracker, LineCrossingCounter

# 1. Initialize Detector & Tracker
detector = TFLitePersonDetector(
    model_path="models/yolov8n_int8.tflite",
    use_npu=True,          # Set True on i.MX8, False on x86/Laptop
    confidence_thresh=0.3
)
tracker = CentroidTracker(maxDisappeared=15, smoothing=0.75)
counter = LineCrossingCounter(
    line_start=(691, 496), 
    line_end=(1593, 648),
    display_shape=(1920, 1080),
    ml_shape=(640, 360)
)

# 2. In your frame processing loop:
def process_incoming_frame(bgr_frame):
    # Run NPU prediction
    boxes, scores, class_ids, scale_params = detector.predict(bgr_frame)
    
    # Update tracker
    tracked_objects = tracker.update(boxes)
    
    # Update crossing counter
    crossing_count = counter.update(tracked_objects, tracker.boxes)
    
    return tracked_objects, crossing_count
```

---

## 🛠️ 5. Running & Testing

### On Laptop (Windows / Linux CPU):
```bash
python main.py --video data/videos/vid.mp4
```

### On NXP i.MX8 Edge Hardware (NPU Mode):
```bash
python main.py --config configs/imx462_camera.json --profile
```

### Checking System Logs & Output Metrics:
* Logs are saved to: `outputs/logs/line_crossing.log`
* Performance metrics are logged to: `outputs/metrics/metrics.csv` and `metrics.json`

---

## ❓ FAQ & Troubleshooting for Coworkers

* **Q: Why does the video stream stay fluid even if inference takes 80ms?**
  * *A:* GStreamer runs visual streaming and NPU inference on separate threads. `appsink drop=true max-buffers=1` automatically discards intermediate frames so the buffer never backs up.
* **Q: How does hardware disconnect recovery work?**
  * *A:* GStreamer bus errors catch unplugs, set `Gst.State.NULL` to release hardware locks, and trigger a stepped-backoff reconnect loop (3s for 2 mins ➔ 3 mins thereafter).
* **Q: How do I change the detection target from Person to Cars/Vehicles?**
  * *A:* In `src/inference.py`, change `class_ids = [0]` (Person in COCO dataset) to `class_ids = [2, 3, 5, 7]` (Car, Motorcycle, Bus, Truck).

---

*Handover document maintained by [Your Name]. Feel free to open issues or ask questions!*
