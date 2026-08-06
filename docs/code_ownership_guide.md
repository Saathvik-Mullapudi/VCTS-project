# 🎓 Code Ownership & Interview Preparation Guide

This guide is designed to help you prepare for your technical conversation with the founder. It details the exact data flow, code map, function calls, and architectural design decisions in your project.

---

## 1. What does "Owning the Code" mean?
Founders of startups and engineering-heavy companies do not care about memorized algorithms (DSA). They care about **execution**. 
"Owning the code" means you can:
1. **Explain the "Why":** Why did you choose GStreamer over OpenCV standard capture? Why did you choose Centroid Tracking over DeepSORT?
2. **Trace the Data Flow:** If a single frame comes from the camera sensor, you can trace exactly how it transforms, which memory buffers it touches, and where it goes.
3. **Debug Instantly:** If they ask *"How would you add class filtering for cars?"* or *"What happens if the camera drops connection?"*, you know exactly which file and function to modify.

---

## 2. End-to-End Data Flow Pipeline

Here is the exact journey of a single video frame through your system:

```text
+------------------+     [GStreamer C-Level Pipeline]
|  Camera Sensor   | ➔ v4l2src reads raw YUY2/NV12 frame
+------------------+
        |
        v
+------------------+     ➔ imxvideoconvert_g2d performs hardware color conversion
|   i.MX8 GPU/VPU  |       and scales the frame to 1920x1080 (RGB)
+------------------+
        |
        v
+------------------+     ➔ appsink drops old frames if Python is busy (drop=true)
| GStreamer Queue  |       and buffers the latest frame
+------------------+
        |
        v                [Python Orchestrator Boundary]
+------------------+     ➔ main.py pulls the raw buffer using Gst.AppSink
|  appsink pull    |
+------------------+
        |
        +-----------------------------------+----------------------------------+
        | (Visual Path - 11 FPS)            | (Inference Path - 6 FPS)         |
        v                                   v                                  v
+------------------+                  +------------------+               +------------------+
| cairooverlay     |                  | Preprocessing    |               | NPU Inference    |
| (Draw overlays   |                  | (Resize to       |               | (YOLOv8n TFLite  |
|  and counters)   |                  |  640x360 RGB)    |               |  Integer Quant)  |
+------------------+                  +------------------+               +------------------+
        |                                                                          |
        v                                                                          v
+------------------+                                                     +------------------+
| RTSP stream      |                                                     | Postprocessing   |
| (udpsink / VLC)  |                                                     | (NMS threshold   |
+------------------+                                                     |  box filtering)  |
        ▲                                                                          |
        |                                                                          v
        |                                                                +------------------+
        |                                                                | Centroid Tracker |
        +----------------------------------------------------------------| (Match objects & |
                                                                         |  do Vector Math) |
                                                                         +------------------+
```

---

## 3. Code Map: Files, Classes, and Functions

### 📂 File 1: `main.py` (The Orchestrator)
*   **Role:** The entry point. Handles arguments, initializes all modules, manages the main loop, and shuts down gracefully.
*   **Key Class:** `LineCrossingDetector`
*   **Key Functions:**
    *   `__init__()`: Reads configuration JSONs (e.g. `configs/imx462_camera.json`) and overrides defaults in `configs/settings.py`. Initializes `GstPipeline`, `TFLiteInferenceEngine`, `CentroidTracker`, `SystemMonitor`, and `MetricsRecorder`.
    *   `run()`: Starts the system monitor and runs the GLib/GStreamer main event loop (`GLib.MainLoop().run()`). It also intercepts camera error flags (`self.pipeline.error_occurred`) to trigger the auto-reconnect stepped backoff.
    *   `_on_frame_callback()`: The critical connector. It is triggered by GStreamer's `cairooverlay` element whenever a new frame is ready to draw. It pulls the raw image buffer, pre-processes it, calls inference, tracks the detections, processes the cross-product math, and draws the green/red line and count boxes onto the screen.

### 📂 File 2: `src/gst_pipeline.py` (Media Ingestion)
*   **Role:** Configures and builds the optimized GStreamer pipeline string.
*   **Key Class:** `GstPipeline`
*   **Key Functions:**
    *   `_build_pipeline_str()`: Declares the hardware-accelerated pipeline. Uses `imxvideoconvert_g2d` to convert YUV camera formats to RGB16/RGB on the NXP GPU, saving CPU. Uses `appsink drop=true max-buffers=1` to ensure we only process the latest frame and prevent latency queues.
    *   `_bus_call_callback()`: Listens to the GStreamer element bus. Specifically catches `Gst.MessageType.ERROR` (e.g. camera unplugged) and sets `self.error_occurred = True` so `main.py` can restart the system.

### 📂 File 3: `src/inference.py` (NPU Execution)
*   **Role:** Pre-processes frames and executes YOLOv8n on the VeriSilicon NPU.
*   **Key Class:** `TFLiteInferenceEngine`
*   **Key Functions:**
    *   `preprocess()`: Resizes the input image to $640 \times 360$ (ML input dimensions) and prepares it as a quantized uint8 array.
    *   `infer()`: Passes the array to the TFLite Interpreter. It invokes the NPU delegate.
    *   `postprocess()`: Decodes the output boxes, scores, and class IDs. Filters out everything except Class ID 0 (Person), and runs Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes.

### 📂 File 4: `src/tracker.py` (Object Tracking & Vector Math)
*   **Role:** Maintains identities of people and calculates boundary crossings.
*   **Key Class:** `CentroidTracker`
*   **Key Functions:**
    *   `update()`: Receives bounding boxes from the current frame. Calculates the center of each box and associates them with existing IDs using Euclidean distance minimization.
    *   `check_crossing()`: Calculates the bottom-center foot point of each box:
        $$F_t = (x_t + w/2, y_t + h)$$
        Performs the vector cross-product against the line segment $P_{start}P_{end}$:
        $$d = (x_{foot} - x_{start})(y_{end} - y_{start}) - (y_{foot} - y_{start})(x_{end} - x_{start})$$
        If the sign of $d$ changes from positive to negative (or vice-versa) compared to the previous frame, a crossing is counted.

### 📂 File 5: `src/system_monitor.py` (Hardware Diagnostics)
*   **Role:** Runs a background thread that collects hardware resource usage.
*   **Key Class:** `SystemMonitor`
*   **Key Functions:**
    *   `_sample_gc_loads()`: Reads the Vivante NPU kernel debug node at `/sys/kernel/debug/gc/load` to parse Core 1 (NPU utilization) and Core 0 (GPU utilization).
    *   `format_system_summary()`: Compiles all recorded samples to print average and peak resource metrics (CPU, NPU, RAM, temperature) at process shutdown.

---

## 4. Key Design Decisions (The "Why")

Be ready to explain these 4 major design decisions. This shows senior-level engineering maturity:

| Decision | Why it matters |
| :--- | :--- |
| **GStreamer over OpenCV** | OpenCV's `cv2.VideoCapture` is blocking and handles thread scaling poorly. GStreamer offloads colorspace conversion and image scaling to the i.MX8's internal 2D GPU (`imxvideoconvert_g2d`), keeping the CPU clean. |
| **Appsink Buffer Dropping** | By setting `drop=true` and `max-buffers=1`, GStreamer drops older frames if the NPU is busy. This guarantees **zero latency build-up**; the stream is always real-time, even if NPU FPS is lower than camera FPS. |
| **Centroid Matching over DeepSORT** | DeepSORT requires running a secondary feature-extraction neural network for appearances. On edge hardware like the i.MX8, running a second model would choke the NPU. Centroid tracking runs in **under 0.2 ms** using simple geometry. |
| **Foot-Point Segment Crossing** | Tracking the box center causes false counts if a box stretches (e.g. someone raises their hands). Tracking the **foot-point** (bottom center) corresponds directly to their contact with the floor, which is stable. |

---

## 5. Potential Interview Scenarios

*   **Founder:** *"How do you handle clock drift or time jumps on the board?"*
    *   **You:** *"Embedded boards lack battery-backed RTCs and sync time via NTP after booting. This causes system clock jumps of years. To prevent this from breaking our throughput metrics, I implemented monotonic clocks (`time.monotonic()`) for all frame-by-frame FPS metrics, ensuring accurate throughput profiling regardless of NTP updates."*
*   **Founder:** *"What happens if the camera ribbon cable is physically disconnected?"*
    *   **You:** *"The GStreamer bus catches the physical interface dropout and fires an error. Our loop intercept catches this, releases the camera resource, and enters a stepped-backoff retry loop (every 3s for fast reconnect, backing off to 3m). This prevents CPU lockups and allows the system to auto-resume as soon as the sensor is reconnected."*
*   **Founder:** *"How did you verify NPU hardware acceleration?"*
    *   **You:** *"I parsed NXP's Vivante NPU kernel debugging interface at `/sys/kernel/debug/gc/load`. During active tracking, CPU utilization remained low (~47%) while NPU utilization peaked at 42.0%, proving that YOLOv8n matrix calculations were fully offloaded to the VIP8000 NPU."*
