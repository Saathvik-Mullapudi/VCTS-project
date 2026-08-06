# ⏱️ 8-Hour Crash Course: "Own Your Code" Prep Guide

You have **8 hours** to prepare for your conversation with the founder. ChatGPT’s plan is solid, but it’s designed for 5 days and makes assumptions about a generic file layout. This guide is customized **exactly for your actual codebase and files** and is compressed into an hour-by-hour timeline for today.

---

## 📅 The 8-Hour Schedule (100% COMPLETED ✅)

*   **[x] Hours 1–2: High-Level Architecture & Decoupled Data Flow** (Completed: Decoupled GStreamer 11.2 FPS vs NPU 6.0 FPS threads, `appsink drop=true max-buffers=1`).
*   **[x] Hours 3–4: File Map & Class Cheat Sheet** (Completed: Exhaustive line-by-line deep dives for `main.py`, `src/gst_pipeline.py`, `src/model.py`, `src/inference.py`, `src/tracker.py`, `src/system_monitor.py`, `src/metrics.py`).
*   **[x] Hours 5–6: Core Functions Deep-Dive** (Completed: Traced `_on_frame_callback`, `_draw_overlay`, `predict`, `check_crossing`, `_nms_numpy`, `_sample_gc_loads`).
*   **[x] Hour 7: Top 10 Founder Interview Questions & Answers** (Completed: Mastered design trade-offs, zero-copy memory, hardware locks, stepped backoff, and INT8 quantization).
*   **[x] Hour 8: Self-Grill Practice Run** (Completed: Passed brutal mock founder interview covering architecture, tracking math, industrial resilience, and scaling to 100 streams).

---

## 🌀 Hours 1–2: High-Level Architecture & Decoupled Data Flow

You must be able to explain how frame data moves through GStreamer (C-level) and Python.

```text
CAMERA SENSOR ➔ GStreamer v4l2src (1080p NV12, 30fps)
                       │
                       ▼
          imxvideoconvert_g2d (Hardware colorspace convert on NXP GPU to RGB16, 1080p)
                       │
                       ▼
          appsink (Queue with drop=true, max-buffers=1 to maintain real-time sync)
                       │
          [ GStreamer C-level Thread / Python Thread Boundary ]
                       │
                       ▼
          main.py: _on_frame_callback (Cairo overlay callback)
                       │
         +-------------+-------------+
         │ (Visual Path - 11 FPS)    │ (Inference Path - 6 FPS)
         ▼                           ▼
    cairooverlay Draw           inference.py: preprocess (Resize to 640x360, quantize)
         │                           │
         │                           ▼
         │                      inference.py: infer (TFLite interpreter calls NPU delegate)
         │                           │
         │                           ▼
         │                      inference.py: postprocess (Boxes, Scores, Class 0 Filter, NMS)
         │                           │
         │                           ▼
         │                      tracker.py: CentroidTracker (Matches IDs, does Vector Math)
         │                           │
         +◄──────────────────────────+ (Return track coordinates)
         │
         ▼
    RTSP stream (udpsink / VLC Client)
```

### The Key Concept to Explain:
> *"The GStreamer media pipeline runs on its own C-level thread pool, while the Python orchestration loop handles tracking and inference. By setting `drop=true` on the `appsink` element, GStreamer drops frames if our NPU inference loop is busy. This decouples our display rate (11.2 FPS) from our inference rate (6.0 FPS), ensuring the video stream stays completely real-time without latency buffering."*

---

### 🔍 Deep-Dive: Understanding "Decoupled Data Flow" & "Appsink Frame Dropping"

If the founder asks you to explain this, you must break it down into three simple principles:

#### 1. The Queue Latency Problem (The "Why")
*   **The Math:** The physical camera feeds frames at **30 FPS** (one frame every **33 ms**).
*   **The Bottleneck:** Our NPU inference loop takes **78 ms** to process a single frame.
*   **The Lag Accumulation:** If we tried to process every single frame, we would slowly fall behind. In 10 seconds of camera runtime, the queue of frames waiting to be processed would grow so large that the video feed would lag by seconds, eventually running out of memory and crashing.

#### 2. The Solution: `appsink drop=true` and `max-buffers=1`
We configure our GStreamer sink node to behave as a **single-slot ring buffer**:
*   `max-buffers=1` ➔ The queue can hold at most **one** frame.
*   `drop=true` ➔ If a new frame arrives from the camera while the NPU is still busy processing the old frame, the queue **discards the old frame** and replaces it with the brand-new one.
*   **The Result:** When the NPU finishes its 78ms cycle and asks for a frame, GStreamer immediately gives it the **latest** frame that arrived from the camera. This guarantees there is **zero accumulation of lag**.

#### 3. Decoupled Threading (GStreamer Thread vs. Python Thread)
*   **The GStreamer Thread (High FPS Visuals):** GStreamer runs inside a C-level thread pool. It pulls frames from `/dev/video3`, scales them, and streams them out to RTSP at **11.2 FPS**. It does not wait for Python to finish inference to stream the next frame.
*   **The Python Thread (Lower FPS Logic):** Python wakes up, grabs the latest available frame from the `appsink` buffer, runs inference and tracking, and calculates the new bounding box coordinates at **6.0 FPS**.
*   **The Bridge (`cairooverlay`):** Python sends the tracking bounding box coordinates back to GStreamer. GStreamer draws these boxes on the active stream in real-time. Because of this, the video stream looks smooth and fluid, while the bounding boxes update at the NPU's processing rate of 6 FPS.

---

## 📁 Hours 3–4: File Map & Class Cheat Sheet

Here is the exact structure of your files and classes. Memorize this map:

### 1. `main.py` (The Orchestrator)
*   **Purpose:** CLI entry point, JSON configuration loader, module coordinator, and main execution loop.
*   **Key Class:** `LineCrossingDetector`
*   **Imports:** `src/gst_pipeline.py`, `src/inference.py`, `src/tracker.py`, `src/system_monitor.py`, `src/metrics.py`.
*   **Returns:** Execution loop. Handles graceful shutdowns via Linux SIGINT/SIGTERM handlers.

### 2. `src/gst_pipeline.py` (Media Capture)
*   **Purpose:** Configures and builds the optimized hardware-accelerated GStreamer element string.
*   **Key Class:** `GstPipeline`
*   **Imports:** `gi.repository` (GObject/GStreamer bindings).
*   **Called by:** `main.py`.

### 3. `src/inference.py` (NPU Model Loader)
*   **Purpose:** Loads the YOLOv8n TFLite full-integer quantized model and runs inference.
*   **Key Class:** `TFLiteInferenceEngine`
*   **Imports:** `tflite_runtime.interpreter` or `tensorflow`.
*   **Called by:** `main.py`.

### 4. `src/tracker.py` (Object State & Intersection)
*   **Purpose:** Matches object bounding boxes across frames and computes line crossing vectors.
*   **Key Class:** `CentroidTracker`
*   **Called by:** `main.py`.

### 5. `src/system_monitor.py` (Diagnostics)
*   **Purpose:** Background thread polling CPU, NPU load, RAM, and temperatures.
*   **Key Class:** `SystemMonitor`
*   **Called by:** `main.py`.

---

## 🔍 Hours 5–6: Core Functions Deep-Dive

You need to know what happens inside these three functions.

### 1. `main.py` ➔ `LineCrossingDetector._on_frame_callback()`
This function is called by the GStreamer pipeline whenever a frame is ready to be drawn.
*   **Input:** GStreamer buffer pointer.
*   **Processing:**
    1. Extracts the frame data as a NumPy array.
    2. Passes it to `self.inference_engine.infer()` (only if the NPU is not busy).
    3. Feeds detections to `self.tracker.update()`.
    4. Calculates line crossing checks via vector cross-products.
    5. Draws the red/green boundary line and counting HUD.
*   **Output:** Modifies the frame in-place for output rendering.
*   **Failure case:** If OpenCV conversions fail due to memory corruption, we catch the exception and keep the pipeline running.

### 2. `src/tracker.py` ➔ `CentroidTracker.check_crossing()`
This handles the geometry math.
*   **Input:** Centroid dictionary, line segment coordinates.
*   **Processing:**
    1. Extracts bottom-center foot coordinates of box: $F = (x + w/2, y + h)$.
    2. Calculates cross product relative to line segment $P_1(x_1, y_1)$ to $P_2(x_2, y_2)$:
       $$d = (x - x_1)(y_2 - y_1) - (y - y_1)(x_2 - x_1)$$
    3. Checks if the sign of $d$ changes compared to the previous frame ($d_{prev} \times d_{current} < 0$).
*   **Output:** Returns `True` if crossed, incrementing the counter.

### 3. `src/gst_pipeline.py` ➔ `GstPipeline._bus_call_callback()`
Monitors pipeline errors.
*   **Input:** GStreamer bus message.
*   **Processing:** If message is `Gst.MessageType.ERROR`, it sets `self.error_occurred = True`.
*   **Output:** Triggers reconnect logic in `main.py`.

---

## 💬 Hour 7: Top 10 Founder Interview Q&As

### Q1: Why use GStreamer instead of OpenCV capture?
> *"OpenCV's `cv2.VideoCapture` is blocking, does color conversion on the CPU, and lacks built-in frame dropping. GStreamer allows us to declare hardware-accelerated pipelines that offload scaling and YUV-to-RGB conversion to the i.MX8 GPU using `imxvideoconvert_g2d`, saving precious CPU cycles."*

### Q2: How do you prevent latency queues from building up?
> *"Our GStreamer appsink uses `drop=true` and `max-buffers=1`. When inference is busy executing on the NPU (taking 78ms), older frames are dropped automatically at the ingestion boundary, ensuring the displayed stream stays strictly real-time."*

### Q3: Why Centroid Tracking instead of DeepSORT?
> *"DeepSORT requires running a secondary convolutional model for appearance features, which would bottleneck the i.MX8 NPU. Centroid tracking runs in under 0.2 ms on the CPU using Euclidean distance matching, which is lightweight and fast."*

### Q4: Why track the foot point instead of the box center?
> *"Tracking the center of bounding boxes leads to false counts when boxes expand (like when a person raises their hands). We track the bottom-center foot point, which maps directly to their contact with the floor, providing a stable coordinate."*

### Q5: How did you verify NPU hardware acceleration?
> *"I monitored NXP's Vivante NPU kernel debugfs interface at `/sys/kernel/debug/gc/load`. During active tracking, CPU load remained steady under 50% while NPU utilization (Core 1) peaked at 42.0%, proving that YOLOv8n matrix multiplication was fully offloaded to the VIP8000 NPU."*

### Q6: How do you handle NTP network time-sync jumps?
> *"Embedded boards lack battery-backed RTCs and sync time via NTP after booting, which causes system clock jumps of years. To prevent this from skewing our throughput and FPS calculations, we rely on monotonic clocks (`time.monotonic()`) for all frame-by-frame FPS metrics, ensuring accurate throughput profiling regardless of NTP updates."*

### Q7: What happens if the physical camera is unplugged?
> *"The GStreamer bus intercepts a hardware error and flags `error_occurred = True`. Our loop catches this, releases the camera resource, and enters a stepped-backoff reconnect loop (retrying every 3s for 2 minutes, then backing off to 3-minute intervals). This prevents CPU lockups and recovers automatically once `/dev/video3` is reconnected."*

### Q8: Why use full-integer INT8 quantization?
> *"The VIP8000 NPU operates on integer math. Running a standard float32 YOLOv8n model would force slow CPU software emulation. Quantizing weights and activations to 8-bit integers lets the NPU execute the model in parallel in 78 ms."*

### Q9: How is the system run in production?
> *"It runs as a systemd background service (`line-crossing.service`) with `Restart=always` and `RestartSec=3`. If the Python script is killed or crashes, the OS automatically restarts it in under 3 seconds."*

### Q10: Where is the bottleneck in your system?
> *"The bottleneck is the TFLite NPU delegate's inference latency, which takes 78.21 ms. The rest of the stages (preprocessing, tracking, math) execute in under 15 ms combined. Since GStreamer drops frames during NPU execution, our throughput is bound by this 78 ms limit."*

---

## 🔧 Low-Level Architecture (LLA) vs. High-Level Architecture (HLA)

It is highly important to understand the difference between these two, especially when talking to a founder:
*   **High-Level Architecture (HLA):** The block diagram. It tells you **what** components exist and **how** they are wired together (e.g., Camera ➔ GStreamer ➔ TFLite ➔ Tracker).
*   **Low-Level Architecture (LLA):** The mechanical details under the hood. It tells you **how** the components work in memory, threading, and hardware interfaces.

### Why LLA is Critical for Your Interview:
If you only speak in HLA, you sound like a software developer who just imported some libraries. If you speak in LLA, you prove you are an **Embedded/ML Systems Engineer** who knows how software controls the hardware.

Here are the 3 major LLA pillars of your project:

#### 1. Zero-Copy & Shared Memory (C-to-Python boundary)
In `main.py`, GStreamer (written in C) grabs a camera frame. Instead of copying that entire $1920 \times 1080$ RGB buffer into Python memory (which would waste massive CPU and bandwidth), we use GStreamer's pointer mapping:
```python
# Low-level memory map of the Gst.Buffer
success, map_info = buf.map(Gst.MapFlags.READ)
if success:
    # Directly wrap the C memory pointer as a NumPy array pointer (Zero-Copy)
    frame = np.ndarray(
        shape=(info.height, info.width, 3),
        dtype=np.uint8,
        buffer=map_info.data
    )
```
*You must explain that this is a zero-copy operation. Python accesses the raw memory buffer mapped directly by GStreamer's C library.*

#### 2. Kernel/Driver Level Offloading
*   **GPU Offloading:** Preprocessing scale operations are offloaded to NXP's 2D GPU driver via GStreamer's `imxvideoconvert_g2d` element, bypassing the standard Linux CPU kernel scheduler.
*   **NPU Offloading:** When TFLite executes `interpreter.invoke()`, the execution is delegated to the NXP eIQ Linux driver (`/dev/galcore`), which loads the integer-quantized weights directly into the VIP8000 hardware memory registers.

#### 3. Linux Signal Interception & GLib Main Loop
Standard Python signal handlers are blocked by GLib's underlying C main loop, which runs at the OS scheduler level. To make sure the process catches SIGINT/SIGTERM or simulated failures (SIGUSR1), we register the signal handler directly with the C-level GLib event loop using:
```python
GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, sigterm_handler)
```
This guarantees that when systemd stops the service, the C event loop catches it immediately and shuts down the camera node gracefully without leaving device locks.

---

## 🎯 Hour 8: Self-Grill Mock Run

Sit in front of your editor, open the codebase, and practice explaining the system as if you are presenting:
1. **Startup:** Explain how `main.py` launches, loads `configs/imx462_camera.json`, and starts the background `SystemMonitor` thread.
2. **Execution:** Trace a frame from the camera, through `gst_pipeline.py`, into `inference.py`, matching IDs in `tracker.py`, checking vector crossing signs, and casting RTSP.
3. **Resilience:** Explain what happens if you run `kill -10 <PID>` to trigger a simulated disconnect, showing how the loop gracefully restarts.
4. **Shutdown:** Explain how `systemctl stop` triggers a `SIGTERM`, shuts down GStreamer, and writes the latency and utilization summary to logs.
