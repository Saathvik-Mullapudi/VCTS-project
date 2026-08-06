# 📊 Internship Presentation Outline: Live Line Crossing Detector on i.MX8 NPU

This outline follows a clean, professional engineering structure: **Initial Prototype ➔ Model Acceleration ➔ Pipeline Refinement ➔ Accuracy Math ➔ Resilience ➔ Benchmark Results**.

---

## Slide 1: Title Slide
* **Slide Title:** Edge-AI Line Crossing Detector on NXP i.MX8
* **Subtitle:** Porting, Hardware Offloading, and Optimization of Real-Time Vision Systems
* **Presenter Name:** [Your Name]

---

## Slide 2: Initial Prototype & Performance Bottlenecks
* **Initial Laptop Prototyping:**
  * Started with a standard OpenCV Python script (`cv2.VideoCapture`) running a YOLO model on a development laptop.
* **Edge Hardware Constraints (NXP i.MX8):**
  * Porting the script to the i.MX8 board caused CPU usage to hit **85%+**, and video lag accumulated over time.
* **Initial Optimization Attempt (Manual Frame Skipping):**
  * Implemented a manual frame-skipper in Python (processing every 3rd or 5th frame).
* **Limitations of Fixed Frame Skipping:**
  * Fixed skipping was rigid—if inference latency or camera framerate fluctuated, video feed lag still accumulated, and tracking accuracy dropped.
* **Speaker Notes:**
  > *"When I first ported our laptop computer vision prototype to the i.MX8 edge board, CPU usage reached 85%, and the video stream accumulated lag. My first step was writing a manual frame-skipper in Python. However, fixed frame-skipping proved too rigid to handle dynamic camera framerates and inference timing variations."*

---

## Slide 3: Model Quantization & NPU Hardware Delegation
* **Float32 CPU Bottleneck:**
  * Running standard float32 YOLOv8n on the CPU took **> 400 ms** per frame.
* **Full-Integer Quantization:**
  * Converted the YOLOv8n model to **8-bit full-integer TFLite quantization** (`uint8`/`int8`).
* **NPU Hardware Delegate (`libvx_delegate.so`):**
  * Configured TFLite runtime to delegate matrix operations to the onboard **VeriSilicon VIP8000 NPU**.
* **Performance Impact:**
  * Inference latency dropped from > 400 ms down to **78.2 ms**, maintaining average NPU utilization at **21.5%**.
* **Speaker Notes:**
  > *"To reduce inference latency, I quantized YOLOv8n to 8-bit full-integer TFLite format. I then configured the TFLite runtime to load NXP's libvx_delegate.so driver, offloading tensor calculations directly to the onboard VIP8000 NPU. This reduced inference latency to 78 milliseconds while using 21.5% of the NPU capacity."*

---

## Slide 4: Pipeline Refinement & Dynamic Frame Dropping
* **GPU Colorspace Offloading (`imxvideoconvert_g2d`):**
  * Identified that software YUV-to-RGB conversion was choking CPU cores.
  * Replaced OpenCV capture with GStreamer C-plugins, offloading frame scaling and colorspace conversion to the i.MX8 **2D GPU**.
* **Transition from Manual Skipping to AppSink Ring Buffer:**
  * **Removed manual frame-skipping code** and configured GStreamer **`appsink (drop=true max-buffers=1)`**.
* **Decoupled System Architecture:**
  * `appsink` dynamically discards intermediate frames while the NPU is busy, ensuring Python *always* receives the latest frame without lag.
  * Decoupled the **Visual Streaming Rate (11.2 FPS)** from the **NPU Inference Rate (6.0 FPS)**.
* **Speaker Notes:**
  > *"After offloading model execution, I turned to pipeline optimization. First, I offloaded colorspace conversion and scaling to the i.MX8 GPU using GStreamer's imxvideoconvert_g2d element. Next, I replaced the manual frame-skipping logic with GStreamer's appsink element configured with drop=true and max-buffers=1. This dynamically drops stale frames at the buffer boundary, guaranteeing zero video lag while keeping video streaming smooth at 11.2 FPS."*

---

## Slide 5: Boundary Math & False-Positive Elimination
* **Target Coordinates & Jitter:**
  * Bounding box centroid jitter near the boundary line caused false crossing counts.
* **3-Layer Math Defense:**
  1. **Foot-Point Tracking:** Tracked bottom-center point $F_t = \left(\frac{x_1 + x_2}{2}, y_2\right)$ for stable contact coordinates on the ground.
  2. **Exponential Moving Average (EMA) Smoothing:** Applied EMA smoothing ($\alpha = 0.75$) to centroid coordinates to eliminate high-frequency box jitter.
  3. **Vector Cross-Product Math + Spatial & Temporal Filters:**
     $$d(P) = (x - x_1)(y_2 - y_1) - (y - y_1)(x_2 - x_1)$$
     Enforced a **30-pixel spatial padding zone** and **2-frame temporal debouncing state machine** to prevent false counts from loitering.
* **Speaker Notes:**
  > *"To ensure count accuracy, I addressed bounding box jitter near the line. I updated the math to track foot points instead of box centers, applied Exponential Moving Average smoothing to centroids, and added a 30-pixel spatial padding zone with a 2-frame debouncing state machine. This eliminated false counts caused by loitering individuals."*

---

## Slide 6: Industrial Resilience & Fault Tolerance
* **Hardware Disconnect Recovery:**
  * Intercepts GStreamer bus error messages (`Gst.MessageType.ERROR`) when camera hardware unplugs.
  * Transitions pipeline to `Gst.State.NULL` to release `/dev/video3` hardware locks.
  * **Stepped Backoff:** Retries every 3 seconds for 2 minutes (40 retries), then backs off to 3-minute intervals.
* **OS Service Integration (Systemd):**
  * Registered `GLib.unix_signal_add(signal.SIGTERM)` directly with the C event loop.
  * Configured as a background daemon (`line-crossing.service`) that auto-recovers within **3 seconds** if killed.
* **Speaker Notes:**
  > *"For field deployments, I implemented a self-healing layer. If a camera cable is unplugged, our GStreamer bus callback catches the error, releases hardware device locks, and enters a stepped-backoff reconnect loop. If the process is terminated, systemd automatically restarts the service within 3 seconds."*

---

## Slide 7: Benchmarking & System Verification (30.7-Minute Run)
* **Continuous Stability Test:** Processed **26,911 frames** over 30.7 continuous minutes.
* **Benchmark Results:**
  * **RTSP Video Output:** 11.2 FPS
  * **NPU Inference Rate:** 6.0 FPS
  * **Average NPU Load:** 21.5% (Peak: 42.0%)
  * **Average CPU Load:** 47.3% (Peak: 84.0%)
  * **RAM Memory Stability:** Flat at 18.9% (413 MB) — zero memory leaks.
  * **Thermal Load:** 80.6°C (safely under 85°C thermal throttle threshold).
* **Speaker Notes:**
  > *"I verified system performance with a 30-minute stress test on live camera feeds. The benchmark results confirmed stability: RAM usage remained flat at 18.9% with zero memory leaks, NPU load averaged 21.5%, and CPU temperature stabilized safely at 80.6°C."*

---

## Slide 8: Key Takeaways & Learnings
* **Systems Engineering Perspective:**
  * Learned to profile hardware registers first (`/sys/kernel/debug/gc/load`) rather than guessing bottlenecks.
* **Iterative Architectural Design:**
  * Recognizing when initial implementations (like manual frame skipping) should be replaced with better pipeline paradigms (`appsink` ring buffers).
* **Production Preparedness:**
  * Designing software that handles hardware unplugs, signals, and device locks cleanly.
* **Speaker Notes:**
  > *"Reflecting on this project, my biggest learning was approaching software from a systems perspective—profiling low-level hardware registers, iterating on pipeline architecture, and building resilient code designed for real-world hardware."*
