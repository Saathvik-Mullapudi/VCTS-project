# 📑 Master Repository Cheat Sheet (Quick Reference)

Keep this open during your call! This table lists every file, key class, main functions, exact line numbers, and core tasks.

---

| File Path | Key Class / Method | Line No. | Core Task / Description |
| :--- | :--- | :--- | :--- |
| **`main.py`** | `LineCrossingDetector` | [L78](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py#L78) | **Main Orchestrator:** Manages execution, configuration, and GLib loop. |
| | `_init_subsystems()` | [L90](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py#L90) | Instantiates Detector, Tracker, Counter, Monitor, and Profiler. |
| | `run()` | [L500](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py#L500) | GLib event loop, stepped backoff (3s/180s) reconnect, `finally:` cleanup. |
| | `_on_new_sample()` | [L284](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py#L284) | `appsink` callback: pulls GstSample, extracts NumPy frame, triggers detection. |
| | `_process_frame_for_detection()`| [L205](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py#L205) | Calls `predict()`, `tracker.update()`, `counter.update()`, updates `with self.lock:`. |
| | `_draw_overlay()` | [L328](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py#L328) | `cairooverlay` callback: reads snapshot under `with self.lock:`, calls Cairo draw. |
| | `GLib.unix_signal_add()` | [L532](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/main.py#L532) | Intercepts `SIGTERM` (systemd stop) and `SIGUSR1` (simulated failure) in C-space. |
| | | | |
| **`src/gst_pipeline.py`** | `extract_frame_from_sample()` | [L19](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/gst_pipeline.py#L19) | Zero-copy buffer mapping (`buf.map`), wraps C memory as NumPy array. |
| | `build_pipeline()` | [L119](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/gst_pipeline.py#L119) | Builds GStreamer string (`imxvideoconvert_g2d`, `appsink drop=true max-buffers=1`). |
| | `handle_bus_message()` | [L101](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/gst_pipeline.py#L101) | Catches `Gst.MessageType.ERROR` on camera disconnect, sets `error_occurred=True`. |
| | `start_rtsp_server()` | [L176](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/gst_pipeline.py#L176) | Bridges local UDP stream to RTSP server (`rtsp://0.0.0.0:8554/video`). |
| | `stop_pipeline()` | [L223](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/gst_pipeline.py#L223) | Transitions state to `Gst.State.NULL` to release `/dev/video3` hardware locks. |
| | | | |
| **`src/model.py`** | `TFLitePersonDetector` | [L32](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/model.py#L32) | **AI Engine Wrapper:** Loads TFLite runtime and handles bounding box scaling. |
| | `__init__()` | [L33](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/model.py#L33) | Loads `libvx_delegate.so` (VIP8000 NPU driver), reads scale & zero_point params. |
| | `predict()` | [L121](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/model.py#L121) | Orchestrates ROI tiling crop ➔ Preprocess ➔ NPU `invoke()` ➔ Postprocess ➔ Box remap. |
| | `_map_boxes_tile_to_display()`| [L109](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/model.py#L109) | Remaps bounding box coordinates from cropped tile back to 1080p display space. |
| | | | |
| **`src/inference.py`** | `CPUPreprocessor` | [L11](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/inference.py#L11) | Pre-allocates single memory buffer `(1, 360, 640, 3)` to avoid RAM fragmentation. |
| | `CPUPreprocessor.process()` | [L18](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/inference.py#L18) | Resizes frame to 640x360, letterbox padding, uint8/int8 quantization scaling. |
| | `OptimizedPostprocessor` | [L46](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/inference.py#L46) | **Output Decoder:** De-quantizes tensors, filters Person class, runs NMS. |
| | `_nms_numpy()` | [L53](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/inference.py#L53) | Non-Maximum Suppression algorithm: eliminates overlapping boxes ($\text{IoU} > 0.45$). |
| | `process()` | [L77](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/inference.py#L77) | De-quantizes $V_{float} = (q - z) \cdot s$, keeps Class 0 (score $> 0.3$), un-letterboxes. |
| | | | |
| **`src/tracker.py`** | `CentroidTracker` | [L10](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py#L10) | **Object Tracker:** Maintains track IDs using Euclidean distance minimization. |
| | `update()` | [L36](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py#L36) | Computes `cdist` matrix, applies EMA centroid smoothing ($\alpha=0.75$) to stop jitter. |
| | `get_foot_point()` | [L101](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py#L101) | Calculates bottom-center foot point $F = (x + w/2, y_2)$ of bounding box. |
| | `LineCrossingCounter` | [L113](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py#L113) | **Geometry Counter:** Evaluates line crossing vector cross-products. |
| | `get_side()` | [L123](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py#L123) | Vector cross-product: $d(P) = (x_2 - x_1)(p_y - y_1) - (y_2 - y_1)(p_x - x_1)$. |
| | `get_distance()` | [L129](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py#L129) | Perpendicular distance from foot point to line segment. |
| | `update()` | [L137](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/tracker.py#L137) | Enforces 30px spatial buffer padding + 2-frame temporal debouncing state machine. |
| | | | |
| **`src/renderer.py`** | `draw_overlay_cairo()` | [L68](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/renderer.py#L68) | Paints boundary line, blue boxes, green foot-points, and HUD text via Cairo context. |
| | | | |
| **`src/system_monitor.py`**| `SystemMonitor` | [L13](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/system_monitor.py#L13) | **Hardware Monitor Thread:** Daemon thread polling CPU, RAM, NPU, GPU, Temp. |
| | `_sample_gc_loads()` | [L176](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/system_monitor.py#L176) | Reads `/sys/kernel/debug/gc/load` to parse Core 0 (GPU) and Core 1 (NPU). |
| | `_sample_temperature_c()` | [L141](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/system_monitor.py#L141) | Reads `/sys/class/thermal/thermal_zone*/temp` to record peak SoC core temperature. |
| | `format_system_summary()`| [L371](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/system_monitor.py#L371) | Computes Mean, Max, p95 for CPU, NPU, RAM, Temp at shutdown. |
| | | | |
| **`src/metrics.py`** | `ThroughputProfiler` | [L15](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/metrics.py#L15) | **Software Profiler:** Nanosecond context manager (`with profiler.stage(...)`). |
| | `stage()` | [L53](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/metrics.py#L53) | High-precision `time.perf_counter_ns()` context manager timer. |
| | `format_profiling_summary()`| [L72](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/metrics.py#L72) | Generates Throughput (FPS) table and Latency report (Mean, Median, p95). |
| | `MetricsRecorder` | [L160](file:///c:/Users/saath/OneDrive/Desktop/vcts_1/line_crossing/src/metrics.py#L160) | Writes structured `metrics.csv` and `metrics.json` reports to disk. |
