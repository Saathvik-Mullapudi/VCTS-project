"""
gstreamer_pipeline.py
---------------------
Proper GStreamer pipeline for the NXP i.MX board.

This replaces the OpenCV-based capture in main.py with a native GStreamer pipeline
that feeds frames directly into the TFLite model via the `tensor_filter` element,
using the i.MX NPU for hardware-accelerated inference.

Pipeline structure:
    filesrc / v4l2src
        → decodebin
        → videoconvert
        → videoscale
        → video/x-raw,width=320,height=320,format=RGB
        → tensor_converter          (NNStreamer: converts raw video to tensor)
        → tensor_filter             (NNStreamer: runs TFLite model on NPU)
        → appsink                   (returns results to Python)

Requirements (on the i.MX board):
    - GStreamer 1.0 + Python bindings: sudo apt install python3-gst-1.0
    - NNStreamer:                      sudo apt install nnstreamer nnstreamer-tflite
    - TFLite runtime:                  pip3 install tflite-runtime
"""

import sys
import numpy as np
import threading
import queue
import time

# -----------------------------------------------------------------------
# Try to import GStreamer Python bindings (only available on Linux/i.MX).
# On Windows (dev machine) this import will fail – we catch that below.
# -----------------------------------------------------------------------
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    from gi.repository import Gst, GstApp, GLib
    GST_AVAILABLE = True
    Gst.init(None)
except (ImportError, ValueError):
    GST_AVAILABLE = False
    print("[GStreamer] WARNING: GStreamer Python bindings not found.")
    print("[GStreamer]          Install on i.MX: sudo apt install python3-gst-1.0 nnstreamer")
    print("[GStreamer]          Falling back to OpenCV capture on this machine.")


# -----------------------------------------------------------------------
# Helper: build the GStreamer pipeline string
# -----------------------------------------------------------------------
def _build_pipeline_str(source: str, model_path: str, use_npu: bool = True) -> str:
    """
    Build a GStreamer pipeline string for the given source and TFLite model.

    Args:
        source:     File path, /dev/videoX, or rtsp:// URL.
        model_path: Path to the .tflite model file.
        use_npu:    If True, use i.MX NPU delegate via tensor_filter.
                    If False, fall back to CPU inference.

    Returns:
        GStreamer pipeline description string.
    """
    # --- Source element ---------------------------------------------------
    if source.startswith('/dev/video'):
        # Live camera via V4L2 (Linux)
        src_element = f"v4l2src device={source}"
    elif source.startswith(('rtsp://', 'http://', 'https://')):
        # RTSP / HTTP stream
        src_element = f"rtspsrc location={source} latency=200"
    else:
        # Local video file
        src_element = f"filesrc location={source}"

    # --- Decode & pre-process -------------------------------------------
    decode_elements = (
        "! decodebin "
        "! videoconvert "
        "! videoscale "
        "! video/x-raw,width=320,height=320,format=RGB "
    )

    # --- NNStreamer inference elements -----------------------------------
    # tensor_converter: converts raw video/x-raw → other/tensor
    # tensor_filter:    runs the TFLite model (with optional NPU delegate)
    if use_npu:
        # On i.MX 8M Plus, the NPU delegate is exposed via the VSI/Galcore library.
        # NNStreamer uses `accelerator` property to select the NPU.
        filter_element = (
            "! tensor_converter "
            f"! tensor_filter framework=tensorflow-lite model={model_path} "
            "   accelerator=true:npu "     # i.MX NPU
            "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
    else:
        # CPU fallback – same pipeline, no accelerator
        filter_element = (
            "! tensor_converter "
            f"! tensor_filter framework=tensorflow-lite model={model_path} "
            "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

    pipeline_str = f"{src_element} {decode_elements} {filter_element}"
    print(f"[GStreamer] Pipeline: {pipeline_str}")
    return pipeline_str


# -----------------------------------------------------------------------
# Main class: GStreamerPipeline
# -----------------------------------------------------------------------
class GStreamerPipeline:
    """
    Wraps a GStreamer pipeline that runs TFLite inference via NNStreamer's
    tensor_filter element, using the i.MX NPU when available.

    Usage:
        pipeline = GStreamerPipeline(
            source="data/videos/vid.mp4",
            model_path="models/yolov8n_saved_model/yolov8n_int8.tflite",
            use_npu=True,
        )
        pipeline.start()

        while True:
            output_tensor = pipeline.get_result(timeout=1.0)
            if output_tensor is None:
                break
            # Post-process output_tensor here …

        pipeline.stop()
    """

    def __init__(self, source: str, model_path: str, use_npu: bool = True):
        if not GST_AVAILABLE:
            raise RuntimeError(
                "GStreamer Python bindings not available on this machine.\n"
                "Install on i.MX: sudo apt install python3-gst-1.0 nnstreamer nnstreamer-tflite"
            )

        self.source = source
        self.model_path = model_path
        self.use_npu = use_npu
        self._result_queue = queue.Queue(maxsize=2)
        self._running = False
        self._pipeline = None
        self._loop = None
        self._loop_thread = None
        self._frame_count = 0
        self._start_time = None

    # ------------------------------------------------------------------
    def start(self):
        """Build and start the GStreamer pipeline."""
        pipeline_str = _build_pipeline_str(self.source, self.model_path, self.use_npu)

        self._pipeline = Gst.parse_launch(pipeline_str)
        appsink = self._pipeline.get_by_name("sink")
        if appsink is None:
            raise RuntimeError("[GStreamer] Could not find appsink element named 'sink'.")

        # Connect new-sample signal to our callback
        appsink.connect("new-sample", self._on_new_sample)

        # Connect bus to handle EOS and errors
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos",   self._on_eos)
        bus.connect("message::error", self._on_error)

        # Start GLib main loop in a background thread so it doesn't block
        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()

        # Set the pipeline to PLAYING
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("[GStreamer] Failed to set pipeline to PLAYING state.")

        self._running = True
        self._start_time = time.time()
        print(f"[GStreamer] Pipeline started. NPU={'enabled' if self.use_npu else 'disabled (CPU)'}")

    # ------------------------------------------------------------------
    def stop(self):
        """Stop the pipeline and release resources."""
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        print(f"[GStreamer] Pipeline stopped. Processed {self._frame_count} frames at {fps:.2f} FPS.")

    # ------------------------------------------------------------------
    def get_result(self, timeout: float = 1.0):
        """
        Retrieve the next inference result tensor from the queue.

        Returns:
            numpy.ndarray: The raw output tensor from the TFLite model,
                           or None if the pipeline has ended / timed out.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    def _on_new_sample(self, sink):
        """
        Callback invoked by GStreamer for each output tensor from tensor_filter.
        Converts the buffer to a numpy array and pushes it to the result queue.
        """
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)

        # Get tensor dimensions from caps
        # NNStreamer exposes: dimension, type
        try:
            dim_str = structure.get_string("dimension")      # e.g. "84:8400:1:1"
            tensor_type = structure.get_string("type")        # e.g. "float32"

            dims = list(map(int, dim_str.split(":")))
            dtype_map = {
                "float32": np.float32,
                "float16": np.float16,
                "int8":    np.int8,
                "uint8":   np.uint8,
            }
            dtype = dtype_map.get(tensor_type, np.float32)

            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            tensor = np.ndarray(dims, dtype=dtype, buffer=mapinfo.data).copy()
            buf.unmap(mapinfo)

        except Exception as e:
            print(f"[GStreamer] Error parsing tensor: {e}")
            return Gst.FlowReturn.ERROR

        self._frame_count += 1

        # Non-blocking put (drop oldest if queue is full)
        try:
            self._result_queue.put_nowait(tensor)
        except queue.Full:
            try:
                self._result_queue.get_nowait()
                self._result_queue.put_nowait(tensor)
            except queue.Empty:
                pass

        return Gst.FlowReturn.OK

    # ------------------------------------------------------------------
    def _on_eos(self, bus, msg):
        print("[GStreamer] End of stream.")
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.quit()

    # ------------------------------------------------------------------
    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"[GStreamer] Error: {err.message}")
        print(f"[GStreamer] Debug: {debug}")
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.quit()


# -----------------------------------------------------------------------
# Convenience: OpenCV fallback for Windows / dev machines
# -----------------------------------------------------------------------
def open_capture_with_fallback(source: str):
    """
    On the i.MX board: use GStreamer filesrc/v4l2src pipeline.
    On Windows (dev machine): fall back to plain OpenCV VideoCapture.
    """
    import cv2
    if GST_AVAILABLE:
        src = str(source)
        if src.startswith('/dev/video'):
            pipeline = f"v4l2src device={src} ! videoconvert ! video/x-raw,format=BGR ! appsink"
        elif src.startswith(('rtsp://', 'http://')):
            pipeline = f"rtspsrc location={src} latency=200 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink"
        else:
            pipeline = f"filesrc location={src} ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[GStreamer] Opened via GStreamer: {pipeline}")
            return cap, True
    # Fallback
    cap = cv2.VideoCapture(source)
    print("[GStreamer] Opened via OpenCV fallback (no GStreamer).")
    return cap, False


# -----------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run this script directly on the i.MX board to test the pipeline:
        python3 src/line_crossing/gstreamer_pipeline.py data/videos/vid.mp4 models/yolov8n_saved_model/yolov8n_int8.tflite
    """
    import argparse
    parser = argparse.ArgumentParser(description="GStreamer + TFLite smoke test")
    parser.add_argument("source",     help="Video file / /dev/videoX / rtsp:// URL")
    parser.add_argument("model_path", help="Path to the .tflite model")
    parser.add_argument("--cpu",      action="store_true", help="Disable NPU, use CPU")
    args = parser.parse_args()

    pipeline = GStreamerPipeline(
        source=args.source,
        model_path=args.model_path,
        use_npu=not args.cpu,
    )
    pipeline.start()

    print("Receiving inference results (Ctrl-C to stop)…")
    try:
        while True:
            result = pipeline.get_result(timeout=2.0)
            if result is None:
                print("No result received (pipeline may have ended).")
                break
            print(f"  Output tensor shape: {result.shape}, dtype: {result.dtype}, "
                  f"first 5 values: {result.ravel()[:5]}")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
