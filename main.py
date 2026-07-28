#!/usr/bin/env python3
import os
import logging

os.makedirs("outputs/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/logs/line_crossing.log")
    ]
)
logger = logging.getLogger(__name__)
import time
import threading
import cv2
import numpy as np
from datetime import datetime
import argparse
try:
    import cairo
except ImportError:
    cairo = None




from configs.settings import *

# ---------------------------------------------------------------------------
# GStreamer Python bindings (optional - only available on Linux / i.MX8).
# On Windows this import will fail; GstCapture falls back to cv2.VideoCapture.
# ---------------------------------------------------------------------------
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import Gst, GLib, GstRtspServer
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False



from src.system_monitor import SystemMonitor
from src.metrics import MetricsRecorder, ThroughputProfiler
from src.tracker import CentroidTracker, LineCrossingCounter
from src.model import TFLitePersonDetector, NPU_AVAILABLE
from src.renderer import draw_overlay_cv, draw_overlay_cairo

# ============================================================================
# MAIN LINE CROSSING DETECTOR
# ============================================================================
class LineCrossingDetector:
    
    def __init__(self, model_path, video_src, output_file=None, use_tiling=USE_TILING, tile_padding=TILE_PADDING,
                 metrics_csv_path="outputs/metrics/metrics.csv", metrics_json_path="outputs/metrics/metrics.json", preview_enabled=True,
                 debug_mapping=False, profiling_enabled=PROFILING_ENABLED):
        self.video_src = video_src
        self.output_file = output_file
        self.use_tiling = use_tiling
        self.tile_padding = tile_padding
        self.metrics_csv_path = metrics_csv_path
        self.metrics_json_path = metrics_json_path
        self.preview_enabled = preview_enabled
        self.debug_mapping = debug_mapping
        self.profiling_enabled = profiling_enabled
        # Detect if the source is a camera. On Linux it's '/dev/video*', on Windows it's an integer index like '0'
        self.is_camera = video_src.isdigit() or video_src.startswith("/dev/video")
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.actual_fps = 30.0
        self.video_frame_count = 0
        self.video_last_fps_time = time.time()
        self.video_fps = 0.0
        # Read the true encoded FPS of the video file before the pipeline starts
        if not self.is_camera:
            _probe = cv2.VideoCapture(video_src)
            self.source_fps = _probe.get(cv2.CAP_PROP_FPS) or 0.0
            _probe.release()
        else:
            self.source_fps = 0.0
        self.lock = threading.Lock()

        
        self.person_boxes = []
        self.person_boxes_display = []
        self.person_scores = []
        self.tracked_objects = {}
        self.track_foot_points = []
        self.crossing_count = 0
        
        # Load model
        self.detector = TFLitePersonDetector(model_path)
        self.in_h, self.in_w = self.detector.in_h, self.detector.in_w
        self.npu_status = self.detector.npu_status
        
        # Initialize tracker and counter
        self.tracker = CentroidTracker(
            maxDisappeared=MAX_DISAPPEARED,
            maxDistance=MAX_TRACK_DISTANCE,
        )
        self.counter = LineCrossingCounter(LINE_START, LINE_END, self.in_w, self.in_h, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self.profiler = ThroughputProfiler([
            "demux",
            "decoder",
            "appsink",
            "color_convert",
            "preprocess",
            "inference",
            "postprocess",
            "tracking",
            "line_crossing",
            "overlay",
            "encoder",
            "output_write"
        ])
        if profiling_enabled:
            self.profiler.enable()
        self.metrics_recorder = MetricsRecorder(self._build_metrics_metadata(model_path))
        self.system_monitor = SystemMonitor(
            interval_sec=SYSTEM_MONITOR_INTERVAL,
            frame_number_fn=lambda: self.frame_count,
            on_sample=self.metrics_recorder.add_system_record,
        )

    def _build_metrics_metadata(self, model_path):
        return {
            "timestamp_start": datetime.now().isoformat(timespec="seconds"),
            "model_path": model_path,
            "video_src": self.video_src,
            "output_file": self.output_file,
            "use_tiling": self.use_tiling,
            "tile_padding": self.tile_padding,
            "conf_threshold": CONF_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD,
            "display_width": DISPLAY_WIDTH,
            "display_height": DISPLAY_HEIGHT,
            "ml_input_width": self.in_w,
            "ml_input_height": self.in_h,
            "npu_available": NPU_AVAILABLE,
            "system_monitor_interval_sec": SYSTEM_MONITOR_INTERVAL,
            "line_start": list(LINE_START),
            "line_end": list(LINE_END),
        }
    


    def _get_tile_bounds(self):
        """Right-half ROI with optional left padding (display coordinates)."""
        # Ensure the tile extends `tile_padding` pixels to the left of the crossing line
        x0 = max(0, min(DISPLAY_WIDTH // 2, LINE_START[0]) - self.tile_padding)
        return x0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT

    def _map_boxes_tile_to_full_ml(self, boxes, tile_x0, tile_y0):
        """Map detections from tile pixel space to full-frame ML coordinates."""
        if not boxes:
            return boxes
        ml_scale_x = self.in_w / DISPLAY_WIDTH
        ml_scale_y = self.in_h / DISPLAY_HEIGHT
        mapped = []
        for x1, y1, x2, y2 in boxes:
            dx1 = tile_x0 + x1
            dy1 = tile_y0 + y1
            dx2 = tile_x0 + x2
            dy2 = tile_y0 + y2
            mapped.append([
                int(dx1 * ml_scale_x),
                int(dy1 * ml_scale_y),
                int(dx2 * ml_scale_x),
                int(dy2 * ml_scale_y),
            ])
        if self.debug_mapping and mapped:
            logger.debug(
                f"[MAP] tile_x0={tile_x0} tile_y0={tile_y0} "
                f"tile_box={boxes[0]} mapped_ml={mapped[0]}"
            )
        return mapped

    def _map_boxes_tile_to_display(self, boxes, tile_x0, tile_y0):
        """Map detections from tile pixel space directly to display coordinates."""
        if not boxes:
            return boxes
        mapped = []
        for x1, y1, x2, y2 in boxes:
            mapped.append([
                int(tile_x0 + x1),
                int(tile_y0 + y1),
                int(tile_x0 + x2),
                int(tile_y0 + y2),
            ])
        return mapped

    def _map_boxes_ml_to_display(self, boxes):
        """Map full-frame ML coordinates to display coordinates."""
        if not boxes:
            return boxes
        scale_x = DISPLAY_WIDTH / self.in_w
        scale_y = DISPLAY_HEIGHT / self.in_h
        return [
            [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            ]
            for x1, y1, x2, y2 in boxes
        ]



    def _process_frame_for_detection(self, frame):
        logger.debug(f"_process_frame_for_detection called for frame {self.frame_count}, frame id: {id(frame)}")

        frame_display = frame

        if self.use_tiling:
            tile_x0, tile_y0, tile_x1, tile_y1 = self._get_tile_bounds()
            frame_tile = frame_display[tile_y0:tile_y1, tile_x0:tile_x1]
            tile_h, tile_w = frame_tile.shape[:2]
            
            person_boxes, person_scores, person_classes = self.detector.predict(
                frame_tile, tile_h, tile_w, profiler=self.profiler if self.profiling_enabled else None
            )
            
            person_boxes_display = self._map_boxes_tile_to_display(person_boxes, tile_x0, tile_y0)
            person_boxes = self._map_boxes_tile_to_full_ml(person_boxes, tile_x0, tile_y0)
        else:
            frame_ml = frame_display
            person_boxes, person_scores, person_classes = self.detector.predict(
                frame_ml, self.in_h, self.in_w, profiler=self.profiler if self.profiling_enabled else None
            )
            
            person_boxes_display = self._map_boxes_ml_to_display(person_boxes)

        with self.profiler.stage("tracking") if self.profiling_enabled else open(os.devnull, 'w') as _:
            tracked_objects, _ = self.tracker.update(person_boxes)
        
        with self.profiler.stage("line_crossing") if self.profiling_enabled else open(os.devnull, 'w') as _:
            crossing_count = self.counter.update(tracked_objects, self.tracker, person_boxes)
        logger.debug(f"Frame {self.frame_count}: Tracked {len(tracked_objects)} objects, crossings: {crossing_count}")

        with self.lock:
            self.person_boxes = person_boxes
            self.person_boxes_display = person_boxes_display
            self.person_scores = person_scores
            self.tracked_objects = tracked_objects
            self.crossing_count = crossing_count



    def _draw_overlay_cv(self, frame):
        with self.lock:
            boxes = list(self.person_boxes_display)
            tracked = dict(self.tracked_objects)
            crossings = self.crossing_count
        tile_bounds = self._get_tile_bounds() if self.use_tiling else None
        return draw_overlay_cv(
            frame, boxes, tracked, crossings,
            LINE_START, LINE_END,
            DISPLAY_WIDTH, DISPLAY_HEIGHT, self.in_w, self.in_h,
            tile_bounds=tile_bounds,
        )

    def _on_new_sample(self, appsink):
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            now = time.time()
            self.fps = 10 / (now - self.last_fps_time) if (now - self.last_fps_time) > 0 else 0
            self.actual_fps = self.fps if self.fps > 0 else 30.0
            self.last_fps_time = now
            
        with self.profiler.stage("appsink") if self.profiling_enabled else open(os.devnull, 'w') as _:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            
        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        fmt = structure.get_value("format")

        with self.profiler.stage("color_convert") if self.profiling_enabled else open(os.devnull, 'w') as _:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                logger.warning("Could not map GStreamer frame buffer")
                return Gst.FlowReturn.OK

            try:
                if fmt == "RGB":
                    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=map_info.data).copy()
                elif fmt in ("BGRx", "RGBx", "RGBA", "BGRA"):
                    frame = np.ndarray((height, width, 4), dtype=np.uint8, buffer=map_info.data)[:, :, :3].copy()
                else:
                    logger.warning(f"Unsupported GStreamer frame format: {fmt}")
                    return Gst.FlowReturn.OK
            finally:
                buf.unmap(map_info)

        logger.debug(f"Processing frame {self.frame_count} synchronously, frame id: {id(frame)}")
        self._process_frame_for_detection(frame)

        if self.frame_count % 30 == 1:
            with self.lock:
                det = len(self.person_boxes)
                trk = len(self.tracked_objects)
                c_count = self.crossing_count
            logger.info(f"[Video Frame {self.video_frame_count}] det:{det} trk:{trk} cross:{c_count} fps:{self.video_fps:.1f} (AI running at {self.fps:.1f} fps)")

        return Gst.FlowReturn.OK

    def _draw_overlay(self, overlay, ctx, ts, dur):
        self.video_frame_count += 1
        if self.video_frame_count % 30 == 0:
            now = time.time()
            self.video_fps = 30 / (now - self.video_last_fps_time) if (now - self.video_last_fps_time) > 0 else 0
            self.video_last_fps_time = now
            
        with self.profiler.stage("overlay") if self.profiling_enabled else open(os.devnull, 'w') as _:
            with self.lock:
                boxes = list(self.person_boxes_display)
                tracked = dict(self.tracked_objects)
                crossings = self.crossing_count
            tile_bounds = self._get_tile_bounds() if self.use_tiling else None
            draw_overlay_cairo(
                ctx, boxes, tracked, crossings,
                LINE_START, LINE_END,
                DISPLAY_WIDTH, DISPLAY_HEIGHT, self.in_w, self.in_h,
                tile_bounds=tile_bounds,
            )

    def _build_pipeline(self):
        from src.gst_pipeline import build_pipeline, handle_bus_message
        
        def bus_callback(bus, msg):
            handle_bus_message(bus, msg, self.pipeline, self.loop)
            
        self.pipeline = build_pipeline(
            video_src=self.video_src,
            is_camera=self.is_camera,
            output_file=self.output_file,
            display_w=DISPLAY_WIDTH,
            display_h=DISPLAY_HEIGHT,
            draw_cb=self._draw_overlay,
            sample_cb=self._on_new_sample,
            bus_cb=bus_callback,
            has_cairo=(cairo is not None),
            profiler=self.profiler if self.profiling_enabled else None
        )

    def _run_cv2_fallback(self):
        logger.info("GStreamer unavailable. Running OpenCV/TFLite fallback for local debugging.")
        cap = cv2.VideoCapture(self.video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.video_src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        output_fps = src_fps if src_fps and src_fps > 0 else 30.0
        writer = None
        if self.output_file:
            fourcc = cv2.VideoWriter_fourcc(*WINDOWS_OUTPUT_CODEC)
            writer = cv2.VideoWriter(
                self.output_file,
                fourcc,
                output_fps,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            )

        self.system_monitor.start()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_budget_ms = (1000.0 / fps) if (REALTIME_PLAYBACK and not self.is_camera and fps > 0) else 0

            while cap.isOpened():
                frame_start = time.perf_counter()
                with self.profiler.stage("decode") if self.profiling_enabled else open(os.devnull, 'w') as _:
                    ok, frame = cap.read()
                if not ok:
                    logger.info("End of stream")
                    break

                if frame.shape[1] != DISPLAY_WIDTH or frame.shape[0] != DISPLAY_HEIGHT:
                    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    now = time.time()
                    self.fps = 10 / (now - self.last_fps_time) if (now - self.last_fps_time) > 0 else 0
                    self.actual_fps = self.fps if self.fps > 0 else output_fps
                    self.last_fps_time = now

                self._process_frame_for_detection(frame)
                
                with self.profiler.stage("overlay") if self.profiling_enabled else open(os.devnull, 'w') as _:
                    overlay_frame = self._draw_overlay_cv(frame.copy())

                if writer is not None:
                    with self.profiler.stage("output_write") if self.profiling_enabled else open(os.devnull, 'w') as _:
                        writer.write(overlay_frame)

                if self.preview_enabled:
                    preview_frame = overlay_frame
                    if overlay_frame.shape[1] > PREVIEW_WIDTH or overlay_frame.shape[0] > PREVIEW_HEIGHT:
                        preview_frame = cv2.resize(overlay_frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                    cv2.imshow("Line Crossing Debug", preview_frame)
                    elapsed_ms = (time.perf_counter() - frame_start) * 1000
                    sleep_ms = max(1, int(frame_budget_ms - elapsed_ms)) if frame_budget_ms > 0 else 1
                    if cv2.waitKey(sleep_ms) & 0xFF == ord("q"):
                        logger.info("Stopped by user")
                        break
                elif frame_budget_ms > 0:
                    elapsed_ms = (time.perf_counter() - frame_start) * 1000
                    remaining = frame_budget_ms - elapsed_ms
                    if remaining > 0:
                        time.sleep(remaining / 1000)

                if self.frame_count % 30 == 1:
                    with self.lock:
                        det = len(self.person_boxes)
                        trk = len(self.tracked_objects)
                        c_count = self.crossing_count
                    logger.info(f"[{self.frame_count}] det:{det} trk:{trk} cross:{c_count} fps:{self.fps:.1f}")
        finally:
            self.system_monitor.stop()
            cap.release()
            if writer is not None:
                writer.release()
            if self.preview_enabled:
                cv2.destroyAllWindows()
            self.metrics_recorder.save(self.metrics_csv_path, self.metrics_json_path)
            logger.info(f"Done! Total crossings: {self.crossing_count}")
            
            if self.profiling_enabled:
                logger.info("\n" + self.profiler.format_profiling_summary(self.actual_fps))

    def _start_rtsp_server(self):
        from src.gst_pipeline import start_rtsp_server
        self.server = start_rtsp_server(GST_UDP_PORT, GST_RTSP_PORT)

    def run(self):
        logger.info(f"Input: {'Camera' if self.is_camera else 'Video'} {self.video_src}")
        logger.info(f"Output: {self.output_file if self.output_file else 'Display only'}")
        logger.info(f"Line: {LINE_START} -> {LINE_END}")
        if self.use_tiling:
            tile_x0, _, tile_x1, _ = self._get_tile_bounds()
            logger.info(f"Tiling: right-half ROI x={tile_x0}-{tile_x1} (padding={self.tile_padding})")
        else:
            logger.info("Tiling: disabled")

        if not GST_AVAILABLE:
            self._run_cv2_fallback()
            return
        
        self.loop = GLib.MainLoop()
        self._build_pipeline()
        self._start_rtsp_server()
        
        logger.info("Starting pipeline...")
        self.system_monitor.start()
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.system_monitor.stop()
            self.pipeline.set_state(Gst.State.NULL)
            self.metrics_recorder.save(self.metrics_csv_path, self.metrics_json_path)
            logger.info(f"Done! Total crossings: {self.crossing_count}")
            
            if self.profiling_enabled:
                report_fps = self.source_fps if self.source_fps > 0 else self.video_fps
                logger.info("\n" + self.profiler.format_profiling_summary(report_fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL)
    parser.add_argument('-v', '--video', default=DEFAULT_SOURCE)
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--use-tiling', action=argparse.BooleanOptionalAction, default=USE_TILING,
                        help='Detect on right-half tile where the line is (default: on)')
    parser.add_argument('--tile-padding', type=int, default=TILE_PADDING,
                        help='Pixels to extend tile left of frame center (default: 120)')
    parser.add_argument('--metrics-csv', default='outputs/metrics/metrics.csv',
                        help='CSV file to write collected metrics (default: outputs/metrics/metrics.csv)')
    parser.add_argument('--metrics-json', default='outputs/metrics/metrics.json',
                        help='JSON file to write collected metrics (default: outputs/metrics/metrics.json)')
    parser.add_argument('--preview', action='store_true', default=False,
                        help='Enable local display output (only use if board has a monitor attached)')
    parser.add_argument('--debug-mapping', action='store_true',
                        help='Print one tile-to-ML box mapping per inference frame')
    parser.add_argument('--profile', action='store_true',
                        help='Enable detailed latency profiling')
    args = parser.parse_args()
    
    if GST_AVAILABLE:
        Gst.init(None)
        # Required for Wayland/KMS display on i.MX boards
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/0"
        os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    else:
        logger.info("GStreamer bindings not available on this machine; using OpenCV fallback.")
    
    LineCrossingDetector(
        args.model, args.video, args.output,
        use_tiling=args.use_tiling,
        tile_padding=args.tile_padding,
        metrics_csv_path=args.metrics_csv,
        metrics_json_path=args.metrics_json,
        preview_enabled=args.preview,
        debug_mapping=args.debug_mapping,
        profiling_enabled=(PROFILING_ENABLED or args.profile),
    ).run()
