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
from contextlib import contextmanager
import numpy as np
import cv2
from collections import OrderedDict
from datetime import datetime
import argparse
import csv
import json
from pathlib import Path
try:
    import cairo
except ImportError:
    cairo = None
try:
    import psutil
except ImportError:
    psutil = None



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
from src.metrics import MetricsRecorder, PipelineProfiler
from src.tracker import CentroidTracker, LineCrossingCounter
from src.model import TFLitePersonDetector, NPU_AVAILABLE

# ============================================================================
# MAIN LINE CROSSING DETECTOR
# ============================================================================
class LineCrossingDetector:
    
    def __init__(self, model_path, video_src, output_file=None, use_tiling=USE_TILING, tile_padding=TILE_PADDING,
                 metrics_csv_path="outputs/metrics/metrics.csv", metrics_json_path="outputs/metrics/metrics.json", preview_enabled=True,
                 debug_mapping=False):
        self.video_src = video_src
        self.output_file = output_file
        self.use_tiling = use_tiling
        self.tile_padding = tile_padding
        self.metrics_csv_path = metrics_csv_path
        self.metrics_json_path = metrics_json_path
        self.preview_enabled = preview_enabled
        self.debug_mapping = debug_mapping
        # Detect if the source is a camera. On Linux it's '/dev/video*', on Windows it's an integer index like '0'
        self.is_camera = video_src.isdigit() or video_src.startswith("/dev/video")
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0
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
        self.profiler = PipelineProfiler([
            "decode",
            "resize",
            "inference",
            "nms",
            "tracking",
            "overlay_draw",
            "output_write",
            "imshow_display",
        ])
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
            "skip_frames": SKIP_FRAMES,
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
        x0 = max(0, DISPLAY_WIDTH // 2 - self.tile_padding)
        x0 = min(x0, LINE_START[0])
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

    def _on_pad_caps_notify(self, pad, pspec, element_name):
        """Callback when pad caps change."""
        caps = pad.get_current_caps()
        pad_name = pad.get_name()
        logger.debug(f"=== [CAPS CHANGE] {element_name}:{pad_name} ===")
        if caps:
            logger.debug(caps.to_string())
        else:
            logger.debug("(no caps)")
    
    def _monitor_element_pads(self, element, element_name):
        """Monitor both src and sink pads of an element for caps changes."""
        sink_pad = element.get_static_pad("sink")
        if sink_pad:
            sink_pad.connect("notify::caps", self._on_pad_caps_notify, element_name)
        
        src_pad = element.get_static_pad("src")
        if src_pad:
            src_pad.connect("notify::caps", self._on_pad_caps_notify, element_name)
    
    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End of stream")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"{err}: {debug}")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            if message.src == self.pipeline and new == Gst.State.PLAYING:
                logger.info("=== [PIPELINE PLAYING] Printing current caps for key elements ===")
                # Print caps for key elements after pipeline is playing
                self._print_key_element_caps()
    
    def _print_key_element_caps(self):
        """Print caps for all key elements we're interested in."""
        # Helper to print caps for a pad
        def print_pad_caps(element, pad_name, element_label):
            pad = element.get_static_pad(pad_name)
            if pad:
                caps = pad.get_current_caps()
                logger.debug(f"=== {element_label} {pad_name.upper()} CAPS ===")
                if caps:
                    logger.debug(caps.to_string())
                else:
                    logger.debug("(no caps yet)")
        
        # Iterate through all elements to find the ones we want
        it = self.pipeline.iterate_elements()
        while True:
            result, element = it.next()
            if result != Gst.IteratorResult.OK:
                break
            
            element_name = element.get_name()
            
            # Check for elements we care about
            if "vpudec" in element_name:
                print_pad_caps(element, "src", "vpudec")
            elif "imxvideoconvert" in element_name:
                print_pad_caps(element, "sink", element_name)
                print_pad_caps(element, "src", element_name)
            elif "vpuenc" in element_name:
                print_pad_caps(element, "sink", element_name)
            elif element_name == "ml_sink":
                print_pad_caps(element, "sink", "appsink (ml_sink)")
            elif element_name == "overlay":
                print_pad_caps(element, "sink", "cairooverlay")
                print_pad_caps(element, "src", "cairooverlay")



    def _process_frame_for_detection(self, frame):
        logger.debug(f"_process_frame_for_detection called for frame {self.frame_count}, frame id: {id(frame)}")
        
        if self.frame_count % (SKIP_FRAMES + 1) != 1:
            logger.debug(f"Skipping frame {self.frame_count} (not inference frame)")
            return

        frame_display = frame

        if self.use_tiling:
            tile_x0, tile_y0, tile_x1, tile_y1 = self._get_tile_bounds()
            frame_tile = frame_display[tile_y0:tile_y1, tile_x0:tile_x1]
            tile_h, tile_w = frame_tile.shape[:2]
            
            person_boxes, person_scores, person_classes = self.detector.predict(frame_tile, tile_h, tile_w)
            
            person_boxes_display = self._map_boxes_tile_to_display(person_boxes, tile_x0, tile_y0)
            person_boxes = self._map_boxes_tile_to_full_ml(person_boxes, tile_x0, tile_y0)
        else:
            frame_ml = frame_display
            person_boxes, person_scores, person_classes = self.detector.predict(frame_ml, self.in_h, self.in_w)
            
            person_boxes_display = self._map_boxes_ml_to_display(person_boxes)

        tracked_objects, _ = self.tracker.update(person_boxes)
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

        if self.use_tiling:
            tile_x0, tile_y0, tile_x1, tile_y1 = self._get_tile_bounds()
            cv2.rectangle(frame, (tile_x0, tile_y0), (tile_x1, tile_y1), (128, 128, 255), 2)

        cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 3)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        scale_x = DISPLAY_WIDTH / self.in_w
        scale_y = DISPLAY_HEIGHT / self.in_h
        for objectID, centroid in tracked.items():
            cx = int(centroid[0] * scale_x)
            cy = int(centroid[1] * scale_y)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"ID{objectID}",
                (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Crossings: {crossings}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        return frame

    def _on_new_sample(self, appsink):
        self.frame_count += 1
        self.profiler.mark_frame()
        
        if self.frame_count % 10 == 0:
            now = time.time()
            self.fps = 10 / (now - self.last_fps_time) if (now - self.last_fps_time) > 0 else 0
            self.actual_fps = self.fps if self.fps > 0 else 30.0
            self.last_fps_time = now
            
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
            
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        
        logger.debug(f"Passing frame {self.frame_count} to _process_frame_for_detection, frame id: {id(frame)}")
        self._process_frame_for_detection(frame)

        if self.frame_count % 30 == 1:
            with self.lock:
                det = len(self.person_boxes)
                trk = len(self.tracked_objects)
                c_count = self.crossing_count
            logger.info(f"[{self.frame_count}] det:{det} trk:{trk} cross:{c_count} fps:{self.fps:.1f}")

        return Gst.FlowReturn.OK

    def _draw_overlay(self, overlay, ctx, ts, dur):
        scale_x = DISPLAY_WIDTH / self.in_w
        scale_y = DISPLAY_HEIGHT / self.in_h

        with self.lock:
            boxes = list(self.person_boxes_display)
            tracked = dict(self.tracked_objects)
            crossings = self.crossing_count

        if self.use_tiling:
            tile_x0, tile_y0, tile_x1, tile_y1 = self._get_tile_bounds()
            ctx.set_source_rgb(1.0, 0.5, 0.5)
            ctx.set_line_width(2)
            ctx.rectangle(tile_x0, tile_y0, tile_x1 - tile_x0, tile_y1 - tile_y0)
            ctx.stroke()
            
        ctx.set_source_rgb(1.0, 1.0, 0.0)
        ctx.set_line_width(3)
        ctx.move_to(LINE_START[0], LINE_START[1])
        ctx.line_to(LINE_END[0], LINE_END[1])
        ctx.stroke()
        
        ctx.set_source_rgb(0.0, 0.0, 1.0)
        ctx.set_line_width(2)
        for box in boxes:
            x1, y1, x2, y2 = box
            ctx.rectangle(x1, y1, x2-x1, y2-y1)
            ctx.stroke()
            
        for (objectID, centroid) in tracked.items():
            cx = int(centroid[0] * scale_x)
            cy = int(centroid[1] * scale_y)
            ctx.set_source_rgb(0.0, 1.0, 0.0)
            ctx.arc(cx, cy, 8, 0, 2 * 3.14159)
            ctx.fill()
            
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(16)
            ctx.move_to(cx - 10, cy - 10)
            ctx.show_text(f"ID{objectID}")
            
        ctx.set_source_rgb(1.0, 1.0, 0.0)
        ctx.set_font_size(32)
        ctx.move_to(10, 40)
        ctx.show_text(f"Crossings: {crossings}")

    def _build_pipeline(self):
        if self.is_camera:
            source = f"v4l2src device={self.video_src} ! video/x-raw,framerate=30/1"
        else:
            # Use the exact verified hardware-accelerated decode pipeline
            source = f"filesrc location={self.video_src} ! qtdemux ! h264parse ! vpudec ! imxvideoconvert_g2d ! video/x-raw,format=BGRx"
            
        sink_str = "fakesink sync=false"  # Temporarily use fakesink instead of kmssink

        if self.output_file:
            pipeline_str = (
                f"{source} ! "
                f"imxvideoconvert_g2d ! video/x-raw,width={DISPLAY_WIDTH},height={DISPLAY_HEIGHT},format=BGRx ! "
                f"tee name=t "
                f"t. ! queue max-size-buffers=2 leaky=downstream ! cairooverlay name=overlay ! "
                f"tee name=out "
                f"out. ! queue max-size-buffers=10 leaky=downstream ! "
                f"imxvideoconvert_g2d ! video/x-raw,format=RGB16 ! "
                f"vpuenc_h264 bitrate={GST_BITRATE_FILE} ! h264parse ! mp4mux ! filesink location={self.output_file} "
                f"out. ! queue max-size-buffers=2 leaky=downstream ! {sink_str} "
                f"out. ! queue max-size-buffers=2 leaky=downstream ! "
                f"imxvideoconvert_g2d ! video/x-raw,format=RGB16 ! "
                f"vpuenc_h264 bitrate={GST_BITRATE_STREAM} ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port={GST_UDP_PORT} "
                f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! video/x-raw,format=RGB ! appsink name=ml_sink emit-signals=true drop=true max-buffers=2 sync=false"
            )
        else:
            pipeline_str = (
                f"{source} ! "
                f"imxvideoconvert_g2d ! video/x-raw,width={DISPLAY_WIDTH},height={DISPLAY_HEIGHT},format=BGRx ! "
                f"tee name=t "
                f"t. ! queue max-size-buffers=2 leaky=downstream ! cairooverlay name=overlay ! "
                f"tee name=out "
                f"out. ! queue max-size-buffers=2 leaky=downstream ! {sink_str} "
                f"out. ! queue max-size-buffers=2 leaky=downstream ! "
                f"imxvideoconvert_g2d ! video/x-raw,format=RGB16 ! "
                f"vpuenc_h264 bitrate={GST_BITRATE_STREAM} ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port={GST_UDP_PORT} "
                f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! video/x-raw,format=RGB ! appsink name=ml_sink emit-signals=true drop=true max-buffers=2 sync=false"
            )
            
        logger.info(f"Pipeline: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        overlay = self.pipeline.get_by_name("overlay")
        if overlay and cairo is not None:
            overlay.connect("draw", self._draw_overlay)
        elif overlay:
            logger.warning("Python cairo not available; Cairo overlay callback disabled")
            
        ml_sink = self.pipeline.get_by_name("ml_sink")
        if ml_sink:
            ml_sink.connect("new-sample", self._on_new_sample)
            
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

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
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.info("End of stream")
                    break

                if frame.shape[1] != DISPLAY_WIDTH or frame.shape[0] != DISPLAY_HEIGHT:
                    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                self.frame_count += 1
                self.profiler.mark_frame()
                if self.frame_count % 10 == 0:
                    now = time.time()
                    self.fps = 10 / (now - self.last_fps_time) if (now - self.last_fps_time) > 0 else 0
                    self.actual_fps = self.fps if self.fps > 0 else output_fps
                    self.last_fps_time = now

                self._process_frame_for_detection(frame)
                overlay_frame = self._draw_overlay_cv(frame.copy())

                if writer is not None:
                    writer.write(overlay_frame)

                if self.preview_enabled:
                    preview_frame = overlay_frame
                    if overlay_frame.shape[1] > PREVIEW_WIDTH or overlay_frame.shape[0] > PREVIEW_HEIGHT:
                        preview_frame = cv2.resize(overlay_frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                    cv2.imshow("Line Crossing Debug", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Stopped by user")
                        break

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

    def _start_rtsp_server(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(GST_RTSP_PORT)
        
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_shared(True)
        # Bridge the UDP stream from the main pipeline into the RTSP server
        factory.set_launch(
            f'( udpsrc port={GST_UDP_PORT} caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            'rtph264pay name=pay0 pt=96 )'
        )
        
        mounts = self.server.get_mount_points()
        mounts.add_factory("/video", factory)
        self.server.attach(None)
        logger.info(f"[RTSP] Server running at rtsp://0.0.0.0:{GST_RTSP_PORT}/video")

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
    ).run()
