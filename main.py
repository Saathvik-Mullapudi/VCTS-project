#!/usr/bin/env python3
import os
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

# Try tflite-runtime first (for i.MX8)
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    NPU_AVAILABLE = True
    print("[OK] Using tflite-runtime (i.MX8 mode)")
except ImportError:
    # Try TensorFlow (for Windows laptop)
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        # For TensorFlow >= 2.14, experimental.load_delegate is still valid
        try:
            load_delegate = tf.lite.experimental.load_delegate
        except AttributeError:
            load_delegate = None  # N/A on Windows
        NPU_AVAILABLE = False
        print("[OK] Using TensorFlow for TFLite (Windows mode)")
    except ImportError as e:
        print(f"[ERROR] Need either tensorflow or tflite-runtime installed! Error: {e}")
        exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_MODEL = "models/yolov8n_saved_model/yolov8n_full_integer_quant.tflite"
DEFAULT_SOURCE = "data/videos/vid.mp4"
DEFAULT_OUTPUT = "outputs/videos/output.mp4"

# Resolution
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080
ML_INPUT_WIDTH, ML_INPUT_HEIGHT = 640, 360
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720

# Performance
SKIP_FRAMES = 2  # Process 1 frame, skip 2 (runs inference on every 3rd frame)

# Detection parameters
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

# Person class (COCO class 0)
PERSON_CLASS_IDS = [0]

# Tracking
MAX_DISAPPEARED = 15
MAX_TRACK_DISTANCE = 120

# Line crossing for data/videos/vid.mp4 (1920x1080 display space)
LINE_START = (691, 496)
LINE_END = (1593, 648)

# Tiling - run inference on the right half where the crossing line lives
USE_TILING = True
TILE_PADDING = 120  # pixels to extend the tile left of frame center

# Profiling
PROFILE_PRINT_EVERY = 30

# System monitoring
SYSTEM_MONITOR_INTERVAL = 1.0

# Windows fallback
WINDOWS_OUTPUT_CODEC = "mp4v"

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



class PipelineProfiler:
    def __init__(self, stage_names):
        self.stage_names = list(stage_names)
        self.totals_ns = {name: 0 for name in self.stage_names}
        self.counts = {name: 0 for name in self.stage_names}
        self.frame_count = 0
        self._current_frame_ns = {name: 0 for name in self.stage_names}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            self.totals_ns[name] += elapsed
            self._current_frame_ns[name] += elapsed
            self.counts[name] += 1

    def mark_frame(self):
        self.frame_count += 1

    def average_ms(self, name):
        count = self.counts.get(name, 0)
        if count == 0:
            return 0.0
        return self.totals_ns[name] / count / 1e6

    def snapshot(self):
        return {name: self.average_ms(name) for name in self.stage_names}

    def format_summary(self):
        parts = [f"{name}:{self.average_ms(name):.2f}ms" for name in self.stage_names]
        return " ".join(parts)

    def snapshot_frame(self):
        snapshot = {
            f"{name}_ms": self._current_frame_ns[name] / 1e6
            for name in self.stage_names
        }
        snapshot["stages_total_ms"] = sum(snapshot.values())
        for name in self.stage_names:
            self._current_frame_ns[name] = 0
        return snapshot


class MetricsRecorder:
    def __init__(self, config_metadata):
        self.config_metadata = dict(config_metadata)
        self.frame_records = []
        self.system_records = []

    def add_frame_record(self, record):
        self.frame_records.append(dict(record))

    def add_system_record(self, record):
        self.system_records.append(dict(record))

    def save(self, csv_path, json_path):
        csv_path = Path(csv_path) if csv_path else None
        json_path = Path(json_path) if json_path else None
        if csv_path:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_csv(csv_path)
        if json_path:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_json(json_path)

    def _save_csv(self, csv_path):
        rows = []
        for record in self.frame_records:
            row = self._base_row("frame", record)
            row.update(record)
            rows.append(row)
        for record in self.system_records:
            row = self._base_row("system", record)
            row.update(record)
            rows.append(row)

        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["record_type"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_json(self, json_path):
        payload = {
            "metadata": dict(self.config_metadata),
            "summary": {
                "frame_records": len(self.frame_records),
                "system_records": len(self.system_records),
                "timestamp_end": datetime.now().isoformat(timespec="seconds"),
            },
            "frame_metrics": self.frame_records,
            "system_metrics": self.system_records,
        }
        start_ts = self.config_metadata.get("timestamp_start")
        if start_ts:
            try:
                start_dt = datetime.fromisoformat(start_ts)
                payload["summary"]["elapsed_seconds"] = (datetime.now() - start_dt).total_seconds()
            except Exception:
                pass
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _base_row(self, record_type, record):
        row = {
            "record_type": record_type,
            **self.config_metadata,
        }
        row.update(record)
        return row


class SystemMonitor:
    def __init__(self, interval_sec=1.0, frame_number_fn=None, on_sample=None):
        self.interval_sec = max(float(interval_sec), 0.1)
        self.records = []
        self.latest = {}
        self._frame_number_fn = frame_number_fn
        self._on_sample = on_sample
        self._stop_event = threading.Event()
        self._thread = None
        self._last_energy_sample = {}
        self._prime_cpu_metrics()

    def _prime_cpu_metrics(self):
        if psutil is None:
            return
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None, percpu=True)
        except Exception:
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval_sec + 0.5)

    def _run(self):
        next_sample = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_sample:
                if self._stop_event.wait(next_sample - now):
                    break
            else:
                sample = self.sample()
                self.latest = sample
                self.records.append(sample)
                if self._on_sample is not None:
                    try:
                        self._on_sample(sample)
                    except Exception:
                        pass
                print(f"[SYS] {self.format_sample(sample)}")
                next_sample = time.monotonic() + self.interval_sec

    def sample(self):
        timestamp = datetime.now().isoformat(timespec="seconds")
        frame_number = None
        if self._frame_number_fn is not None:
            try:
                frame_number = int(self._frame_number_fn())
            except Exception:
                frame_number = None
        cpu_total = self._sample_cpu_total()
        cpu_per_core = self._sample_cpu_per_core()
        ram_percent = self._sample_ram_percent()
        temperature_c, temperature_source = self._sample_temperature_c()
        gpu_util_percent, gpu_source = self._sample_gpu_utilization()
        npu_util_percent, npu_source = self._sample_npu_utilization()
        power_w, power_source = self._sample_power_w()

        return {
            "timestamp": timestamp,
            "frame_number": frame_number,
            "cpu_total_percent": cpu_total,
            "cpu_per_core_percent": cpu_per_core,
            "ram_percent": ram_percent,
            "temperature_c": temperature_c,
            "temperature_source": temperature_source,
            "gpu_utilization_percent": gpu_util_percent,
            "gpu_source": gpu_source,
            "npu_utilization_percent": npu_util_percent,
            "npu_source": npu_source,
            "power_w": power_w,
            "power_source": power_source,
        }

    def format_sample(self, sample):
        cpu_total = self._fmt_percent(sample.get("cpu_total_percent"))
        cpu_cores = sample.get("cpu_per_core_percent") or []
        cpu_cores_str = ", ".join(self._fmt_percent(value) for value in cpu_cores) if cpu_cores else "n/a"
        ram = self._fmt_percent(sample.get("ram_percent"))
        temp = self._fmt_temp(sample.get("temperature_c"), sample.get("temperature_source"))
        gpu = self._fmt_percent(sample.get("gpu_utilization_percent"), sample.get("gpu_source"))
        npu = self._fmt_percent(sample.get("npu_utilization_percent"), sample.get("npu_source"))
        power = self._fmt_power(sample.get("power_w"), sample.get("power_source"))
        return (
            f"cpu={cpu_total} cores=[{cpu_cores_str}] ram={ram} "
            f"temp={temp} gpu={gpu} npu={npu} power={power}"
        )

    def _sample_cpu_total(self):
        if psutil is None:
            return None
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def _sample_cpu_per_core(self):
        if psutil is None:
            return []
        try:
            return [float(value) for value in psutil.cpu_percent(interval=None, percpu=True)]
        except Exception:
            return []

    def _sample_ram_percent(self):
        if psutil is None:
            return None
        try:
            return float(psutil.virtual_memory().percent)
        except Exception:
            return None

    def _sample_temperature_c(self):
        temps = []
        if psutil is not None:
            try:
                for sensor_name, entries in (psutil.sensors_temperatures() or {}).items():
                    for entry in entries:
                        current = getattr(entry, "current", None)
                        if current is None:
                            continue
                        label = (getattr(entry, "label", "") or sensor_name or "").strip()
                        temps.append((float(current), label))
            except Exception:
                pass
        if not temps:
            temps.extend(self._sample_sysfs_temperatures())
        if not temps:
            return None, None
        value, label = max(temps, key=lambda item: item[0])
        return value, label or None

    def _sample_sysfs_temperatures(self):
        temps = []
        thermal_root = Path("/sys/class/thermal")
        if not thermal_root.exists():
            return temps
        for zone in thermal_root.glob("thermal_zone*"):
            temp_file = zone / "temp"
            if not temp_file.exists():
                continue
            raw = self._read_text(temp_file)
            value = self._parse_scaled_value(raw, scale=1000.0)
            if value is not None:
                temps.append((value, zone.name))
        return temps

    def _sample_npu_utilization(self):
        return self._sample_devfreq_utilization(("npu", "neutron"))

    def _sample_gpu_utilization(self):
        return self._sample_devfreq_utilization(("gpu", "galcore"))

    def _sample_devfreq_utilization(self, keywords):
        candidates = []
        roots = [Path("/sys/class/devfreq"), Path("/sys/devices")]
        for root in roots:
            if not root.exists():
                continue
            try:
                for entry in root.iterdir():
                    if not entry.is_dir():
                        continue
                    name = self._read_text(entry / "name") or entry.name
                    if any(keyword in name.lower() for keyword in keywords):
                        candidates.append(entry)
            except Exception:
                continue
        for devfreq_dir in candidates:
            util = self._read_metric_percent(devfreq_dir, ["utilization", "load"])
            if util is not None:
                return util, str(devfreq_dir)
            busy = self._read_scaled_number(devfreq_dir / "busy_time")
            total = self._read_scaled_number(devfreq_dir / "total_time")
            if busy is not None and total and total > 0:
                return (busy / total) * 100.0, str(devfreq_dir)
        return None, None

    def _sample_power_w(self):
        power_dirs = []
        for root in (Path("/sys/class/powercap"), Path("/sys/class/power_supply"), Path("/sys/class/hwmon")):
            if not root.exists():
                continue
            try:
                for entry in root.iterdir():
                    if entry.is_dir():
                        power_dirs.append(entry)
            except Exception:
                continue

        for directory in power_dirs:
            direct = self._read_power_direct(directory)
            if direct is not None:
                return direct, str(directory)

        energy_sample = self._sample_energy_based_power(power_dirs)
        if energy_sample is not None:
            return energy_sample

        return None, None

    def _read_power_direct(self, directory):
        watts = self._read_metric_watts(directory, ["power_now", "power_uw", "power1_input", "power2_input"])
        if watts is not None:
            return watts

        current_ua = self._read_scaled_number(directory / "current_now")
        voltage_uv = self._read_scaled_number(directory / "voltage_now")
        if current_ua is not None and voltage_uv is not None:
            return (current_ua * voltage_uv) / 1e12

        return None

    def _sample_energy_based_power(self, power_dirs):
        now = time.monotonic()
        for directory in power_dirs:
            energy_file = directory / "energy_uj"
            if not energy_file.exists():
                continue
            energy_uj = self._read_scaled_number(energy_file)
            if energy_uj is None:
                continue
            key = str(energy_file)
            previous = self._last_energy_sample.get(key)
            self._last_energy_sample[key] = (now, energy_uj)
            if previous is None:
                continue
            prev_time, prev_energy = previous
            elapsed = now - prev_time
            if elapsed <= 0:
                continue
            return ((energy_uj - prev_energy) / elapsed) / 1e6, key
        return None

    def _read_metric_percent(self, directory, names):
        for name in names:
            raw = self._read_text(directory / name)
            if raw is None:
                continue
            percent = self._parse_percentage(raw)
            if percent is not None:
                return percent
        return None

    def _read_metric_watts(self, directory, names):
        for name in names:
            raw = self._read_text(directory / name)
            if raw is None:
                continue
            watts = self._parse_scaled_value(raw, scale=1e6)
            if watts is not None:
                return watts
        return None

    def _read_scaled_number(self, path):
        raw = self._read_text(path)
        if raw is None:
            return None
        return self._parse_number(raw)

    def _read_text(self, path):
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except Exception:
            return None

    def _parse_number(self, raw):
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _parse_scaled_value(self, raw, scale):
        value = self._parse_number(raw)
        if value is None:
            return None
        return value / scale

    def _parse_percentage(self, raw):
        value = self._parse_number(raw)
        if value is None:
            return None
        if value <= 1.0:
            return value * 100.0
        return value

    def _fmt_percent(self, value, source=None):
        if value is None:
            return "n/a"
        formatted = f"{value:.1f}%"
        return f"{formatted} ({Path(source).name})" if source else formatted

    def _fmt_temp(self, value, source=None):
        if value is None:
            return "n/a"
        formatted = f"{value:.1f}C"
        return f"{formatted} ({Path(source).name})" if source else formatted

    def _fmt_power(self, value, source=None):
        if value is None:
            return "n/a"
        formatted = f"{value:.2f}W"
        return f"{formatted} ({Path(source).name})" if source else formatted

# ============================================================================
# CPU PREPROCESSOR (FROM OVERSPEED PROJECT)
# ============================================================================
class CPUPreprocessor:
    def __init__(self, target_width, target_height, input_dtype):
        self.target_width = target_width
        self.target_height = target_height
        self.input_dtype = input_dtype
        self.output_buffer = np.zeros((1, target_height, target_width, 3), dtype=input_dtype)
        
    def process(self, frame):
        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = self.target_height, self.target_width
        
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        if new_w != orig_w or new_h != orig_h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = frame
        
        if self.input_dtype == np.uint8:
            np.copyto(self.output_buffer[0], padded)
        elif self.input_dtype == np.int8:
            np.subtract(padded, 128, out=self.output_buffer[0], casting='unsafe')
        else:
            self.output_buffer[0] = padded.astype(np.float32) / 255.0
            
        return self.output_buffer, scale, pad_x, pad_y

# ============================================================================
# OPTIMIZED POSTPROCESSOR (FROM OVERSPEED PROJECT)
# ============================================================================
class OptimizedPostprocessor:
    def __init__(self, conf_threshold=0.3, nms_threshold=0.45, topk=500, class_ids=None):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.topk = topk
        self.class_ids = class_ids or [0]
    
    def _nms_numpy(self, boxes, scores, iou_threshold):
        if boxes.shape[0] == 0:
            return np.array([], dtype=int)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=int)
    
    def process(self, output_data, frame_shape, input_shape, scale_params, scale, pad_x, pad_y):
        out = output_data
        if scale_params['scale'] is not None:
            out = (out.astype(np.float32) - scale_params['zero_point']) * scale_params['scale']
        
        if len(out.shape) == 3:
            out = np.squeeze(out).T
        else:
            out = np.squeeze(out)
        
        if out.ndim == 1:
            out = np.expand_dims(out, 0)
        if out.shape[0] == 0 or out.shape[1] < 5:
            return [], [], []
        
        frame_h, frame_w = frame_shape
        in_h, in_w = input_shape
        
        class_scores = out[:, 4:]
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        
        conf_mask = max_scores > self.conf_threshold
        if not np.any(conf_mask):
            return [], [], []
        
        valid_indices = np.where(conf_mask)[0]
        valid_indices = valid_indices[np.isin(class_ids[valid_indices], self.class_ids)]
        if len(valid_indices) == 0:
            return [], [], []

        sel_scores = max_scores[valid_indices]
        sel_cls = class_ids[valid_indices]
        sel_preds = out[valid_indices, :4]
        
        if sel_scores.shape[0] > self.topk:
            topk_inds = np.argpartition(-sel_scores, self.topk)[:self.topk]
            sel_scores = sel_scores[topk_inds]
            sel_cls = sel_cls[topk_inds]
            sel_preds = sel_preds[topk_inds]
        
        box_scale_x = in_w if np.nanmax(sel_preds[:, [0, 2]]) <= 2.0 else 1.0
        box_scale_y = in_h if np.nanmax(sel_preds[:, [1, 3]]) <= 2.0 else 1.0

        cx = (sel_preds[:, 0] * box_scale_x - pad_x) / scale
        cy = (sel_preds[:, 1] * box_scale_y - pad_y) / scale
        bw = (sel_preds[:, 2] * box_scale_x) / scale
        bh = (sel_preds[:, 3] * box_scale_y) / scale
        
        x1 = np.clip(cx - bw / 2.0, 0, frame_w - 1)
        y1 = np.clip(cy - bh / 2.0, 0, frame_h - 1)
        x2 = np.clip(cx + bw / 2.0, 0, frame_w - 1)
        y2 = np.clip(cy + bh / 2.0, 0, frame_h - 1)
        
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        keep = self._nms_numpy(boxes, sel_scores.astype(np.float32), self.nms_threshold)
        
        if keep.size == 0:
            return [], [], []
        
        return boxes[keep].astype(np.int32).tolist(), sel_scores[keep].tolist(), sel_cls[keep].tolist()

# ============================================================================
# CENTROID TRACKER (FROM OVERSPEED PROJECT)
# ============================================================================
class CentroidTracker:
    def __init__(self, maxDisappeared=15, smoothing=0.75, maxDistance=120):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.crossed_status = {}
        self.smoothing = smoothing  # EMA factor: lower = smoother
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.crossed_status[self.nextObjectID] = {"side": None, "counted": False, "pending_frames": 0, "pending_side": None}
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.crossed_status:
            del self.crossed_status[objectID]

    def update(self, boxes):
        if len(boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, []

        inputCentroids = np.zeros((len(boxes), 2), dtype="int")
        inputFootPoints = []
        for (i, box) in enumerate(boxes):
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = box
            else:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)
            footY = int(y2)
            inputFootPoints.append((cX, footY))

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                # EMA smoothing to reduce centroid jitter
                old = self.objects[objectID]
                new = inputCentroids[col]
                smoothed = (int(old[0] * (1 - self.smoothing) + new[0] * self.smoothing),
                            int(old[1] * (1 - self.smoothing) + new[1] * self.smoothing))
                self.objects[objectID] = smoothed
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects, inputFootPoints

    def get_foot_point(self, objectID, boxes):
        for (i, box) in enumerate(boxes):
            x1, y1, x2, y2 = box
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            if np.linalg.norm(np.array([cX, cY]) - np.array(self.objects[objectID])) < 120:
                return (cX, int(y2))
        return None

# ============================================================================
# LINE CROSSING COUNTER
# ============================================================================
class LineCrossingCounter:
    def __init__(self, line_start, line_end, ml_width, ml_height):
        self.line_start = line_start
        self.line_end = line_end
        self.ml_width = ml_width
        self.ml_height = ml_height
        self.crossing_count = 0

    def get_side(self, point, line_start, line_end):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        return ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) > 0

    def get_distance(self, point, line_start, line_end):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        num = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))
        den = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        return num / den if den > 0 else 0

    def update(self, tracked_objects, tracker, boxes):
        # Scale line from display coords to the active model input space.
        sx = self.ml_width / DISPLAY_WIDTH
        sy = self.ml_height / DISPLAY_HEIGHT
        line_s = (self.line_start[0] * sx, self.line_start[1] * sy)
        line_e = (self.line_end[0]   * sx, self.line_end[1]   * sy)
        
        DEBOUNCE_FRAMES = 5  # Must be on new side for this many frames
        BUFFER_PIXELS = 20   # Spatial padding in ML coords (~60px in display space)
        
        for (objectID, centroid) in tracked_objects.items():
            foot_point = tracker.get_foot_point(objectID, boxes)
            if foot_point is None:
                continue
            
            current_side = self.get_side(foot_point, line_s, line_e)
            dist_to_line = self.get_distance(foot_point, line_s, line_e)
            
            state = tracker.crossed_status[objectID]
            
            if state["side"] is None:
                # First time seeing this object, just record the side
                state["side"] = current_side
                state["pending_frames"] = 0
                state["pending_side"] = None
                continue
            
            if state["counted"]:
                continue
                
            # If the centroid is inside the physical padding buffer, ignore it
            # This prevents jitter from crossing back and forth across the exact mathematical line
            if dist_to_line < BUFFER_PIXELS:
                continue
            
            if current_side != state["side"]:
                # Object appears to have crossed
                if state["pending_side"] == current_side:
                    state["pending_frames"] += 1
                else:
                    # Started crossing to a new side, reset counter
                    state["pending_side"] = current_side
                    state["pending_frames"] = 1
                
                if state["pending_frames"] >= DEBOUNCE_FRAMES:
                    # Confirmed crossing
                    self.crossing_count += 1
                    state["counted"] = True
                    state["side"] = current_side
                    print(f"[ALERT] ID{objectID} crossed! Total: {self.crossing_count}")
            else:
                # Back on original side, reset pending
                state["pending_frames"] = 0
                state["pending_side"] = None
        
        return self.crossing_count

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
        self._load_model(model_path)
        
        # Initialize tracker and counter
        self.tracker = CentroidTracker(
            maxDisappeared=MAX_DISAPPEARED,
            maxDistance=MAX_TRACK_DISTANCE,
        )
        self.counter = LineCrossingCounter(LINE_START, LINE_END, self.in_w, self.in_h)
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
    
    def _load_model(self, model_path):
        print(f"[INFO] Loading: {model_path}")
        self.npu_status = "unknown"
        
        if NPU_AVAILABLE:
            try:
                delegate = load_delegate('libvx_delegate.so')
                self.interpreter = Interpreter(model_path, experimental_delegates=[delegate])
                self.npu_status = "loaded"
                print("[INFO] NPU delegate loaded")
            except Exception as e:
                self.npu_status = "fallback"
                print(f"[WARN] NPU delegate failed, CPU fallback: {e}")
                self.interpreter = Interpreter(model_path, num_threads=4)
        else:
            self.npu_status = "unavailable"
            self.interpreter = Interpreter(model_path, num_threads=4)
            print("[INFO] NPU unavailable, using CPU")
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        input_shape = self.input_details[0]['shape']
        in_h, in_w = int(input_shape[1]), int(input_shape[2])
        in_dtype = self.input_details[0]['dtype']
        
        self.in_h, self.in_w = in_h, in_w
        print(f"[INFO] Model input dtype: {in_dtype}, output dtype: {self.output_details[0]['dtype']}")
        if str(in_dtype) not in ("<class 'numpy.uint8'>", "uint8", "int8", "int32"):
            print("[WARN] Model input does not look like a fully quantized int8/uint8 TFLite model")
        
        self.scale_params = {'scale': None, 'zero_point': 0}
        try:
            qp = self.output_details[0].get('quantization_parameters', {})
            if qp.get('scales'):
                self.scale_params['scale'] = float(qp['scales'][0])
            if qp.get('zero_points'):
                self.scale_params['zero_point'] = int(qp['zero_points'][0])
        except:
            pass
        
        self.preprocessor = CPUPreprocessor(in_w, in_h, in_dtype)
        self.postprocessor = OptimizedPostprocessor(
            conf_threshold=CONF_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            topk=500,
            class_ids=PERSON_CLASS_IDS
        )
        
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_input, _, _, _ = self.preprocessor.process(dummy)
        self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
        self.interpreter.invoke()
        print("[INFO] Model ready")

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
            print(
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
        print(f"\n=== [CAPS CHANGE] {element_name}:{pad_name} ===")
        if caps:
            print(caps.to_string())
        else:
            print("(no caps)")
    
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
            print("[INFO] End of stream")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] {err}: {debug}")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            if message.src == self.pipeline and new == Gst.State.PLAYING:
                print("\n=== [PIPELINE PLAYING] Printing current caps for key elements ===")
                # Print caps for key elements after pipeline is playing
                self._print_key_element_caps()
    
    def _print_key_element_caps(self):
        """Print caps for all key elements we're interested in."""
        # Helper to print caps for a pad
        def print_pad_caps(element, pad_name, element_label):
            pad = element.get_static_pad(pad_name)
            if pad:
                caps = pad.get_current_caps()
                print(f"\n=== {element_label} {pad_name.upper()} CAPS ===")
                if caps:
                    print(caps.to_string())
                else:
                    print("(no caps yet)")
        
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
        print(f"[DEBUG] _process_frame_for_detection called for frame {self.frame_count}, frame id: {id(frame)}")
        
        if self.frame_count % (SKIP_FRAMES + 1) != 1:
            print(f"[DEBUG] Skipping frame {self.frame_count} (not inference frame)")
            return

        frame_display = frame
        print(f"[DEBUG] Frame {self.frame_count}: frame_display id: {id(frame_display)}, min/max: {frame_display.min()}/{frame_display.max()}")

        if self.use_tiling:
            tile_x0, tile_y0, tile_x1, tile_y1 = self._get_tile_bounds()
            frame_tile = frame_display[tile_y0:tile_y1, tile_x0:tile_x1]
            tile_h, tile_w = frame_tile.shape[:2]
            print(f"[DEBUG] Frame {self.frame_count}: Using tiling, tile min/max: {frame_tile.min()}/{frame_tile.max()}")

            input_tensor, scale, pad_x, pad_y = self.preprocessor.process(frame_tile)
            print(f"[DEBUG] Frame {self.frame_count}: Preprocessing done, setting tensor shape: {input_tensor.shape}")
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            print(f"[DEBUG] Frame {self.frame_count}: Inference done, output shape: {output_data.shape}")

            person_boxes, person_scores, person_classes = self.postprocessor.process(
                output_data, (tile_h, tile_w), (self.in_h, self.in_w),
                self.scale_params, scale, pad_x, pad_y
            )
            print(f"[DEBUG] Frame {self.frame_count}: Found {len(person_boxes)} person boxes")
            person_boxes_display = self._map_boxes_tile_to_display(person_boxes, tile_x0, tile_y0)
            person_boxes = self._map_boxes_tile_to_full_ml(person_boxes, tile_x0, tile_y0)
        else:
            frame_ml = frame_display
            print(f"[DEBUG] Frame {self.frame_count}: No tiling, frame_ml id: {id(frame_ml)}, min/max: {frame_ml.min()}/{frame_ml.max()}")

            input_tensor, scale, pad_x, pad_y = self.preprocessor.process(frame_ml)
            print(f"[DEBUG] Frame {self.frame_count}: Preprocessing done, setting tensor shape: {input_tensor.shape}")
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            print(f"[DEBUG] Frame {self.frame_count}: Inference done, output shape: {output_data.shape}")

            person_boxes, person_scores, person_classes = self.postprocessor.process(
                output_data, (self.in_h, self.in_w), (self.in_h, self.in_w),
                self.scale_params, scale, pad_x, pad_y
            )
            print(f"[DEBUG] Frame {self.frame_count}: Found {len(person_boxes)} person boxes")
            person_boxes_display = self._map_boxes_ml_to_display(person_boxes)

        tracked_objects, _ = self.tracker.update(person_boxes)
        crossing_count = self.counter.update(tracked_objects, self.tracker, person_boxes)
        print(f"[DEBUG] Frame {self.frame_count}: Tracked {len(tracked_objects)} objects, crossings: {crossing_count}")

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
        
        # 1. Print negotiated caps immediately after pulling sample
        print(f"\n=== [NEGOTIATED CAPS] Frame {self.frame_count} ===")
        if caps:
            for i in range(caps.get_size()):
                print(caps.get_structure(i).to_string())
        
        # 2. Print frame metadata
        s = caps.get_structure(0)
        h, w = s.get_value("height"), s.get_value("width")
        fmt = s.get_value("format") if s.has_field("format") else "unknown"
        buffer_size = buf.get_size()
        # Calculate expected bytes based on format
        expected_bytes = 0
        if fmt == "RGB" or fmt == "BGR":
            expected_bytes = h * w * 3
        elif fmt == "RGBx" or fmt == "BGRx" or fmt == "RGBA" or fmt == "BGRA":
            expected_bytes = h * w * 4
        elif fmt == "NV12":
            expected_bytes = h * w * 3 // 2  # Y plane (h*w) + UV plane (h*w/2)
        
        print(f"\n=== [FRAME METADATA] Frame {self.frame_count} ===")
        print(f"Width: {w}")
        print(f"Height: {h}")
        print(f"Format: {fmt}")
        print(f"Buffer size: {buffer_size}")
        print(f"Expected bytes: {expected_bytes}")
        
        ret, mem = buf.map(Gst.MapFlags.READ)
        if not ret:
            return Gst.FlowReturn.OK
            
        # 3. Print NumPy conversion details
        print(f"\n=== [GSTBUFFER -> NUMPY CONVERSION] Frame {self.frame_count} ===")
        # Try to handle different formats
        frame = None
        if fmt == "RGB" or fmt == "BGR":
            frame = np.ndarray((h, w, 3), dtype=np.uint8, buffer=mem.data).copy()
        elif fmt == "RGBx" or fmt == "BGRx":
            # Drop the x channel, take first 3
            frame_rgbx = np.ndarray((h, w, 4), dtype=np.uint8, buffer=mem.data).copy()
            frame = frame_rgbx[:, :, :3]
        elif fmt == "RGBA" or fmt == "BGRA":
            frame_rgba = np.ndarray((h, w, 4), dtype=np.uint8, buffer=mem.data).copy()
            frame = frame_rgba[:, :, :3]
        else:
            # Fallback to 3 channels as before
            frame = np.ndarray((h, w, 3), dtype=np.uint8, buffer=mem.data).copy()
        
        print(f"numpy shape: {frame.shape}")
        print(f"numpy dtype: {frame.dtype}")
        buf.unmap(mem)
        
        # Save first frame at multiple stages
        if self.frame_count == 1:
            print(f"\n=== [SAVING FIRST FRAME STAGES] ===")
            
            # 01_raw_buffer.png (save as-is first, then convert appropriately)
            # For saving, convert to BGR for OpenCV
            if fmt == "RGB":
                raw_for_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif fmt == "BGR":
                raw_for_save = frame
            elif fmt == "RGBx" or fmt == "BGRx" or fmt == "RGBA" or fmt == "BGRA":
                # BGRx to BGR
                raw_for_save = frame  # since we already took first 3, if fmt was BGRx it's BGR; if RGBx it's RGB, so convert
                if fmt.startswith("RGB"):
                    raw_for_save = cv2.cvtColor(raw_for_save, cv2.COLOR_RGB2BGR)
            else:
                raw_for_save = frame
            
            cv2.imwrite("/tmp/01_raw_buffer.png", raw_for_save)
            print("wrote /tmp/01_raw_buffer.png")
            
            # 02_after_conversion.png (same as raw for now, but saved)
            cv2.imwrite("/tmp/02_after_conversion.png", raw_for_save)
            print("wrote /tmp/02_after_conversion.png")
            
            # 03_before_inference.png (before passing to _process_frame_for_detection)
            cv2.imwrite("/tmp/03_before_inference.png", raw_for_save)
            print("wrote /tmp/03_before_inference.png")
        
        # Also keep the old debug prints for first 5 and every 30th
        is_debug_frame = self.frame_count <= 5 or self.frame_count % 30 == 1
        if is_debug_frame:
            print(f"\n=== [NUMPY FRAME PROPERTIES] Frame {self.frame_count} ===")
            print(f"min/max: {frame.min()}, {frame.max()}")
            print(f"mean: {frame.mean():.2f}")
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                print(f"Channel 0: {frame[:,:,0].mean():.2f}")
                print(f"Channel 1: {frame[:,:,1].mean():.2f}")
                print(f"Channel 2: {frame[:,:,2].mean():.2f}")
            
            # Print first few bytes
            print(f"\n=== [FIRST 32 BYTES] Frame {self.frame_count} ===")
            print(frame.flatten()[:32])
        
        print(f"[DEBUG] Passing frame {self.frame_count} to _process_frame_for_detection, frame id: {id(frame)}")
        self._process_frame_for_detection(frame)

        if self.frame_count % 30 == 1:
            with self.lock:
                det = len(self.person_boxes)
                trk = len(self.tracked_objects)
                c_count = self.crossing_count
            print(f"[{self.frame_count}] det:{det} trk:{trk} cross:{c_count} fps:{self.fps:.1f}")

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
                f"vpuenc_h264 bitrate=8000 ! h264parse ! mp4mux ! filesink location={self.output_file} "
                f"out. ! queue max-size-buffers=2 leaky=downstream ! {sink_str} "
                f"out. ! queue max-size-buffers=2 leaky=downstream ! "
                f"imxvideoconvert_g2d ! video/x-raw,format=RGB16 ! "
                f"vpuenc_h264 bitrate=5000 ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5004 "
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
                f"vpuenc_h264 bitrate=5000 ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5004 "
                f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! video/x-raw,format=RGB ! appsink name=ml_sink emit-signals=true drop=true max-buffers=2 sync=false"
            )
            
        print(f"[INFO] Pipeline: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        overlay = self.pipeline.get_by_name("overlay")
        if overlay and cairo is not None:
            overlay.connect("draw", self._draw_overlay)
        elif overlay:
            print("[WARN] Python cairo not available; Cairo overlay callback disabled")
            
        ml_sink = self.pipeline.get_by_name("ml_sink")
        if ml_sink:
            ml_sink.connect("new-sample", self._on_new_sample)
            
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def _run_cv2_fallback(self):
        print("[INFO] GStreamer unavailable. Running OpenCV/TFLite fallback for local debugging.")
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
                    print("[INFO] End of stream")
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
                        print("[INFO] Stopped by user")
                        break

                if self.frame_count % 30 == 1:
                    with self.lock:
                        det = len(self.person_boxes)
                        trk = len(self.tracked_objects)
                        c_count = self.crossing_count
                    print(f"[{self.frame_count}] det:{det} trk:{trk} cross:{c_count} fps:{self.fps:.1f}")
        finally:
            self.system_monitor.stop()
            cap.release()
            if writer is not None:
                writer.release()
            if self.preview_enabled:
                cv2.destroyAllWindows()
            self.metrics_recorder.save(self.metrics_csv_path, self.metrics_json_path)
            print(f"\n[INFO] Done! Total crossings: {self.crossing_count}")

    def _start_rtsp_server(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_shared(True)
        # Bridge the UDP stream from the main pipeline into the RTSP server
        factory.set_launch(
            '( udpsrc port=5004 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            'rtph264pay name=pay0 pt=96 )'
        )
        
        mounts = self.server.get_mount_points()
        mounts.add_factory("/video", factory)
        self.server.attach(None)
        print(f"[RTSP] Server running at rtsp://0.0.0.0:8554/video")

    def run(self):
        print(f"\n[INFO] Input: {'Camera' if self.is_camera else 'Video'} {self.video_src}")
        print(f"[INFO] Output: {self.output_file if self.output_file else 'Display only'}")
        print(f"[INFO] Line: {LINE_START} -> {LINE_END}")
        if self.use_tiling:
            tile_x0, _, tile_x1, _ = self._get_tile_bounds()
            print(f"[INFO] Tiling: right-half ROI x={tile_x0}-{tile_x1} (padding={self.tile_padding})")
        else:
            print("[INFO] Tiling: disabled")

        if not GST_AVAILABLE:
            self._run_cv2_fallback()
            return
        
        self.loop = GLib.MainLoop()
        self._build_pipeline()
        self._start_rtsp_server()
        
        print("[INFO] Starting pipeline...")
        self.system_monitor.start()
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n[INFO] Stopped")
        finally:
            self.system_monitor.stop()
            self.pipeline.set_state(Gst.State.NULL)
            self.metrics_recorder.save(self.metrics_csv_path, self.metrics_json_path)
            print(f"\n[INFO] Done! Total crossings: {self.crossing_count}")

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
        print("[INFO] GStreamer bindings not available on this machine; using OpenCV fallback.")
    
    LineCrossingDetector(
        args.model, args.video, args.output,
        use_tiling=args.use_tiling,
        tile_padding=args.tile_padding,
        metrics_csv_path=args.metrics_csv,
        metrics_json_path=args.metrics_json,
        preview_enabled=args.preview,
        debug_mapping=args.debug_mapping,
    ).run()
