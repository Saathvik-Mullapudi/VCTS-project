import time
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import csv
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

from configs.settings import PROFILING_WARMUP_FRAMES

class ThroughputProfiler:
    def __init__(self, stage_names):
        self.stage_names = list(stage_names)
        self.received = {name: 0 for name in self.stage_names}
        self.processed = {name: 0 for name in self.stage_names}
        self.skipped = {name: 0 for name in self.stage_names}
        self.dropped = {name: 0 for name in self.stage_names}
        self.total_ns = {name: 0 for name in self.stage_names}
        self._samples = {name: [] for name in self.stage_names}  # per-frame latency samples
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.profiling_enabled = False

    def enable(self):
        self.profiling_enabled = True
        self.start_time = time.time()

    def record_received(self, name):
        if not self.profiling_enabled: return
        with self._lock:
            self.received[name] += 1

    def record_processed(self, name, elapsed_ns=0):
        if not self.profiling_enabled: return
        with self._lock:
            self.processed[name] += 1
            self.total_ns[name] += elapsed_ns

    def record_skipped(self, name):
        if not self.profiling_enabled: return
        with self._lock:
            self.skipped[name] += 1

    def record_dropped(self, name):
        if not self.profiling_enabled: return
        with self._lock:
            self.dropped[name] += 1

    @contextmanager
    def stage(self, name):
        if not self.profiling_enabled:
            yield
            return
            
        with self._lock:
            self.received[name] += 1
            
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            with self._lock:
                self.total_ns[name] += elapsed
                self.processed[name] += 1
                self._samples[name].append(elapsed / 1e6)  # store in ms

    def format_profiling_summary(self, input_fps):
        if not self.profiling_enabled:
            return "No profiling data collected."

        total_wall_time = max(0.001, time.time() - self.start_time)
        
        lines = [
            "=" * 80,
            "Throughput Profiling Report",
            "=" * 80,
            f"Total Wall-Clock Run Time: {total_wall_time:.1f} seconds",
            f"Input Video FPS: {input_fps:.1f}",
            "Runtime FPS: processed inference frames only (latest-frame buffer on GStreamer path)",
            "=" * 80,
            "",
            f"{'Stage':<18} | {'Received':>8} | {'Processed':>9} | {'Skipped':>7} | {'Avg Time/Frame':>14} | {'Throughput (FPS)':>16}",
            "-" * 80
        ]
        
        name_map = {
            "demux": "0. Demux",
            "decoder": "1. Decoder",
            "appsink": "2. Appsink",
            "color_convert": "3. Color Convert",
            "preprocess": "4. Preprocess",
            "inference": "5. Inference",
            "postprocess": "6. Postprocess",
            "tracking": "7. Tracking",
            "line_crossing": "8. Line Crossing",
            "overlay": "9. Overlay",
            "encoder": "11. Encoder",
            "output_write": "12. Output Writer"
        }
        
        with self._lock:
            for name in self.stage_names:
                display_name = name_map.get(name, name.capitalize())
                recv = self.received.get(name, 0)
                proc = self.processed.get(name, 0)
                skip = self.skipped.get(name, 0)
                
                # Check for drops
                if recv > proc + skip:
                    self.dropped[name] = recv - proc - skip
                drop = self.dropped.get(name, 0)
                
                # Time
                t_ns = self.total_ns.get(name, 0)
                avg_ms = (t_ns / proc / 1e6) if proc > 0 else 0
                avg_str = f"{avg_ms:.2f} ms" if t_ns > 0 else "N/A (HW)"
                
                # Throughput
                tput = proc / total_wall_time
                tput_str = f"{tput:.1f} FPS"
                
                line = f"{display_name:<18} | {recv:>8} | {proc:>9} | {skip:>7} | {avg_str:>14} | {tput_str:>16}"
                if drop > 0:
                    line += f"  <-- ({drop} dropped!)"
                lines.append(line)
        
        lines.extend(["", "=" * 80])

        # ── Latency Report ────────────────────────────────────────────────────
        lines.extend([
            "",
            "=" * 80,
            "Per-Stage Latency Report",
            "=" * 80,
            f"{'Stage':<18} | {'Avg (ms)':>8} | {'Min (ms)':>8} | {'Max (ms)':>8} | {'Median':>8} | {'p95 (ms)':>8}",
            "-" * 80
        ])

        with self._lock:
            for name in self.stage_names:
                display_name = name_map.get(name, name.capitalize())
                samples = self._samples.get(name, [])
                if len(samples) < 2:
                    lines.append(f"{display_name:<18} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")
                    continue
                arr = np.array(samples)
                lines.append(
                    f"{display_name:<18} | {np.mean(arr):>8.2f} | {np.min(arr):>8.2f} | "
                    f"{np.max(arr):>8.2f} | {np.median(arr):>8.2f} | {np.percentile(arr, 95):>8.2f}"
                )

        lines.extend(["", "=" * 80])
        return "\n".join(lines)

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
            except Exception as e:
                logger.debug(f"Could not parse start timestamp for metrics summary: {e}")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _base_row(self, record_type, record):
        row = {
            "record_type": record_type,
            **self.config_metadata,
        }
        row.update(record)
        return row


