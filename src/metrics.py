import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import csv
import json
import logging

logger = logging.getLogger(__name__)

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


