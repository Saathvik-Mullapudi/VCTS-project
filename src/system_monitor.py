import time
import threading
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
try:
    import psutil
except ImportError:
    psutil = None

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
        self._has_logged_info_sample = False
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
                formatted_sample = self.format_sample(sample)
                if not self._has_logged_info_sample:
                    logger.info(f"System Metrics: {formatted_sample}")
                    self._has_logged_info_sample = True
                else:
                    logger.debug(f"System Metrics: {formatted_sample}")
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

