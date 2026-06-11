"""
Background hardware monitor for Apple Silicon (and any platform with psutil).

Data sources
  psutil       → CPU %, RAM used/free/total          (pip install psutil)
  torch.mps    → GPU memory allocated                (built-in with PyTorch)
  powermetrics → GPU active %, power draw, thermal   (macOS; needs passwordless sudo)

Passwordless sudo setup (one-time):
    echo "$(whoami) ALL = NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics

If powermetrics is unavailable, the monitor degrades to psutil-only and says so.
"""

import os
import re
import subprocess
import threading

try:
    import torch
except ImportError:
    torch = None  # GPU memory metric is skipped; everything else still works


class SystemMonitor:
    """
    Start a background thread that polls hardware metrics every `interval_ms` ms.
    Call latest() from the training loop to get the most-recent snapshot as a dict.

    All keys are prefixed with "sys/" for clean namespacing in W&B / MLflow.
    """

    def __init__(self, interval_ms: int = 2_000):
        self.interval_ms = interval_ms
        self._metrics: dict = {}
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        threading.Thread(target=self._psutil_loop,       daemon=True).start()
        threading.Thread(target=self._powermetrics_loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> dict:
        with self._lock:
            return dict(self._metrics)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set(self, metrics: dict) -> None:
        with self._lock:
            self._metrics.update(metrics)

    def _psutil_loop(self) -> None:
        try:
            import psutil
        except ImportError:
            print("monitor ▸ psutil not installed — CPU/RAM metrics off  (pip install psutil)")
            return

        print("monitor ▸ psutil active (CPU/RAM/GPU-mem)")
        while not self._stop.wait(self.interval_ms / 1000):
            vm     = psutil.virtual_memory()
            gpu_mb = 0.0
            try:
                gpu_mb = torch.mps.current_allocated_memory() / 1e6
            except Exception:
                pass
            self._set({
                "sys/cpu_pct":     psutil.cpu_percent(),
                "sys/ram_used_gb": round(vm.used      / 1e9, 2),
                "sys/ram_free_gb": round(vm.available / 1e9, 2),
                "sys/ram_pct":     vm.percent,
                "sys/gpu_mem_mb":  round(gpu_mb, 1),
            })

    def _powermetrics_loop(self) -> None:
        # Already root (e.g. `sudo python monitor.py`) → run powermetrics directly.
        # Otherwise `sudo -n` exits immediately if a password would be required —
        # no-op on machines where the sudoers entry hasn't been added.
        prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        try:
            proc = subprocess.Popen(
                [*prefix, "powermetrics",
                 "--samplers", "gpu_power,cpu_power,thermal",
                 "-i", str(self.interval_ms), "-n", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception as e:
            print(f"monitor ▸ powermetrics unavailable ({e}) — power/thermal metrics off")
            return

        print("monitor ▸ powermetrics starting (GPU/power/thermal)")
        _PATTERNS = [
            (r"GPU HW active residency:\s+([\d.]+)%",                "sys/gpu_active_pct"),
            (r"GPU HW active frequency:\s+([\d.]+) MHz",             "sys/gpu_freq_mhz"),
            (r"GPU idle residency:\s+([\d.]+)%",                     "sys/gpu_idle_pct"),
            (r"GPU Power:\s+([\d.]+) mW",                            "sys/gpu_power_mw"),
            (r"CPU Power:\s+([\d.]+) mW",                            "sys/cpu_power_mw"),
            (r"ANE Power:\s+([\d.]+) mW",                            "sys/ane_power_mw"),
            (r"Combined Power \(CPU \+ GPU \+ ANE\):\s+([\d.]+) mW", "sys/total_power_mw"),
        ]
        _THERMAL = {"Nominal": 0, "Light": 1, "Moderate": 2, "Heavy": 3, "Critical": 4}

        for line in proc.stdout:
            if self._stop.is_set():
                proc.terminate()
                break
            line = line.strip()
            for pattern, key in _PATTERNS:
                m = re.search(pattern, line)
                if m:
                    self._set({key: float(m.group(1))})
            m = re.search(r"Current pressure level:\s+(\w+)", line)
            if m:
                self._set({"sys/thermal": _THERMAL.get(m.group(1), -1)})

        # sudo -n fails instantly (closing stdout) when passwordless sudo isn't set up
        if proc.wait() != 0 and not self._stop.is_set():
            print("monitor ▸ powermetrics needs root — power/thermal metrics off. "
                  "Run with sudo (e.g. `sudo python monitor.py`) or set up "
                  "passwordless sudo (see src/monitor.py docstring)")
