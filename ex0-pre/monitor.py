"""
Standalone hardware monitor — prints live system metrics to the console.

    python monitor.py                 # sample every 2 s (CPU/RAM/GPU-mem)
    sudo python monitor.py            # + GPU power/thermal (powermetrics needs root)
    python monitor.py --interval 0.5  # sample every 0.5 s
    python monitor.py --track         # also log to W&B/MLflow as its own run
    python monitor.py --track --run-name monitor_during_3digit_runs

Same metrics that train.py logs when run with --monitor (system/* in W&B/MLflow).
Stop with Ctrl-C.
"""

import argparse
import time
from datetime import datetime

from src.monitor import SystemMonitor
from src.tracker import ExperimentTracker


def main() -> None:
    p = argparse.ArgumentParser(description="Print live hardware metrics")
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between samples (default 2)")
    p.add_argument("--track", action="store_true",
                   help="log samples to W&B/MLflow as a standalone run")
    p.add_argument("--run-name", default=None,
                   help="tracker run name (default: monitor_<timestamp>)")
    args = p.parse_args()

    tracker = None
    if args.track:
        name = args.run_name or f"monitor_{datetime.now().isoformat(timespec='seconds')}"
        tracker = ExperimentTracker(
            name, {"kind": "monitor", "interval_s": args.interval}, True, True
        )

    mon = SystemMonitor(interval_ms=int(args.interval * 1000))
    mon.start()
    step = 0

    try:
        while True:
            time.sleep(args.interval)
            metrics = mon.latest()
            if not metrics:
                continue
            line = "  ".join(
                f"{k.removeprefix('system/')}={v}" for k, v in sorted(metrics.items())
            )
            print(time.strftime("%H:%M:%S"), line, flush=True)
            if tracker:
                tracker.log(metrics, step=step)
            step += 1
    except KeyboardInterrupt:
        mon.stop()
        if tracker:
            tracker.finish()
        print("\nstopped")


if __name__ == "__main__":
    main()
