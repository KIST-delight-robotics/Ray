"""PMIC power logger for Raspberry Pi 5.

Periodically samples `vcgencmd pmic_read_adc`, writes every rail to a CSV, and
tracks the minimum voltage / maximum current seen per rail during the run. On
exit (Ctrl-C) it prints a summary table — useful for checking whether the 5V
supply sags or how much current the robot draws under load.

Usage:
    # Live logging (Ctrl-C to stop and print summary)
    uv run python scripts/hardware/pmic_log.py [--interval SEC] [--out FILE]

    # Re-summarize a CSV produced by a previous run
    uv run python scripts/hardware/pmic_log.py --summary pmic_log_*.csv

Note: sampling is limited by how fast `vcgencmd pmic_read_adc` returns
(~0.1s on a Pi 5), so transient sags shorter than the interval may be missed.
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

# Matches e.g. "VDD_CORE_A current(7)=2.6303214A" or "EXT5V_V volt(24)=5.09V"
_LINE_RE = re.compile(r"^(?P<name>\S+)\s+(?:current|volt)\(\d+\)=(?P<val>[-+]?\d*\.?\d+)")

_COMPUTED_COLS = ("total_power_w", "est_input_a")


def read_pmic() -> tuple[dict[str, float], dict[str, float]]:
    """Run vcgencmd and return (currents, volts) keyed by rail name.

    Raises RuntimeError if vcgencmd is unavailable or returns an error.
    """
    try:
        proc = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as e:
        raise RuntimeError("vcgencmd not found — are you on a Raspberry Pi?") from e
    if proc.returncode != 0:
        raise RuntimeError(f"vcgencmd failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return parse_pmic_output(proc.stdout)


def parse_pmic_output(text: str) -> tuple[dict[str, float], dict[str, float]]:
    """Parse pmic_read_adc stdout into (currents, volts) dicts.

    Rail name is the token minus its trailing _A / _V suffix.
    """
    currents: dict[str, float] = {}
    volts: dict[str, float] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        val = float(m.group("val"))
        if name.endswith("_A"):
            currents[name[:-2]] = val
        elif name.endswith("_V"):
            volts[name[:-2]] = val
    return currents, volts


def compute_derived(currents: dict[str, float], volts: dict[str, float]) -> tuple[float, float]:
    """Return (total_power_w, est_input_current_a).

    Total power = sum of per-rail current*voltage for rails present in both.
    Estimated input current = total_power / EXT5V voltage.
    """
    total_power = sum(currents[r] * volts[r] for r in currents.keys() & volts.keys())
    ext5v = volts.get("EXT5V")
    est_input = total_power / ext5v if ext5v and ext5v > 0 else 0.0
    return total_power, est_input


class Tracker:
    """Tracks running min voltage / max current per rail plus derived extremes."""

    def __init__(self) -> None:
        self.min_volt: dict[str, tuple[float, str]] = {}
        self.max_curr: dict[str, tuple[float, str]] = {}
        self.max_power: tuple[float, str] = (0.0, "")
        self.max_input: tuple[float, str] = (0.0, "")
        self.samples = 0

    def update(
        self,
        ts: str,
        currents: dict[str, float],
        volts: dict[str, float],
        total_power: float,
        est_input: float,
    ) -> None:
        self.samples += 1
        for rail, v in volts.items():
            cur = self.min_volt.get(rail)
            if cur is None or v < cur[0]:
                self.min_volt[rail] = (v, ts)
        for rail, a in currents.items():
            cur = self.max_curr.get(rail)
            if cur is None or a > cur[0]:
                self.max_curr[rail] = (a, ts)
        if total_power > self.max_power[0]:
            self.max_power = (total_power, ts)
        if est_input > self.max_input[0]:
            self.max_input = (est_input, ts)

    def print_summary(self) -> None:
        if self.samples == 0:
            print("\nNo samples collected.")
            return
        print(f"\n\n=== Summary over {self.samples} samples ===\n")

        print("Minimum voltage per rail:")
        for rail in sorted(self.min_volt):
            v, ts = self.min_volt[rail]
            print(f"  {rail:<12s} min {v:8.4f} V   @ {ts}")

        print("\nMaximum current per rail:")
        for rail in sorted(self.max_curr):
            a, ts = self.max_curr[rail]
            print(f"  {rail:<12s} max {a:8.4f} A   @ {ts}")

        print("\nKey metrics:")
        if "EXT5V" in self.min_volt:
            v, ts = self.min_volt["EXT5V"]
            print(f"  EXT5V input voltage   min {v:8.4f} V   @ {ts}")
        print(f"  Total power           max {self.max_power[0]:8.4f} W   @ {self.max_power[1]}")
        print(f"  Est. input current    max {self.max_input[0]:8.4f} A   @ {self.max_input[1]}")
        print()


def run_logger(interval: float, out_path: str) -> None:
    """Sample in a loop, append to CSV, and print a summary on interrupt."""
    print("Reading initial PMIC sample...")
    currents, volts = read_pmic()
    curr_rails = sorted(currents)
    volt_rails = sorted(volts)
    header = ["timestamp", *_COMPUTED_COLS] + [f"{r}_A" for r in curr_rails] + [f"{r}_V" for r in volt_rails]

    tracker = Tracker()
    stop = {"flag": False}

    def _handle_sigint(_signum: int, _frame: object) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"Logging to {out_path}  (interval={interval}s)")
    print("Press Ctrl-C to stop and print summary.\n")

    start = time.monotonic()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        while not stop["flag"]:
            loop_start = time.monotonic()
            ts = datetime.now().isoformat(timespec="milliseconds")
            try:
                currents, volts = read_pmic()
            except RuntimeError as e:
                print(f"\nRead error: {e}")
                break
            total_power, est_input = compute_derived(currents, volts)
            tracker.update(ts, currents, volts, total_power, est_input)

            row = (
                [ts, f"{total_power:.4f}", f"{est_input:.4f}"]
                + [f"{currents.get(r, ''):.4f}" if r in currents else "" for r in curr_rails]
                + [f"{volts.get(r, ''):.4f}" if r in volts else "" for r in volt_rails]
            )
            writer.writerow(row)
            f.flush()

            ext5v = volts.get("EXT5V", 0.0)
            ext5v_min = tracker.min_volt.get("EXT5V", (ext5v, ""))[0]
            elapsed = time.monotonic() - start
            print(
                f"\r  [{elapsed:6.0f}s] n={tracker.samples:5d}  "
                f"EXT5V={ext5v:6.3f}V (min {ext5v_min:6.3f})  "
                f"P={total_power:6.3f}W (max {tracker.max_power[0]:6.3f})  "
                f"Iin~{est_input:5.3f}A (max {tracker.max_input[0]:5.3f})",
                end="",
                flush=True,
            )

            sleep_left = interval - (time.monotonic() - loop_start)
            # Sleep in short slices so Ctrl-C is responsive.
            while sleep_left > 0 and not stop["flag"]:
                time.sleep(min(0.1, sleep_left))
                sleep_left -= 0.1

    tracker.print_summary()
    print(f"CSV saved to {out_path}")


def summarize_csv(path: str) -> None:
    """Compute min-voltage / max-current summary from an existing CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Empty CSV.")
            return
        a_cols = [c for c in reader.fieldnames if c.endswith("_A")]
        v_cols = [c for c in reader.fieldnames if c.endswith("_V")]

        tracker = Tracker()
        for row in reader:
            ts = row.get("timestamp", "")
            currents = {c[:-2]: float(row[c]) for c in a_cols if row.get(c)}
            volts = {c[:-2]: float(row[c]) for c in v_cols if row.get(c)}
            power = float(row["total_power_w"]) if row.get("total_power_w") else 0.0
            est_input = float(row["est_input_a"]) if row.get("est_input_a") else 0.0
            tracker.update(ts, currents, volts, power, est_input)

    print(f"Summarizing {path}")
    tracker.print_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Raspberry Pi 5 PMIC voltage/current")
    parser.add_argument("--interval", type=float, default=1.0, help="Sample interval in seconds")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: pmic_log_<timestamp>.csv)",
    )
    parser.add_argument("--summary", metavar="CSV", help="Summarize an existing CSV and exit")
    args = parser.parse_args()

    if args.summary:
        summarize_csv(args.summary)
        return

    out_path = args.out or f"pmic_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
    try:
        run_logger(args.interval, out_path)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
