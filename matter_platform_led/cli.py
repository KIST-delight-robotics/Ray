"""Command-line entry point — run on/off signals directly to verify the path.

Usage (see README for the full walk-through)::

    python -m matter_platform_led --backend mock commission
    python -m matter_platform_led --backend mock on
    python -m matter_platform_led --backend mock status
    python -m matter_platform_led off            # uses backend from config.toml

Exit code is 0 on success, 1 on any MatterError (with a readable message).
"""

from __future__ import annotations

import argparse
import logging
import sys

from matter_platform_led.controller import MatterLedController
from matter_platform_led.exceptions import MatterError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m matter_platform_led",
        description="Send Matter On/Off signals to a WiFi light (or a mock for hardware-free verification).",
    )
    parser.add_argument("--config", default=None, help="path to config.toml (default: packaged config)")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["mock", "chip_tool"],
        help="override the backend from config.toml",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "action",
        choices=["commission", "on", "off", "toggle", "status", "share"],
        help="what to do ('share' opens a window for a phone to co-pair)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the action, print the result. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        with MatterLedController.from_config(args.config, backend_override=args.backend) as ctrl:
            if args.action == "commission":
                ctrl.commission()
                print("commissioned ✔")
            elif args.action == "on":
                ctrl.on()
                print("on ✔")
            elif args.action == "off":
                ctrl.off()
                print("off ✔")
            elif args.action == "toggle":
                ctrl.toggle()
                print("toggled ✔")
            elif args.action == "status":
                st = ctrl.status()
                state = "ON" if st.on else "OFF"
                suffix = "" if st.reachable else " (unreachable — best-effort)"
                print(f"status: {state}{suffix}")
            elif args.action == "share":
                code = ctrl.share()
                print("commissioning window open — add to your phone Home app with:")
                print(f"  {code}")
    except MatterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
