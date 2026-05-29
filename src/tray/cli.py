"""Command-line entrypoint for the tray package.

Usage:
    python -m tray run --config configs/<name>.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tray",
        description="Camera trajectory estimation on TUM RGB-D (epipolar + ICP).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a single experiment configuration.")
    run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML experiment config under configs/.",
    )
    run.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where per-config outputs are written (default: results/).",
    )
    run.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing TUM sequence folders (default: data/).",
    )
    run.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Truncate each sequence to N frames (useful for quick smoke tests).",
    )
    run.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "run":
        from tray.experiments.runner import run_experiment
        import json

        metrics = run_experiment(
            args.config,
            results_dir=args.results_dir,
            data_dir=args.data_dir,
            max_frames=args.max_frames,
        )
        print(json.dumps(metrics, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
