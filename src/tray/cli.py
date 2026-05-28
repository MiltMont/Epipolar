"""Command-line entrypoint for the tray package.

Usage:
    python -m tray run --config configs/<name>.yaml
"""

from __future__ import annotations

import argparse
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
        help="Directory where per-config outputs are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        # Wired in a later phase — see plan Phase 2.
        raise NotImplementedError(
            "tray.experiments.runner is not implemented yet (see plan Phase 2)."
        )

    parser.error(f"Unknown command: {args.command}")
    return 2
