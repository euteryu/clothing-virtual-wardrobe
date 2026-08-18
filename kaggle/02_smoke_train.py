"""Run a bounded end-to-end GPU training smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wardrobe_seg.data import discover_fashionpedia
from wardrobe_seg.train import train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/outputs/smoke"))
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.max_images < 2 or args.epochs < 1 or args.batch_size < 1:
        parser.error("max-images >= 2, epochs >= 1, and batch-size >= 1 are required")
    return args


def main() -> None:
    args = parse_args()
    paths = discover_fashionpedia(args.input_root)
    report = train_baseline(
        paths, args.output_dir, args.max_images, args.epochs, args.batch_size, args.workers
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

