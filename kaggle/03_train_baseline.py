"""Train the first measured Fashionpedia baseline with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wardrobe_seg.data import discover_fashionpedia
from wardrobe_seg.train import train_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/outputs/baseline"))
    parser.add_argument("--max-images", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.max_images < 64 or args.epochs < 1 or args.batch_size < 1:
        parser.error("max-images >= 64, epochs >= 1, and batch-size >= 1 are required")
    return args


def main() -> None:
    args = parse_args()
    paths = discover_fashionpedia(args.input_root)
    report = train_epochs(
        paths=paths,
        output_dir=args.output_dir,
        max_images=args.max_images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

