"""Evaluate a baseline checkpoint with COCO mask metrics on frozen validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wardrobe_seg.data import discover_fashionpedia
from wardrobe_seg.evaluate import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/outputs/evaluation"))
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-side", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    paths = discover_fashionpedia(args.input_root)
    report = evaluate_checkpoint(
        paths=paths,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_images=args.max_images,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        max_side=args.max_side,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
