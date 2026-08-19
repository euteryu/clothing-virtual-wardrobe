"""Run the single predefined improvement: continue epoch 03 through epoch 05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wardrobe_seg.data import discover_fashionpedia
from wardrobe_seg.train import train_epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/outputs/improvement")
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != 3 or int(state.get("max_images", -1)) != 4000:
        raise ValueError("Improvement must resume the fixed 4,000-image epoch-03 baseline")
    report = train_epochs(
        paths=discover_fashionpedia(args.input_root),
        output_dir=args.output_dir,
        max_images=4000,
        epochs=5,
        batch_size=2,
        workers=args.workers,
        seed=2026,
        resume=args.checkpoint,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
