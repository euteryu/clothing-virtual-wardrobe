"""Select a main-garment confidence threshold and render validation failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wardrobe_seg.data import discover_fashionpedia
from wardrobe_seg.review import analyze_operating_point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/outputs/review"))
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()
    report = analyze_operating_point(
        paths=discover_fashionpedia(args.input_root),
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_images=args.max_images,
        workers=args.workers,
        max_side=args.max_side,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

