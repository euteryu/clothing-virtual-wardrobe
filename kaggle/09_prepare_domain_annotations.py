"""Prepare a private image set for efficient human mask correction in CVAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wardrobe_seg.annotation import prepare_cvat_preannotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--proposal-confidence", type=float, default=0.3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = prepare_cvat_preannotations(
        photo_dir=args.photo_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        expected_images=args.expected_images,
        max_side=args.max_side,
        confidence_threshold=args.proposal_confidence,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
