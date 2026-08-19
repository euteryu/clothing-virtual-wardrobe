"""Run the frozen model on a private personal-photo applicability set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wardrobe_seg.personal import run_personal_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run_personal_inference(
        photo_dir=args.photo_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        expected_images=args.expected_images,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
