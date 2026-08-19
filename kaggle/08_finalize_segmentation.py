"""Compare final checkpoints and freeze the application operating policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wardrobe_seg.data import discover_fashionpedia
from wardrobe_seg.review import analyze_operating_point, select_application_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/outputs/finalization")
    )
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()

    reports = []
    paths = discover_fashionpedia(args.input_root)
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(state["epoch"])
        report = analyze_operating_point(
            paths=paths,
            checkpoint_path=checkpoint,
            output_dir=args.output_dir / f"epoch_{epoch:02d}",
            max_images=args.max_images,
            workers=args.workers,
            max_side=args.max_side,
        )
        reports.append(report)

    final_report = select_application_policy(reports)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "final_segmentation_policy.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
