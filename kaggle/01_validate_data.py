"""Discover and validate the attached Fashionpedia competition data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from wardrobe_seg.data import FashionpediaDataset, discover_fashionpedia


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_fashionpedia(args.input_root)
    dataset = FashionpediaDataset(paths, "all")
    counts: Counter[int] = Counter()
    pixels = 0
    for index in range(min(args.samples, len(dataset))):
        image, target = dataset[index]
        counts.update(target["labels"].tolist())
        pixels += image.numel()
    report = {
        "status": "PASS",
        "csv": str(paths.csv_path),
        "images": len(dataset),
        "categories_including_background": len(dataset.category_names),
        "validated_samples": min(args.samples, len(dataset)),
        "decoded_tensor_values": pixels,
        "sample_instance_counts": dict(sorted(counts.items())),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

