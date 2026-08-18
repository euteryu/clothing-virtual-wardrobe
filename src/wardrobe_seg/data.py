from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


@dataclass(frozen=True)
class FashionpediaPaths:
    csv_path: Path
    image_dir: Path
    labels_path: Path


def discover_fashionpedia(input_root: Path) -> FashionpediaPaths:
    """Discover the official Kaggle competition layout without relying on its slug."""
    csvs = list(input_root.rglob("train.csv"))
    labels = list(input_root.rglob("label_descriptions.json"))
    candidates: list[FashionpediaPaths] = []
    for csv_path in csvs:
        base = csv_path.parent
        image_dirs = [base / "train", base / "train_images"]
        image_dir = next((p for p in image_dirs if p.is_dir()), None)
        label_path = next((p for p in labels if p.parent == base), None)
        if image_dir and label_path:
            candidates.append(FashionpediaPaths(csv_path, image_dir, label_path))
    if len(candidates) != 1:
        found = [str(p.csv_path.parent) for p in candidates]
        raise RuntimeError(
            "Expected exactly one Fashionpedia Kaggle layout containing train.csv, "
            f"train/, and label_descriptions.json; found {found}"
        )
    return candidates[0]


def decode_rle(encoded: str, height: int, width: int) -> np.ndarray:
    """Decode Kaggle's one-indexed, column-major run-length mask."""
    values = np.fromstring(encoded, dtype=np.int64, sep=" ")
    if values.size == 0 or values.size % 2:
        raise ValueError("RLE must contain start/length pairs")
    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths
    if starts.min() < 0 or ends.max() > height * width:
        raise ValueError("RLE coordinates are outside the image")
    flat = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends, strict=True):
        flat[start:end] = 1
    return flat.reshape((height, width), order="F")


def load_categories(labels_path: Path) -> tuple[dict[int, int], list[str]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda x: int(x["id"]))
    source_ids = [int(item["id"]) for item in categories]
    mapping = {source_id: index + 1 for index, source_id in enumerate(source_ids)}
    return mapping, ["__background__", *[item["name"] for item in categories]]


def read_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    # High-resolution Fashionpedia masks can exceed Python's conservative
    # 128 KiB default CSV-field limit. Use the platform's largest accepted
    # value because these are trusted, mounted competition annotations.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ImageId", "EncodedPixels", "Height", "Width", "ClassId"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            rows[row["ImageId"]].append(row)
    return dict(rows)


def split_image_ids(
    image_ids: list[str], split: str, val_fraction: float = 0.1, seed: int = 2026
) -> list[str]:
    if split not in {"train", "val", "all"}:
        raise ValueError("split must be train, val, or all")
    ids = sorted(image_ids)
    if split == "all":
        return ids
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = max(1, round(len(ids) * val_fraction))
    return sorted(ids[val_count:] if split == "train" else ids[:val_count])


def select_class_aware_ids(
    image_ids: list[str],
    rows: dict[str, list[dict[str, str]]],
    max_images: int | None,
    seed: int = 2026,
    minimum_per_class: int = 5,
) -> list[str]:
    """Guarantee a small class-coverage floor, then sample the remainder uniformly."""
    ids = sorted(image_ids)
    if max_images is None or max_images >= len(ids):
        return ids
    if max_images < 1:
        raise ValueError("max_images must be positive")
    rng = random.Random(seed)
    buckets: dict[int, list[str]] = defaultdict(list)
    for image_id in ids:
        for class_id in {int(row["ClassId"]) for row in rows[image_id]}:
            buckets[class_id].append(image_id)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[str] = []
    seen: set[str] = set()
    offsets = {class_id: 0 for class_id in buckets}
    class_ids = sorted(buckets)
    for _ in range(minimum_per_class):
        for class_id in class_ids:
            bucket = buckets[class_id]
            offset = offsets[class_id]
            while offset < len(bucket) and bucket[offset] in seen:
                offset += 1
            offsets[class_id] = offset + 1
            if offset < len(bucket):
                image_id = bucket[offset]
                selected.append(image_id)
                seen.add(image_id)
                if len(selected) == max_images:
                    return selected
    if len(selected) < max_images:
        remainder = [image_id for image_id in ids if image_id not in seen]
        rng.shuffle(remainder)
        selected.extend(remainder[: max_images - len(selected)])
    return selected


class FashionpediaDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        paths: FashionpediaPaths,
        split: str,
        max_images: int | None = None,
        seed: int = 2026,
    ) -> None:
        self.paths = paths
        self.rows = read_rows(paths.csv_path)
        self.category_map, self.category_names = load_categories(paths.labels_path)
        self.image_ids = split_image_ids(list(self.rows), split, seed=seed)
        self.image_ids = select_class_aware_ids(self.image_ids, self.rows, max_images, seed)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        rows = self.rows[image_id]
        image_path = self.paths.image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        masks: list[np.ndarray] = []
        labels: list[int] = []
        boxes: list[list[float]] = []
        for row in rows:
            if int(row["Height"]) != height or int(row["Width"]) != width:
                raise ValueError(f"Annotation/image size mismatch for {image_id}")
            mask = decode_rle(row["EncodedPixels"], height, width)
            ys, xs = np.where(mask)
            if not len(xs):
                continue
            masks.append(mask)
            labels.append(self.category_map[int(row["ClassId"])])
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        if not masks:
            raise ValueError(f"No valid instances for {image_id}")
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([float(mask.sum()) for mask in masks]),
            "iscrowd": torch.zeros(len(masks), dtype=torch.int64),
        }
        return pil_to_tensor(image).float().div(255), target


def collate_fn(batch: list[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return tuple(zip(*batch, strict=True))
