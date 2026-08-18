from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wardrobe_seg.data import FashionpediaDataset, decode_rle, discover_fashionpedia


def test_decode_rle_uses_kaggle_column_major_order() -> None:
    mask = decode_rle("1 2 5 2", height=3, width=3)
    expected = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_fixture_discovery_and_dataset(tmp_path: Path) -> None:
    root = tmp_path / "mounted" / "fashion"
    image_dir = root / "train"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (3, 3), "white").save(image_dir / "abc.jpg")
    (root / "label_descriptions.json").write_text(
        json.dumps({"categories": [{"id": 0, "name": "shirt"}]}), encoding="utf-8"
    )
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["ImageId", "EncodedPixels", "Height", "Width", "ClassId"]
        )
        writer.writeheader()
        writer.writerow(
            {"ImageId": "abc", "EncodedPixels": "1 2 5 2", "Height": 3, "Width": 3, "ClassId": 0}
        )
    paths = discover_fashionpedia(tmp_path)
    dataset = FashionpediaDataset(paths, "all")
    image, target = dataset[0]
    assert image.shape == (3, 3, 3)
    assert target["labels"].tolist() == [1]
    assert target["boxes"].tolist() == [[0.0, 0.0, 2.0, 3.0]]
    assert int(target["masks"].sum()) == 4

