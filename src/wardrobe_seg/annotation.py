from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils
from torchvision.transforms.functional import pil_to_tensor

from .model import build_model
from .personal import MAIN_GARMENT_MAX_LABEL, _resize_image, discover_personal_images


def encode_coco_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode one binary mask as JSON-safe compressed COCO RLE."""
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}


def prepare_cvat_preannotations(
    photo_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    expected_images: int = 20,
    max_side: int = 1024,
    confidence_threshold: float = 0.3,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Create resized images and recall-oriented COCO masks for human correction."""
    images = discover_personal_images(photo_dir)
    if len(images) != expected_images:
        raise ValueError(f"Expected exactly {expected_images} images; found {len(images)}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite annotation package: {output_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be between zero and one")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    category_names = list(state["categories"])
    model = build_model(len(category_names), pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    started = time.perf_counter()
    with torch.inference_mode():
        for image_id, image_path in enumerate(images, start=1):
            with Image.open(image_path) as source:
                pil_image = ImageOps.exif_transpose(source).convert("RGB")
            image = _resize_image(pil_to_tensor(pil_image).float().div(255), max_side)
            output_name = f"{image_path.stem}.png"
            Image.fromarray(
                (image * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            ).save(image_dir / output_name)
            height, width = image.shape[-2:]
            coco_images.append(
                {"id": image_id, "file_name": output_name, "height": height, "width": width}
            )

            raw = model([image.to(device)])[0]
            keep = (
                (raw["scores"] >= confidence_threshold)
                & (raw["labels"] >= 1)
                & (raw["labels"] <= MAIN_GARMENT_MAX_LABEL)
            )
            masks = (raw["masks"][keep, 0] >= 0.5).to(torch.uint8).cpu().numpy()
            labels = raw["labels"][keep].cpu().tolist()
            scores = raw["scores"][keep].cpu().tolist()
            for mask, label, score in zip(masks, labels, scores, strict=True):
                encoded = encode_coco_mask(mask)
                bbox = mask_utils.toBbox(
                    {"size": encoded["size"], "counts": encoded["counts"].encode("ascii")}
                ).tolist()
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(label),
                        "segmentation": encoded,
                        "area": int(mask.sum()),
                        "bbox": [round(float(value), 2) for value in bbox],
                        "iscrowd": 0,
                        "score": round(float(score), 6),
                    }
                )
                annotation_id += 1
            print(json.dumps({"prepared": image_id, "proposals": len(masks)}))

    categories = [
        {"id": index, "name": category_names[index], "supercategory": "garment"}
        for index in range(1, MAIN_GARMENT_MAX_LABEL + 1)
    ]
    coco = {
        "info": {
            "description": "Model-assisted garment masks requiring human correction",
            "checkpoint_epoch": int(state["epoch"]),
            "proposal_confidence_threshold": confidence_threshold,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": categories,
    }
    annotation_path = output_dir / "instances_preannotations.json"
    annotation_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    instructions = output_dir / "README.txt"
    instructions.write_text(
        "Import images/ and instances_preannotations.json into CVAT as COCO 1.0.\n"
        "Review every image: delete false instances, correct class/mask boundaries, and add every "
        "missed garment. Export the corrected task as COCO 1.0 instances_default.json.\n"
        "Do not train from this proposal file before human correction.\n",
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "purpose": "CVAT preannotations for human-corrected domain adaptation",
        "checkpoint_epoch": int(state["epoch"]),
        "device": str(device),
        "images": len(coco_images),
        "proposed_instances": len(annotations),
        "confidence_threshold": confidence_threshold,
        "max_side": max_side,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "annotation_file": str(annotation_path),
        "archive": str(output_dir.with_suffix(".zip")),
    }
    (output_dir / "preannotation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
    return report
