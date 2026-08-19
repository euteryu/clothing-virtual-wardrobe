from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as nn_functional
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from .model import build_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAIN_GARMENT_MAX_LABEL = 13


def discover_personal_images(photo_dir: Path) -> list[Path]:
    if not photo_dir.is_dir():
        raise NotADirectoryError(photo_dir)
    return sorted(
        path for path in photo_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _resize_image(image: torch.Tensor, max_side: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    return nn_functional.interpolate(
        image.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    )[0]


def _save_outputs(
    image: torch.Tensor,
    output: dict[str, torch.Tensor],
    category_names: list[str],
    image_name: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    base = (image * 255).clamp(0, 255).to(torch.uint8).cpu()
    masks = output["masks"][:, 0] >= 0.5
    overlay = base.clone()
    if len(masks):
        overlay = draw_segmentation_masks(overlay, masks, alpha=0.45)
        labels = [
            f"{category_names[int(label)]} {float(score):.2f}"
            for label, score in zip(output["labels"], output["scores"], strict=True)
        ]
        overlay = draw_bounding_boxes(overlay, output["boxes"], labels=labels, width=2)
    overlay_dir = output_dir / "overlays"
    cutout_dir = output_dir / "cutouts"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.permute(1, 2, 0).numpy()).save(overlay_dir / f"{image_name}.png")

    rgb = base.permute(1, 2, 0).numpy()
    predictions: list[dict[str, Any]] = []
    for index, (label, score, box, mask) in enumerate(
        zip(output["labels"], output["scores"], output["boxes"], masks, strict=True), start=1
    ):
        binary_mask = mask.numpy()
        rgba = np.dstack([rgb, binary_mask.astype(np.uint8) * 255])
        x1, y1, x2, y2 = [round(value) for value in box.tolist()]
        height, width = binary_mask.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        cutout_name = f"{image_name}_{index:02d}_{category_names[int(label)].replace(', ', '_')}.png"
        Image.fromarray(rgba[y1:y2, x1:x2]).save(cutout_dir / cutout_name)
        predictions.append(
            {
                "category": category_names[int(label)],
                "score": round(float(score), 6),
                "box_xyxy": [x1, y1, x2, y2],
                "cutout": str(Path("cutouts") / cutout_name),
            }
        )
    return predictions


def run_personal_inference(
    photo_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    expected_images: int = 20,
    max_side: int = 1024,
    confidence_threshold: float = 0.6,
    device_name: str = "auto",
) -> dict[str, Any]:
    images = discover_personal_images(photo_dir)
    if len(images) != expected_images:
        raise ValueError(f"Expected exactly {expected_images} test images; found {len(images)}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_dir.resolve() == photo_dir.resolve() or photo_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Output directory must be outside the private input photo directory")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    category_names = list(state["categories"])
    if len(category_names) <= MAIN_GARMENT_MAX_LABEL:
        raise ValueError("Checkpoint does not contain the expected main garment categories")
    model = build_model(len(category_names), pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, image_path in enumerate(images, start=1):
            with Image.open(image_path) as source:
                pil_image = ImageOps.exif_transpose(source).convert("RGB")
            image = _resize_image(pil_to_tensor(pil_image).float().div(255), max_side)
            raw = model([image.to(device)])[0]
            keep = (
                (raw["scores"] >= confidence_threshold)
                & (raw["labels"] >= 1)
                & (raw["labels"] <= MAIN_GARMENT_MAX_LABEL)
            )
            output = {key: value[keep].cpu() for key, value in raw.items()}
            predictions = _save_outputs(
                image, output, category_names, image_path.stem, output_dir
            )
            record = {
                "image": image_path.name,
                "inference_size": [int(image.shape[-1]), int(image.shape[-2])],
                "prediction_count": len(predictions),
                "predictions": predictions,
            }
            records.append(record)
            print(json.dumps({"processed": index, "image": image_path.name, "predictions": len(predictions)}))

    report: dict[str, Any] = {
        "status": "PASS",
        "purpose": "frozen personal-photo applicability test; not model selection",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state["epoch"]),
        "device": str(device),
        "photo_count": len(images),
        "confidence_threshold": confidence_threshold,
        "mask_threshold": 0.5,
        "main_garment_classes": category_names[1 : MAIN_GARMENT_MAX_LABEL + 1],
        "max_side": max_side,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "images": records,
    }
    (output_dir / "personal_inference_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
