from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.nn import functional as nn_functional
from torch.utils.data import DataLoader

from .data import FashionpediaDataset, FashionpediaPaths, collate_fn
from .model import build_model


def _encode_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def resize_for_evaluation(
    image: torch.Tensor, target: dict[str, torch.Tensor], max_side: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bound postprocessed mask memory by resizing image and target together."""
    height, width = image.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, target
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    resized_image = nn_functional.interpolate(
        image.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False
    )[0]
    resized_masks = nn_functional.interpolate(
        target["masks"].unsqueeze(1).float(), size=(new_height, new_width), mode="nearest"
    )[:, 0].to(torch.uint8)
    resized_boxes = target["boxes"].clone()
    resized_boxes[:, [0, 2]] *= new_width / width
    resized_boxes[:, [1, 3]] *= new_height / height
    resized_target = dict(target)
    resized_target["masks"] = resized_masks
    resized_target["boxes"] = resized_boxes
    resized_target["area"] = resized_masks.flatten(1).sum(1).float()
    return resized_image, resized_target


def evaluate_checkpoint(
    paths: FashionpediaPaths,
    checkpoint_path: Path,
    output_dir: Path,
    max_images: int = 500,
    batch_size: int = 1,
    workers: int = 2,
    seed: int = 2026,
    max_side: int = 1024,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a Kaggle GPU; CUDA is unavailable")
    device = torch.device("cuda")
    dataset = FashionpediaDataset(paths, "val", max_images=max_images, seed=seed)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["categories"] != dataset.category_names:
        raise ValueError("Checkpoint category mapping does not match validation data")
    model = build_model(len(dataset.category_names), pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    ground_truth: dict[str, Any] = {
        "info": {"description": "Frozen Fashionpedia validation subset"},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in enumerate(dataset.category_names)
            if category_id > 0
        ],
    }
    predictions: list[dict[str, Any]] = []
    annotation_id = 1
    processed = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            resized = [
                resize_for_evaluation(image, target, max_side)
                for image, target in zip(images, targets, strict=True)
            ]
            images, targets = tuple(zip(*resized, strict=True))
            device_images = [image.to(device, non_blocking=True) for image in images]
            outputs = model(device_images)
            for image, target, output in zip(images, targets, outputs, strict=True):
                image_id = processed + 1
                height, width = image.shape[-2:]
                ground_truth["images"].append(
                    {"id": image_id, "width": width, "height": height, "file_name": dataset.image_ids[processed]}
                )
                for box, label, mask, area in zip(
                    target["boxes"], target["labels"], target["masks"], target["area"], strict=True
                ):
                    x1, y1, x2, y2 = box.tolist()
                    ground_truth["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "segmentation": _encode_mask(mask.numpy()),
                            "area": float(area),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
                output_boxes = output["boxes"].cpu()
                output_labels = output["labels"].cpu()
                output_scores = output["scores"].cpu()
                output_masks = (output["masks"][:, 0] >= 0.5).to(torch.uint8).cpu()
                del output
                for box, label, score, mask in zip(
                    output_boxes, output_labels, output_scores, output_masks, strict=True
                ):
                    binary_mask = mask.numpy()
                    if not binary_mask.any():
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "segmentation": _encode_mask(binary_mask),
                            "score": float(score),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                        }
                    )
                processed += 1
                if processed == 1 or processed % 100 == 0:
                    print(json.dumps({"evaluated_images": processed, "predictions": len(predictions)}))
    torch.cuda.synchronize()
    if not predictions:
        raise RuntimeError("Model produced no non-empty validation masks")
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    per_class_ap: dict[str, float | None] = {}
    for class_index, name in enumerate(dataset.category_names[1:]):
        valid = precision[:, :, class_index]
        valid = valid[valid > -1]
        per_class_ap[name] = float(valid.mean()) if valid.size else None
    stats = evaluator.stats.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "split": "val", "image_ids": dataset.image_ids}
    (output_dir / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report: dict[str, Any] = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state["epoch"]),
        "validation_images": len(dataset),
        "ground_truth_instances": len(ground_truth["annotations"]),
        "predicted_instances": len(predictions),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "evaluation_max_side": max_side,
        "batch_size": batch_size,
        "mask_ap": stats[0],
        "mask_ap50": stats[1],
        "mask_ap75": stats[2],
        "mask_ar100": stats[8],
        "per_class_mask_ap": per_class_ap,
    }
    (output_dir / f"evaluation_epoch_{state['epoch']:02d}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
