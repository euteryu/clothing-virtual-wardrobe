from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from .data import FashionpediaDataset, FashionpediaPaths, collate_fn
from .evaluate import resize_for_evaluation
from .model import build_model


def garment_match_counts(
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_masks: torch.Tensor,
    target_labels: torch.Tensor,
    target_masks: torch.Tensor,
    threshold: float,
    max_label: int = 13,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedily match score-ranked masks of the same main-garment class."""
    pred_keep = (pred_scores >= threshold) & (pred_labels >= 1) & (pred_labels <= max_label)
    target_keep = (target_labels >= 1) & (target_labels <= max_label)
    pred_indices = torch.where(pred_keep)[0]
    pred_indices = pred_indices[torch.argsort(pred_scores[pred_indices], descending=True)]
    target_indices = torch.where(target_keep)[0]
    matched_targets: set[int] = set()
    true_positive = 0
    for pred_index in pred_indices:
        label = int(pred_labels[pred_index])
        prediction = pred_masks[pred_index].bool()
        best_iou = 0.0
        best_target: int | None = None
        for target_index in target_indices:
            target_number = int(target_index)
            if target_number in matched_targets or int(target_labels[target_index]) != label:
                continue
            reference = target_masks[target_index].bool()
            intersection = torch.logical_and(prediction, reference).sum().item()
            union = torch.logical_or(prediction, reference).sum().item()
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_target = target_number
        if best_target is not None and best_iou >= iou_threshold:
            matched_targets.add(best_target)
            true_positive += 1
    false_positive = len(pred_indices) - true_positive
    false_negative = len(target_indices) - true_positive
    return true_positive, false_positive, false_negative


def garment_threshold_counts(
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_masks: torch.Tensor,
    target_labels: torch.Tensor,
    target_masks: torch.Tensor,
    thresholds: tuple[float, ...],
    max_label: int = 13,
    iou_threshold: float = 0.5,
) -> dict[float, tuple[int, int, int]]:
    """Reuse one greedy score-ranked matching pass for nested score thresholds."""
    minimum_threshold = min(thresholds)
    keep = (
        (pred_scores >= minimum_threshold) & (pred_labels >= 1) & (pred_labels <= max_label)
    )
    pred_indices = torch.where(keep)[0]
    pred_indices = pred_indices[torch.argsort(pred_scores[pred_indices], descending=True)]
    target_indices = torch.where((target_labels >= 1) & (target_labels <= max_label))[0]
    matched_targets: set[int] = set()
    prediction_events: list[tuple[float, bool]] = []
    for pred_index in pred_indices:
        label = int(pred_labels[pred_index])
        prediction = pred_masks[pred_index].bool()
        best_iou = 0.0
        best_target: int | None = None
        for target_index in target_indices:
            target_number = int(target_index)
            if target_number in matched_targets or int(target_labels[target_index]) != label:
                continue
            reference = target_masks[target_index].bool()
            intersection = torch.logical_and(prediction, reference).sum().item()
            union = torch.logical_or(prediction, reference).sum().item()
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_target = target_number
        matched = best_target is not None and best_iou >= iou_threshold
        if matched:
            matched_targets.add(best_target)
        prediction_events.append((float(pred_scores[pred_index]), matched))
    results: dict[float, tuple[int, int, int]] = {}
    for threshold in thresholds:
        selected = [matched for score, matched in prediction_events if score >= threshold]
        true_positive = sum(selected)
        results[threshold] = (
            true_positive,
            len(selected) - true_positive,
            len(target_indices) - true_positive,
        )
    return results


def _scores(counts: tuple[int, int, int]) -> dict[str, float | int]:
    true_positive, false_positive, false_negative = counts
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _render_triptych(
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    threshold: float,
    category_names: list[str],
) -> torch.Tensor:
    base = (image * 255).clamp(0, 255).to(torch.uint8)
    target_keep = (target["labels"] >= 1) & (target["labels"] <= 13)
    prediction_keep = (
        (output["scores"] >= threshold) & (output["labels"] >= 1) & (output["labels"] <= 13)
    )
    truth = base.clone()
    if target_keep.any():
        truth = draw_segmentation_masks(truth, target["masks"][target_keep].bool(), alpha=0.45)
        truth = draw_bounding_boxes(
            truth,
            target["boxes"][target_keep],
            labels=[category_names[int(label)] for label in target["labels"][target_keep]],
            width=2,
        )
    prediction = base.clone()
    if prediction_keep.any():
        masks = output["masks"][prediction_keep, 0] >= 0.5
        prediction = draw_segmentation_masks(prediction, masks, alpha=0.45)
        labels = [
            f"{category_names[int(label)]} {float(score):.2f}"
            for label, score in zip(
                output["labels"][prediction_keep], output["scores"][prediction_keep], strict=True
            )
        ]
        prediction = draw_bounding_boxes(
            prediction, output["boxes"][prediction_keep], labels=labels, width=2
        )
    return torch.cat([base, truth, prediction], dim=2)


def analyze_operating_point(
    paths: FashionpediaPaths,
    checkpoint_path: Path,
    output_dir: Path,
    max_images: int = 500,
    workers: int = 2,
    seed: int = 2026,
    max_side: int = 1024,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Threshold analysis requires a Kaggle GPU")
    device = torch.device("cuda")
    dataset = FashionpediaDataset(paths, "val", max_images=max_images, seed=seed)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(len(dataset.category_names), pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate_fn
    )
    totals = {threshold: [0, 0, 0] for threshold in thresholds}
    per_image: list[dict[float, tuple[int, int, int]]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            image, target = resize_for_evaluation(images[0], targets[0], max_side)
            output = model([image.to(device)])[0]
            pred_labels = output["labels"].cpu()
            pred_scores = output["scores"].cpu()
            pred_masks = (output["masks"][:, 0] >= 0.5).to(torch.uint8).cpu()
            image_counts = garment_threshold_counts(
                pred_labels,
                pred_scores,
                pred_masks,
                target["labels"],
                target["masks"],
                thresholds,
            )
            for threshold, counts in image_counts.items():
                totals[threshold] = [sum(values) for values in zip(totals[threshold], counts, strict=True)]
            per_image.append(image_counts)
            if index == 0 or (index + 1) % 100 == 0:
                print(json.dumps({"analyzed_images": index + 1}))
    sweep = {str(threshold): _scores(tuple(totals[threshold])) for threshold in thresholds}
    selected_threshold = max(thresholds, key=lambda value: float(sweep[str(value)]["f1"]))
    eligible = [
        index
        for index in range(len(dataset))
        if sum(per_image[index][selected_threshold][::2]) > 0
    ]
    ranked = sorted(
        eligible,
        key=lambda index: float(_scores(per_image[index][selected_threshold])["f1"]),
    )
    review_indices = [("worst", index) for index in ranked[:6]] + [
        ("best", index) for index in ranked[-6:]
    ]
    review_dir = output_dir / "qualitative"
    review_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for rank, (group, index) in enumerate(review_indices, start=1):
            image, target = dataset[index]
            image, target = resize_for_evaluation(image, target, max_side)
            output = model([image.to(device)])[0]
            output = {key: value.cpu() for key, value in output.items()}
            triptych = _render_triptych(
                image, target, output, selected_threshold, dataset.category_names
            )
            Image.fromarray(triptych.permute(1, 2, 0).numpy()).save(
                review_dir / f"{group}_{rank:02d}_{dataset.image_ids[index]}.png"
            )
    report: dict[str, Any] = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state["epoch"]),
        "validation_images": len(dataset),
        "main_garment_classes": dataset.category_names[1:14],
        "mask_iou_match_threshold": 0.5,
        "confidence_sweep": sweep,
        "selected_confidence_threshold": selected_threshold,
        "selection_rule": "maximum micro F1 on frozen validation data",
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "qualitative_directory": str(review_dir),
        "triptych_order": ["input", "ground_truth", "prediction"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "operating_point_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
