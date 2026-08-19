from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import FashionpediaDataset, FashionpediaPaths, collate_fn
from .model import build_model


def _move_targets(
    targets: tuple[dict[str, torch.Tensor], ...], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def train_epochs(
    paths: FashionpediaPaths,
    output_dir: Path,
    max_images: int,
    epochs: int,
    batch_size: int,
    workers: int,
    seed: int = 2026,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Train through `epochs`, saving resumable state after every completed epoch."""
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a Kaggle GPU; CUDA is unavailable")
    device = torch.device("cuda")
    dataset = FashionpediaDataset(paths, "train", max_images=max_images, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        generator=generator,
        persistent_workers=workers > 0,
    )
    model = build_model(len(dataset.category_names)).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.005 * batch_size / 2,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["categories"] != dataset.category_names or state["seed"] != seed:
            raise ValueError("Resume checkpoint category mapping or seed does not match")
        if int(state["max_images"]) != max_images:
            raise ValueError("Resume checkpoint training subset size does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
    if start_epoch >= epochs:
        raise ValueError(f"Checkpoint already completed epoch {start_epoch}; target is {epochs}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "split": "train",
        "image_ids": dataset.image_ids,
        "categories": dataset.category_names,
    }
    (output_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    epoch_reports: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs):
        model.train()
        losses: list[float] = []
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for step, (images, targets) in enumerate(loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            moved_targets = _move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, moved_targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}: {float(loss)}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if step == 1 or step % 100 == 0:
                print(
                    json.dumps(
                        {"epoch": epoch + 1, "step": step, "loss": losses[-1], "lr": optimizer.param_groups[0]["lr"]}
                    )
                )
        scheduler.step()
        torch.cuda.synchronize()
        epoch_report = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "mean_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "runtime_seconds": round(time.perf_counter() - epoch_started, 2),
        }
        epoch_reports.append(epoch_report)
        checkpoint = output_dir / f"checkpoint_epoch_{epoch + 1:02d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "categories": dataset.category_names,
                "seed": seed,
                "max_images": max_images,
            },
            checkpoint,
        )
        print(json.dumps({**epoch_report, "checkpoint": str(checkpoint)}))
    report: dict[str, Any] = {
        "status": "PASS",
        "images": len(dataset),
        "target_epochs": epochs,
        "start_epoch": start_epoch,
        "completed_epoch": epochs,
        "batch_size": batch_size,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "cuda_device": torch.cuda.get_device_name(0),
        "epochs": epoch_reports,
        "latest_checkpoint": str(output_dir / f"checkpoint_epoch_{epochs:02d}.pt"),
    }
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_baseline(
    paths: FashionpediaPaths,
    output_dir: Path,
    max_images: int,
    epochs: int,
    batch_size: int,
    workers: int,
    seed: int = 2026,
) -> dict[str, object]:
    report = train_epochs(paths, output_dir, max_images, epochs, batch_size, workers, seed)
    checkpoint = Path(str(report["latest_checkpoint"]))
    smoke_checkpoint = output_dir / "smoke_model.pt"
    shutil.copy2(checkpoint, smoke_checkpoint)
    report["checkpoint"] = str(smoke_checkpoint)
    report["final_loss"] = report["epochs"][-1]["final_loss"]
    (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
