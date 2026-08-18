from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import FashionpediaDataset, FashionpediaPaths, collate_fn
from .model import build_model


def train_baseline(
    paths: FashionpediaPaths,
    output_dir: Path,
    max_images: int,
    epochs: int,
    batch_size: int,
    workers: int,
    seed: int = 2026,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a Kaggle GPU; CUDA is unavailable")
    device = torch.device("cuda")
    dataset = FashionpediaDataset(paths, "train", max_images=max_images, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    model = build_model(len(dataset.category_names)).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005 * batch_size / 2,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scaler = torch.amp.GradScaler("cuda")
    started = time.perf_counter()
    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        for step, (images, targets) in enumerate(loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {float(loss)}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if step == 1 or step % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "loss": losses[-1]}))
    torch.cuda.synchronize()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "smoke_model.pt"
    torch.save(
        {"model": model.state_dict(), "categories": dataset.category_names, "seed": seed}, checkpoint
    )
    report: dict[str, object] = {
        "status": "PASS",
        "images": len(dataset),
        "epochs": epochs,
        "steps": len(losses),
        "final_loss": losses[-1],
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "checkpoint": str(checkpoint),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

