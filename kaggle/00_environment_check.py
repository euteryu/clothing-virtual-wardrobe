"""Print Kaggle environment facts and prove a real CUDA kernel runs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch


def directory_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "exists": True,
        "free_gib": round(usage.free / 2**30, 2),
        "total_gib": round(usage.total / 2**30, 2),
        "children": sorted(child.name for child in path.iterdir())[:100],
    }


def main() -> None:
    input_root = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
    working_root = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))
    cuda_ok = False
    cuda_error = None
    device_name = None
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            value = (torch.ones(1024, device="cuda") * 2).sum()
            torch.cuda.synchronize()
            cuda_ok = float(value) == 2048.0
        except RuntimeError as exc:  # report the accelerator failure cleanly
            cuda_error = repr(exc)
    report = {
        "status": "PASS" if cuda_ok else "FAIL",
        "current_directory": str(Path.cwd()),
        "input": directory_summary(input_root),
        "working": directory_summary(working_root),
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_kernel_ok": cuda_ok,
        "cuda_device": device_name,
        "cuda_error": cuda_error,
    }
    print(json.dumps(report, indent=2))
    if not cuda_ok:
        raise SystemExit("FAIL: enable a Kaggle GPU before training")


if __name__ == "__main__":
    main()
