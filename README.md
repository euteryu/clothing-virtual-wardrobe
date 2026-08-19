# Clothing Segmenter -> Virtual Wardrobe

A Kaggle-first learning project that fine-tunes an instance-segmentation model
to turn an outfit photo into separate labelled garment masks and cutouts.

## Current milestone

Prove a reproducible Fashionpedia baseline. The interface is deliberately out
of scope until masks work on held-out and ordinary phone photos.

## Kaggle quick start

1. Create a Kaggle notebook and attach the **iMaterialist Fashion 2020 FGVC7**
   competition data. You may need to join/accept the competition rules first.
2. Enable one T4 GPU.
3. In the first cell, clone this repository and install it:

   ```python
   !git clone YOUR_GITHUB_REPOSITORY_URL /kaggle/working/wardrobe
   %cd /kaggle/working/wardrobe
   !pip install -q -e .
   ```

4. Run the environment/input check:

   ```python
   !python kaggle/00_environment_check.py
   ```

5. Validate the real annotations (the script discovers common Kaggle layouts):

   ```python
   !python kaggle/01_validate_data.py --input-root /kaggle/input
   ```

6. Run the deliberately small GPU smoke test:

   ```python
   !python kaggle/02_smoke_train.py --input-root /kaggle/input --max-images 64 --epochs 1
   ```

Do not start a larger run until all three commands finish with `"status":
"PASS"`. Save a notebook version after a successful smoke test so its outputs
persist.

## First measured baseline

After the smoke result has been logged, use a fresh Kaggle notebook/session:

```python
!python kaggle/03_train_baseline.py --input-root /kaggle/input \
  --max-images 4000 --epochs 3
```

This writes a resumable checkpoint after every epoch. Evaluate all three on the
same frozen 500-image validation subset before selecting one:

```python
!python kaggle/04_evaluate.py --input-root /kaggle/input \
  --checkpoint /kaggle/working/outputs/baseline/checkpoint_epoch_01.pt
!python kaggle/04_evaluate.py --input-root /kaggle/input \
  --checkpoint /kaggle/working/outputs/baseline/checkpoint_epoch_02.pt
!python kaggle/04_evaluate.py --input-root /kaggle/input \
  --checkpoint /kaggle/working/outputs/baseline/checkpoint_epoch_03.pt
```

Save & Run All is required for this baseline because its checkpoints, manifests,
and reports are expensive outputs that must survive the live session.

## Local verification (CPU only)

```powershell
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

## Frozen personal-photo applicability test

Keep the 20 consented test photos and the selected epoch-03 checkpoint outside
Git. On a CPU-only Windows computer, install the project, then run:

```powershell
python -m pip install -e ".[dev]"
python kaggle/06_personal_inference.py `
  --photo-dir "C:\path\to\wardrobe_personal_test" `
  --checkpoint "C:\path\to\checkpoint_epoch_03.pt" `
  --output-dir "C:\path\to\wardrobe_personal_results" `
  --device cpu
```

The command requires exactly 20 `.jpg`, `.jpeg`, `.png`, or `.webp` files by
default. It preserves the frozen epoch-03 policy: main garment classes 1-13,
confidence 0.6, predicted mask threshold 0.5, and maximum side 1,024. It writes
labelled overlays, transparent cutouts, and `personal_inference_report.json`.
CPU inference is expected to be slower than Kaggle T4 inference. Do not tune
the checkpoint or thresholds from these personal-photo results.

## Single improvement run

Attach the Fashionpedia competition data and the preserved notebook-1.2 output
to a fresh one-T4 Kaggle notebook. Resolve the epoch-03 checkpoint without
hard-coding a dataset slug, then continue only through epoch 05:

```python
from pathlib import Path

matches = list(Path("/kaggle/input").rglob("checkpoint_epoch_03.pt"))
assert len(matches) == 1, matches
checkpoint = matches[0]
print(checkpoint)
```

```python
!python kaggle/07_continue_training.py \
  --input-root /kaggle/input \
  --checkpoint {checkpoint}
```

Save & Run All so the epoch-04 and epoch-05 checkpoints persist. Evaluate both
with `kaggle/04_evaluate.py` on the same 500-image validation subset and compare
them with epoch 03 by mask AP only.

Training data, credentials, checkpoints, and predictions must not be committed.
