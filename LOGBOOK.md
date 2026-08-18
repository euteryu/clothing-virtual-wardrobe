# Project logbook

## Current status

- Phase: first measured baseline ready.
- Last completed checkpoint: real-data T4 smoke training PASS.
- Current result: 64 images, 32 steps, final loss 1.5387 in 29.4 seconds; no
  held-out quality metric yet.
- Exact next action: train the fixed 4,000-image, 3-epoch baseline, then evaluate
  all epoch checkpoints on the frozen 500-image validation subset.

## Fixed decisions

- Dataset baseline: Fashionpedia / iMaterialist Fashion 2020 FGVC7.
- Task: garment instance segmentation, not virtual try-on.
- Official validation is model-selection data; personal photos form a later
  frozen applicability test.
- No run or check is described as passed until actually executed.

## Run notes

### 2026-08-18 - Kaggle stage 02 GPU smoke training PASS

- Notebook: `clothing-virtual-wardrobe-180826-1.1` live session.
- Command: `python kaggle/02_smoke_train.py --input-root /kaggle/input
  --max-images 64 --epochs 1`.
- Intended proof: complete real-data load, mask decode, pretrained Mask R-CNN
  construction, CUDA forward/backward optimization, checkpoint write, and
  machine-readable report on a bounded sample.
- Result: PASS on a Tesla T4. The 177 MB COCO-pretrained checkpoint downloaded
  successfully. Training used 64 images, one epoch, and 32 optimizer steps.
- Measurements: training runtime 29.4 seconds; step-1 loss 6.5087; step-10 loss
  2.4593; step-20 loss 1.8812; step-30 loss 2.7558; final loss 1.5387. Total
  observed cell time was roughly 50 seconds including setup/download.
- Artifact: `/kaggle/working/outputs/smoke/smoke_model.pt`.
- Interpretation: the end-to-end training path and accelerator are operational,
  and loss is non-degenerate and broadly decreasing. The noisy individual
  steps are expected for a tiny shuffled sample. This run does not establish
  held-out segmentation quality and its checkpoint will not seed the baseline.
- Decision: do not persist or attach the smoke checkpoint downstream. Size the
  first baseline from this measured throughput, train afresh from COCO weights,
  and add held-out COCO mask evaluation before spending a larger GPU budget.

### 2026-08-18 - Kaggle stage 01 initial failure and correction

- Notebook: `clothing-virtual-wardrobe-180826-1.1`.
- Command: `python kaggle/01_validate_data.py --input-root /kaggle/input`.
- Intended proof: discover the mounted iMaterialist Fashion 2020 data, index
  its annotations, and decode representative real masks before GPU training.
- Initial result: FAIL after roughly two minutes while reading `train.csv`.
  Python raised `_csv.Error: field larger than field limit (131072)`.
- Root cause: Fashionpedia contains high-resolution masks whose run-length
  encoded strings exceed Python's conservative default 128 KiB CSV-field
  limit. The original synthetic fixture exercised correct RLE geometry but used
  a very small field, so it did not represent this real input constraint.
- Engineering error: this should have been anticipated from the documented
  high-resolution RLE format and covered by a large-field fixture before the
  first Kaggle run. The dataset not being available locally explains why the
  failure was not directly reproduced, but does not remove that responsibility.
- Correction: commit `5784cf2` raises `csv.field_size_limit` to the largest
  platform-supported integer before parsing, with an overflow-safe fallback.
  A regression test now reads an RLE field larger than 128 KiB.
- Verification: `pytest -q` reported 4 passed and `ruff check .` passed locally.
  After `git pull`, the same stage-01 command was rerun in notebook 1.1 and the
  user reported PASS.
- Decision: proceed to stage 02 only after the corrected real-data validation
  passed. Future fixtures must cover not only format correctness but realistic
  field sizes and other dataset-scale extremes.

### 2026-08-18 - scaffold

- Created portable data discovery, validation, smoke training, and CPU tests.
- No real dataset or GPU execution has occurred yet.
- Next: run the three Kaggle preflight/smoke commands and record measurements.
