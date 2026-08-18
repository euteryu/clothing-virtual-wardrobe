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

### 2026-08-18 - corrected evaluation memory smoke PASS

- Notebook: `clothing-virtual-wardrobe-180826-1.3-evaluation` live session.
- Inputs: Fashionpedia competition data plus the preserved output of notebook
  1.2. No training was repeated. The epoch-03 checkpoint resolved to
  `/kaggle/input/notebooks/minseokryu5432/clothing-virtual-wardrobe-180826-1-2/outputs/baseline/checkpoint_epoch_03.pt`.
- Command configuration: epoch-03 checkpoint, five frozen validation images,
  batch size 1, and maximum evaluation side 1,024 pixels.
- Intended proof: verify that coordinated image/target resizing and single-image
  inference prevent the full-resolution mask-postprocessing OOM before running
  the complete held-out evaluation.
- Result: PASS. Five images containing 44 reference instances produced 223
  predicted instances in 6.37 seconds without an OOM. Reported smoke metrics
  were mask AP 0.5240, AP50 0.7159, AP75 0.5828, and AR100 0.5603.
- Interpretation: the memory correction is operational. Metrics from only five
  deliberately bounded images are not baseline evidence; absent classes appear
  as null and uncertainty is extreme. They must not be quoted as final quality.
- Decision: run all three epoch checkpoints against the exact same frozen
  500-image validation selection at batch size 1 and max side 1,024. Select the
  checkpoint by mask AP only after all three reports complete. Submit this
  evaluation notebook with Save & Run All so reports persist.

### 2026-08-18 - baseline training PASS; evaluation OOM

- Notebook: `clothing-virtual-wardrobe-180826-1.2`, committed Save & Run All.
- Training result: PASS on Tesla T4 using 4,000 images, batch size 2, and three
  epochs (6,000 optimizer steps). Total measured training runtime was 5,419.19
  seconds. Epoch runtimes were 1,825.15, 1,796.07, and 1,796.15 seconds.
- Mean training loss improved from 1.1959 to 0.9272 to 0.7382. Epoch checkpoints
  01, 02, and 03 were successfully written under
  `/kaggle/working/outputs/baseline/` and preserved in the notebook output.
- Evaluation result: FAIL for all three checkpoint cells immediately after the
  first image. Depending on the image/checkpoint, torchvision attempted an
  additional 4.00 GiB, 162 MiB with only 58.81 MiB free, or 12.26 GiB and
  raised `torch.OutOfMemoryError`.
- Root cause: evaluation passed original high-resolution Fashionpedia tensors
  to Mask R-CNN in batches of two. Although the detector operates on internally
  resized tensors, torchvision postprocessing pastes every predicted float mask
  back to the supplied image dimensions. Images with dozens of predictions and
  very large original dimensions therefore created multi-gigabyte mask tensors.
  Training did not expose this because mask losses operate on cropped proposal
  masks rather than pasted full-resolution inference masks.
- Engineering error: the evaluation memory model and real maximum image
  dimensions should have been tested before the committed run. A training smoke
  test cannot validate inference postprocessing memory.
- Correction: evaluate one image at a time; resize the image, reference masks,
  boxes, and areas together to a maximum side of 1,024 pixels; threshold masks
  to bytes before CPU transfer; and record the evaluation resolution in the
  report. A geometry regression test covers this bounded-resolution path.
- Decision: do not retrain. Attach the preserved notebook 1.2 output to a new
  evaluation-only GPU notebook and evaluate all three saved checkpoints using
  the corrected code and the same frozen validation selection.

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
