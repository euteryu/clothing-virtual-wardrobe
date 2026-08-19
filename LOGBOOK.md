# Project logbook

## Current status

- Phase: frozen baseline completed its first mixed natural-photo applicability
  test and did not meet the initial usability target.
- Selected inference policy: epoch-03 checkpoint, main garment classes 1-13,
  confidence threshold 0.6, and predicted mask threshold 0.5.
- Current validation operating point: 544 true positives, 258 false positives,
  and 368 false negatives; micro precision 0.6783, recall 0.5965, and F1 0.6348.
- Applicability result: 12 of 20 images avoided a catastrophic required-garment
  miss (60%), below the initial target of 16 of 20 (80%).
- Exact next action: run the predefined epoch-04/05 continuation on one Kaggle
  T4, then evaluate both checkpoints on the same frozen 500-image validation
  subset and compare with epoch 03 by mask AP only.

## Fixed decisions

- Dataset baseline: Fashionpedia / iMaterialist Fashion 2020 FGVC7.
- Task: garment instance segmentation, not virtual try-on.
- Official validation is model-selection data; personal photos form a later
  frozen applicability test.
- No run or check is described as passed until actually executed.

## Run notes

### 2026-08-19 - single improvement experiment predefined

- Evidence: the baseline missed a required garment in 8 of 20 mixed natural
  photos, while frozen in-domain mask AP rose monotonically from 0.1562 to
  0.1825 to 0.2170 across baseline epochs 01-03.
- Predefined question: does conservative continuation improve mask AP on the
  unchanged frozen validation subset without changing training data,
  architecture, or ontology?
- Intervention: resume the exact 4,000-image epoch-03 checkpoint, including its
  optimizer, scheduler, and AMP scaler. Train exactly epochs 04 and 05. The
  restored learning rates are 0.0005 and 0.00005 respectively; batch size 2,
  seed 2026, and all other settings remain fixed.
- Model selection: evaluate epochs 03, 04, and 05 on the identical frozen
  500-image validation subset at maximum side 1,024 and select strictly by mask
  AP. Do not choose using loss, personal photos, or qualitative preference.
- Stop rule: no extra epochs or second intervention after results. After the
  checkpoint and a validation-only confidence threshold are locked, any final
  applicability measurement requires a new independently frozen photo set; the
  already inspected 20-photo set is not reused for selection or a fresh claim.
- Implementation: `kaggle/07_continue_training.py` rejects any source other
  than a 4,000-image epoch-03 checkpoint. Resume training also verifies the
  checkpoint subset size and records the learning rate used for each epoch.

### 2026-08-19 - frozen mixed natural-photo applicability test: target FAIL

- Scope: 20 consented natural photos containing varied framing, lighting,
  distance, single/multiple people, and layered outfits. The set mixes personal
  uploads and internet-sourced images, so it is not a pure personal-phone test.
- Expected garment instances and the catastrophic-miss scoring rule were frozen
  by visual inspection before model inference. One post-run clerical correction
  changed photo 005 from stale labels belonging to the replaced prior image to
  the two tops that had been visually identified before inference; this was a
  manifest transcription correction, not output-driven relabelling.
- Inference used the selected epoch-03 checkpoint on CPU with main garment
  classes 1-13, confidence 0.6, mask threshold 0.5, and maximum side 1,024.
  It completed all 20 images in 151.33 seconds and wrote 20 overlays, 45
  transparent cutouts, and a machine-readable report.
- Result: 12 of 20 images passed the frozen catastrophic-miss rule (60%). The
  initial usability target required at least 16 of 20 (80%), so the target was
  not met. Failed images were 001, 002, 003, 004, 006, 010, 012, and 014.
- Whole garments were missed in a low-quality video frame and a distant subject;
  one three-person image detected only two dresses. Other failures involved a
  merged upper/lower outfit, a dress reduced to a pants-region mask, a largely
  missed light jacket, a missed sweater, and failure to separate a shirt under
  a jacket.
- Successful cases show useful capability: clean top/bottom outfits were
  usually separated, two-person scenes could yield all required instances, and
  several distant or layered examples worked. Boundaries were often credible
  when an instance was found.
- Fine-grained ontology confusion remained frequent: shirts or overshirts were
  labelled jackets, and sweaters or shirts were labelled generic tops. One
  image also produced duplicate overlapping pants/shorts predictions.
- Artifacts remain outside Git under
  `C:\Users\minse\Downloads\wardrobe_personal_results`: frozen expectations,
  raw inference report, scored assessment, overlays, and cutouts. The private
  photos, predictions, and checkpoint are not committed.
- Decision: report this as a failed initial usability target, not a deployment
  success. Do not adjust the checkpoint or confidence threshold using these
  photos. Any improvement must be defined and selected using training/frozen
  validation data only; this applicability set remains untouched final evidence.

### 2026-08-19 - CPU personal-photo inference runner prepared

- Added `kaggle/06_personal_inference.py` and reusable inference code for the
  frozen 20-photo applicability test on a CPU-only Windows computer.
- The runner enforces exactly 20 supported image files by default and preserves
  the selected policy: epoch-03 weights, classes 1-13, confidence threshold
  0.6, mask threshold 0.5, and maximum side 1,024. It does not accept threshold
  overrides from the command line.
- Outputs are written outside the private input-photo directory and comprise a
  labelled overlay per photo, a transparent PNG cutout per detected garment,
  and a machine-readable `personal_inference_report.json`.
- The epoch-03 checkpoint is not present in the local repository or downloaded
  notebook-1.4 review bundle. It must be downloaded separately from the
  preserved notebook-1.2 output and kept outside Git.
- Local verification: `pytest -q` reported 9 passed and `ruff check .` passed.

### 2026-08-19 - qualitative validation review completed; policy frozen

- Reviewed all 12 saved validation triptychs: the six lowest-F1 and six
  highest-F1 eligible images at the selected confidence threshold of 0.6.
  These deliberately selected extremes illustrate failure modes but are not a
  representative estimate of their prevalence.
- Strong cases contained one or two large, prominent garments. Dresses, tops,
  pants, and skirts were localized with close reference/prediction agreement;
  masks generally followed the visible silhouette, including pose and fringe.
- The clearest recurring failure was confusion between visually adjacent
  garment classes: sweater versus top, coat versus jacket, and jumpsuit versus
  pants. Some lowest-F1 examples therefore contained a visually useful mask but
  received no true positive because matching requires both the exact class and
  mask IoU of at least 0.5.
- Layered and occluded outfits were difficult. In one example a coat, sweater,
  and pants reference was reduced to one jacket prediction; an unusual furry
  cape/jacket/dress combination produced no prediction at 0.6.
- Scale and framing were also material. A distant person with several garments
  produced no prediction at 0.6, while a cropped jumpsuit was identified only
  as the visible pants region. These observations agree with the earlier weak
  small-object AP and expose a likely risk for casual phone photos.
- Boundary quality was not the primary issue in the successful examples. The
  dominant visible risks were missed instances and fine-grained ontology
  confusion, especially under layering, unusual silhouettes, distance, or
  truncation.
- Decision: freeze epoch 03, main garment classes 1-13, confidence threshold
  0.6, evaluation maximum side 1,024, and predicted mask threshold 0.5. Do not
  tune these choices on personal photos. Proceed to a separate, frozen personal
  applicability test and report catastrophic main-garment misses explicitly.

### 2026-08-19 - operating-point review PASS

- Notebook: `clothing-virtual-wardrobe-180826-1.4-review`, completed from the
  preserved notebook-1.2 epoch-03 checkpoint at commit `f8aaa38`.
- Evaluation population: the same deterministic 500-image validation subset,
  maximum side 1,024, restricted to the 13 main garment categories. Predicted
  and reference masks were matched at IoU 0.5.
- Result: PASS in 303.36 seconds. The confidence sweep selected 0.6 by maximum
  validation micro-F1. At that threshold there were 544 true positives, 258
  false positives, and 368 false negatives: precision 0.6783, recall 0.5965,
  and F1 0.6348.
- Sweep F1 by confidence threshold: 0.3 = 0.5743, 0.4 = 0.6092, 0.5 =
  0.6161, 0.6 = 0.6348, 0.7 = 0.6277, 0.8 = 0.6055, and 0.9 = 0.5349.
- Artifacts: `outputs/review/operating_point_report.json` and 12 qualitative
  triptychs under `outputs/review/qualitative/`, comprising six lowest-F1 and
  six highest-F1 eligible validation images in input / ground-truth /
  prediction order.
- Interpretation: 0.6 is the fixed operating threshold for this checkpoint and
  class scope. This validation optimization is not personal-photo or deployment
  evidence, and the reported micro metrics use mask IoU 0.5 rather than COCO
  mask AP.
- Decision pending: visually inspect the 12 triptychs and record recurring
  failure themes before freezing the policy and starting the untouched
  personal-photo applicability test.

### 2026-08-19 - operating-point and qualitative review submitted

- Notebook: `clothing-virtual-wardrobe-180826-1.4-review`, submitted with Save &
  Run All; result pending at the time of this entry.
- Inputs: Fashionpedia competition data and the preserved notebook-1.2 output.
  Only the selected epoch-03 checkpoint is used; no training is repeated and
  notebook-1.3 evaluation output is not a dependency.
- Code version: commit `f8aaa38`.
- Fixed evaluation population: the same deterministic 500-image validation
  subset, maximum side 1,024, main garment category IDs 1 through 13 only.
- Purpose: sweep confidence thresholds 0.3 through 0.9, match predicted and
  reference masks at IoU 0.5, select the threshold with maximum validation
  micro-F1, and render six best plus six worst validation triptychs in the order
  input / reference / prediction.
- Expected runtime: approximately 10-20 minutes; allow up to 30 minutes if mask
  matching or Kaggle output persistence is slower. This entry does not claim
  the run passed or that a threshold was selected.
- Required handover: record the full threshold sweep, selected threshold,
  runtime, output counts, and visual failure themes before freezing the
  inference policy or testing personal photos.

### 2026-08-19 - checkpoint comparison and model selection

- All checkpoints were evaluated on the identical frozen 500-image validation
  subset (4,133 reference instances), batch size 1, max side 1,024.
- Epoch 01: mask AP 0.1562, AP50 0.2452, AP75 0.1641, AR100 0.2702;
  27,307 predictions; runtime 334.70 seconds.
- Epoch 02: mask AP 0.1825, AP50 0.2861, AP75 0.1936, AR100 0.2919;
  24,385 predictions; runtime 334.89 seconds.
- Epoch 03: mask AP 0.2170, AP50 0.3305, AP75 0.2328, AR100 0.3425;
  19,337 predictions; runtime 324.15 seconds.
- Decision: select checkpoint epoch 03. It is best on every predefined summary
  metric and emits fewer raw detections. The monotonic held-out improvement does
  not justify extra training by itself because three epochs were the fixed
  budget; further training would be a new experiment requiring a predefined
  question.
- Exact next action: keep model weights fixed and use validation data to choose
  an operating confidence threshold for the 13 main garment categories. Render
  representative best and worst validation examples at that fixed threshold,
  then freeze the inference policy before testing personal phone photos.

### 2026-08-19 - corrected 500-image evaluation epoch 03 PASS

- Notebook: `clothing-virtual-wardrobe-180826-1.3-evaluation`, committed run;
  total notebook runtime reported as approximately 24 minutes.
- Checkpoint: epoch 03 from the preserved notebook-1.2 output.
- Evaluation population: the frozen 500-image validation subset at maximum side
  1,024 and batch size 1. It contained 4,133 reference instances.
- Result: PASS without OOM in 324.15 seconds. The model emitted 19,337 ranked
  predictions. Mask AP was 0.2170, AP50 0.3305, AP75 0.2328, and AR100 0.3425.
  Size-specific AP was 0.098 small, 0.191 medium, and 0.248 large.
- App-relevant category examples: pants AP 0.7334, dress 0.7194, hat 0.6372,
  sleeve 0.6010, glasses 0.5452, tie 0.5229, shoe 0.5107, skirt 0.4891,
  shorts 0.4878, jacket 0.4824, and top/t-shirt/sweatshirt 0.4644.
- Weak or zero categories were concentrated in rare garments, accessories, and
  decorative parts, including vest, cape, leg warmer, umbrella, epaulette,
  fringe, ribbon, sequin, and tassel. Small-object AP was also substantially
  below large-object AP.
- Interpretation: this is an in-domain validation result, not an OOD test. It is
  a credible bounded-data educational baseline but not evidence of reliable
  deployment across all 46 categories. The prediction/reference ratio of 4.68
  indicates that confidence-threshold and app-category filtering must be chosen
  on validation data before presenting uncluttered wardrobe output. COCO AP
  itself remains valid because predictions are score-ranked.
- Decision pending: compare epoch-01 and epoch-02 summary metrics against epoch
  03 before selecting a checkpoint. Do not choose by training loss alone. After
  selection, run qualitative error review and threshold analysis focused on the
  main wardrobe garment classes, followed by a separate personal-photo domain
  test.

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
