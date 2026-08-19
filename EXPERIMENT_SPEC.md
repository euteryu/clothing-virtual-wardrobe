# Experiment specification

## Question

Can a pretrained Mask R-CNN learn useful garment instance masks from a bounded
Fashionpedia subset within a free Kaggle T4 session?

## Data split

- Split labelled Fashionpedia train deterministically at image level: 90% fit,
  10% validation, seed 2026. Freeze and persist the resulting IDs.
- Smoke subsets are deterministic by sorted image ID and seed 2026.
- Never use competition test annotations or personal test photos for tuning.

## Baseline

- Model: torchvision Mask R-CNN ResNet-50 FPN v2, COCO pretrained.
- Replace classification/mask heads for Fashionpedia categories.
- Optimizer: SGD, lr 0.005 adjusted by batch size, momentum 0.9, decay 0.0005.
- Smoke budget: 64 images, one epoch, batch size 2.
- First baseline budget: deterministic class-coverage subset of 4,000 training
  images, three epochs, batch size 2. Save after every epoch.
- Full budget will be chosen only after measured smoke runtime and memory.

## Evaluation

- Primary: COCO mask AP on the frozen internal validation partition.
- Diagnostics: AP50/AP75, per-class AP/recall, latency, and qualitative failures.
- Checkpoint selected by validation mask AP.

## Compute plan

- CPU: inspection, validation, manifests, metric aggregation.
- GPU: training and inference on one T4.
- Persist checkpoints, resolved config, category mapping, and run report.
- Evaluate each epoch checkpoint on the same 500-image validation subset and
  select the highest mask AP.

## Stop/continue rule

Accept the baseline if it completes reproducibly and produces non-degenerate
held-out predictions. Continue only for an observed failure such as domain
shift, rare-class recall, boundaries, or latency. Do not tune from final tests.

## Single improvement experiment

- Motivation: the frozen mixed natural-photo applicability test missed required
  garments in 8 of 20 images, particularly upper garments, layered clothing,
  distant subjects, and a multi-person dress instance. In-domain validation AP
  had also improved monotonically through the fixed third baseline epoch.
- Question: does conservative continuation of the same model on the exact same
  4,000-image training subset improve frozen validation mask AP without changing
  the data, architecture, or ontology?
- Intervention: resume the epoch-03 optimizer, scheduler, scaler, and model;
  train epochs 04 and 05 only. The restored schedule uses learning rate 0.0005
  for epoch 04 and 0.00005 for epoch 05. All other baseline settings remain
  unchanged.
- Selection: evaluate epochs 03, 04, and 05 on the identical frozen 500-image
  validation subset at maximum side 1,024. Select strictly by mask AP; AP50,
  AP75, AR100, per-class results, runtime, and prediction count are diagnostics.
- Stop: this is the one allowed evidence-driven improvement. Do not add epochs,
  change thresholds, or try another intervention after seeing its results.
- Final applicability: never train or select using the existing 20-photo mixed
  natural set. Because it has already been inspected against the baseline, use
  a new independently frozen consented photo set only after the improved model
  and validation operating threshold are locked.

## Final operating-policy comparison

- Epochs 04 and 05 are practically tied in global COCO mask AP: 0.222325 versus
  0.222213. Retain epoch 04 as the nominal primary-metric winner, but do not
  claim that this 0.000112 difference establishes meaningful superiority.
- For the actual main-garment application, sweep confidence independently for
  both checkpoints on the same frozen validation images and classes 1-13.
  Select the checkpoint/threshold pair by maximum validation micro F1, with
  recall, precision, and then later epoch as deterministic tie-breakers.
- This finalization does not add training or inspect the old personal set. It
  converts two statistically unresolved checkpoints into one reproducible
  operating policy suitable for a new external applicability test.
