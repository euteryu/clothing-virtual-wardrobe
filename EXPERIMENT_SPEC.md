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
- Full budget will be chosen only after measured smoke runtime and memory.

## Evaluation

- Primary: COCO mask AP on the frozen internal validation partition.
- Diagnostics: AP50/AP75, per-class AP/recall, latency, and qualitative failures.
- Checkpoint selected by validation mask AP.

## Compute plan

- CPU: inspection, validation, manifests, metric aggregation.
- GPU: training and inference on one T4.
- Persist checkpoints, resolved config, category mapping, and run report.

## Stop/continue rule

Accept the baseline if it completes reproducibly and produces non-degenerate
held-out predictions. Continue only for an observed failure such as domain
shift, rare-class recall, boundaries, or latency. Do not tune from final tests.
