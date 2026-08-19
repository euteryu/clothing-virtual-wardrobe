# Kaggle stages

Run one version-controlled command per notebook stage:

```text
00 environment and real CUDA computation
01 attached-data discovery and mask validation
02 bounded end-to-end GPU smoke training
03 resumable 4,000-image, 3-epoch baseline training
04 frozen 500-image COCO mask evaluation for every epoch checkpoint
05 frozen applicability test and error review
06 local frozen-photo inference
07 single improvement: resume epoch 03 through epoch 05
08 finalization: compare epoch 04/05 validation operating points and freeze policy
09 prepare private domain images and model-assisted COCO masks for CVAT correction
```

The baseline budget follows the measured 64-image smoke run. Evaluate every
epoch on the exact same validation manifest and select by mask AP, not loss.

Stage 08 acknowledges that epochs 04 and 05 are practically tied in global COCO
mask AP. It sweeps the application confidence threshold for both checkpoints on
main garment classes 1-13, then selects the deployable checkpoint/threshold pair
by validation micro F1. This is the final segmentation operating-policy step.

Stage 09 does not train on predictions. It creates recall-oriented epoch-05
preannotations at confidence 0.3, resized source images, and a COCO archive for
human correction in CVAT. Every mask/category must be reviewed and every missed
garment added before the resulting annotations are eligible for fine-tuning.
