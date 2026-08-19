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
```

The baseline budget follows the measured 64-image smoke run. Evaluate every
epoch on the exact same validation manifest and select by mask AP, not loss.
