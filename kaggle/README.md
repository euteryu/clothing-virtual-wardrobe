# Kaggle stages

Run one version-controlled command per notebook stage:

```text
00 environment and real CUDA computation
01 attached-data discovery and mask validation
02 bounded end-to-end GPU smoke training
03 baseline training (added after measured smoke results)
04 validation/model selection
05 frozen applicability test and error review
```

Stages 03 onward are intentionally gated on actual stage-02 runtime, memory,
loss, and artifact measurements. This avoids guessing a full-session budget.

