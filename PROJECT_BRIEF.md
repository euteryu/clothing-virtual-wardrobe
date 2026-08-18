# Project brief

## Practical goal

Learn and demonstrate a credible instance-segmentation workflow by converting
an ordinary outfit photo into separate labelled garment masks and transparent
cutouts suitable for a personal digital wardrobe.

## Prediction task

Input: one RGB outfit photograph. Output: one mask, garment category, score,
bounding box, and optional transparent cutout per detected garment.

## Success threshold

- Educational baseline: reproducible held-out mask AP plus visually credible
  masks on ordinary phone photographs.
- Initial usability target: no catastrophic missed main garment in at least 80%
  of a small, frozen personal test set; target to be confirmed after collection.
- Production, commercial, and photorealistic virtual try-on claims are excluded.

## Data

- Source: Fashionpedia / iMaterialist Fashion 2020 FGVC7.
- Labels: COCO-style instance polygons/RLE, apparel category, attributes.
- Annotations/ontology: CC BY 4.0; source images retain original source terms.
- The Kaggle competition supplies labelled train and unlabelled test data. Make
  one deterministic image-level validation partition from train; keep Kaggle
  test outside model selection.
- A later personal-photo set must be consented, private, and untouched during
  model selection.

## Constraints

- No local GPU; free Kaggle GPU only, starting with one T4.
- Git repository contains code/config/reports only.
- Stop after a credible baseline and one evidence-driven improvement.

## Main risks

Domain shift to phone photos, confusing fine-grained classes, small/occluded
items, data-layout ambiguity, class imbalance, and inference cost.

## Out of scope

Polished UI, accounts, hosting, recommendations, ecommerce, and generative
virtual try-on.
