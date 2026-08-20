# Project logbook

## Research direction and essential reading

- Product thesis: garment segmentation is not the end product. Its value must be
  demonstrated through a downstream outcome such as reliable garment selection
  from uncontrolled photos, reusable wardrobe-asset creation, improved virtual
  try-on fidelity/control, search and outfit retrieval, or user-correctable
  failure handling. Higher mask AP alone is insufficient evidence of usefulness.
- Most promising project question: for messy person-to-person references with
  multiple people, garments, layering, occlusion, and casual framing, does an
  explicit, user-correctable instance-selection and garment-reconstruction stage
  improve final virtual try-on success over direct or built-in preprocessing?
  Evaluate the final images, not merely intermediate masks.
- Core ablation: compare direct reference VTON, VTON using its native parser,
  our selected instance/cutout, and our selection followed by virtual try-off
  reconstruction. Freeze the person/garment pairs and model settings. Measure
  correct-garment transfer, logo/texture/detail preservation, identity and
  non-target clothing preservation, occlusion/layering failures, artifacts,
  latency, and human preference; also record segmentation-to-output causal
  failures and the benefit of a simple user mask correction.
- Current public gaps that make this worthwhile are robust in-the-wild and
  person-to-person operation; selecting one item from multi-person, layered, or
  multi-garment references; exact high-frequency garment fidelity; controllable
  preservation of the person's identity/body/non-target clothing; full outfits,
  accessories, sizing and physically truthful fit; reliable evaluation beyond
  paired reconstruction metrics; uncertainty/failure detection; and practical
  privacy, latency, licensing, and reproducibility.
- Essential reading, in priority order: MGT / *Enhancing Person-to-Person
  Virtual Try-On with Multi-Garment Virtual Try-Off* (2025); TryOffDiff /
  *Virtual Try-Off via High-Fidelity Garment Reconstruction Using Diffusion
  Models* (2025); RefTon / *Reference Person Shot Assist Virtual Try-On* (CVPR
  2026); TryOnDiffusion / *A Tale of Two UNets* (CVPR 2023); IDM-VTON (ECCV
  2024); CatVTON (ICLR 2025); UP-VTON (ICCVW 2025); BooW-VTON (CVPR 2025);
  Street TryOn (WACV 2025); VTBench (2025); DualFit (ICCVW 2025); Dress Code
  (2022); and Fashionpedia (ECCV 2020).
- Novelty boundary: garment extraction before VTON, mask-versus-mask-free VTON,
  and try-off reconstruction already exist. A credible contribution would need
  a sharply specified unresolved setting, dataset/evaluation protocol, or
  demonstrated integration benefit. Do not claim novelty merely for connecting
  segmentation to a pretrained VTON model.

## Product modes and human interaction boundary

- Consumer VTON should be zero-click in the normal case: the user uploads one
  target-person photo and one desired-clothing reference, and the pipeline
  automatically selects/preprocesses the garment and generates the try-on.
  Mask selection or repair is an exceptional recovery path, not a routine
  requirement imposed on ordinary users.
- Dataset creation is a separate researcher workflow. For new custom labelled
  images, run the current model or a promptable segmenter to pre-annotate, then
  review and correct masks/categories in CVAT. Use SAM 2 click/box prompting for
  rapid mask refinement and polygon/brush editing only for difficult boundaries.
  Export a versioned COCO instance-segmentation dataset, run annotation QA, and
  create image-level train/validation/test splits before any model fitting.
- Corrections made while using the consumer product must not silently become
  training data. They require explicit consent, de-identification/retention
  rules, dataset versioning, and a new predefined experiment and split.
- A reception-camera application is a distinct still-image compliance product,
  not VTON and not necessarily video learning. A guest can pause at a marked
  position for a consented still photograph; person/garment detection,
  segmentation, attributes, and a configurable rule engine then produce green
  (confidently compliant), amber (uncertain/occluded; human review), red
  (confident rule conflict; human-confirmed response), or unable-to-assess.
- Prefer assisted decision support over autonomous admission denial. Do not use
  face recognition; anonymous short-lived visit/session IDs are sufficient.
  Production validation must cover the actual entrance camera, distance,
  lighting, poses, garments, demographic variation, uncertainty calibration,
  accessibility, privacy/signage, security, retention, and human override.
- The present model is only one component of reception compliance. It still
  needs garment attributes/colour, person-to-garment association, calibrated
  rule-level confidence, and domain-specific evaluation. A controlled still
  capture deliberately avoids an unnecessary video-training project.

## Current status

- Phase: segmentation training and validation operating-policy selection are
  complete. Epoch 05 is the final application model at confidence 0.6 for main
  garment classes 1-13 and predicted-mask threshold 0.5.
- Epoch 04 remains the nominal global COCO-mask-AP winner, but epochs 04 and 05
  were practically tied on that metric. Stage 08 resolved the application choice
  using main-garment validation micro F1: 0.651869 for epoch 05 versus 0.647773
  for epoch 04.
- Previous epoch-03 validation operating point: 544 true positives, 258 false
  positives, and 368 false negatives; micro precision 0.6783, recall 0.5965,
  and F1 0.6348 at confidence 0.6. This remains historical, not the epoch-04
  policy.
- Applicability result: 12 of 20 images avoided a catastrophic required-garment
  miss (60%), below the initial target of 16 of 20 (80%).
- Exact next action: test the locked epoch-05 policy once on a new independently
  frozen consented applicability set. Do not reuse the previously inspected 20
  images for model selection or a fresh performance claim.

## Fixed decisions

- Dataset baseline: Fashionpedia / iMaterialist Fashion 2020 FGVC7.
- Task: garment instance segmentation, not virtual try-on.
- Official validation is model-selection data; personal photos form a later
  frozen applicability test.
- No run or check is described as passed until actually executed.
- Retrospective hyperparameter rationale: batch size 2 meant two images per
  optimizer update and was chosen as a conservative, conventional fit for
  high-resolution Mask R-CNN training on one 16 GB T4 after the real GPU smoke
  test. It was not established as globally optimal. A batch of thousands would
  exceed both the 4,000-image subset and GPU memory; instance segmentation must
  retain large feature maps, proposals, and masks. The SGD learning rate,
  momentum 0.9, and weight decay 0.0005 came from standard torchvision Mask
  R-CNN transfer-learning practice with adjustment for the small batch. The
  smooth loss decline and initial validation-AP gains demonstrate stability,
  not proof that the hyperparameters were optimal.
- Class balance means primarily unequal labelled instances across categories:
  common tops, pants, dresses, and shoes provide much more learning signal than
  rare vest, cape, jumpsuit, or decorative-part classes. Class-aware sampling,
  loss weighting, more rare examples, or a product-specific merged ontology are
  possible remedies. Foreground/background and scale imbalance are distinct:
  distant or small garments occupy few pixels and may need crops, scale
  augmentation, higher resolution, and representative small-object examples.
- Training loss is a composite optimization signal for classification, boxes,
  masks, proposals, and objectness on training images. Its theoretical lower
  bound is approximately zero but its numeric scale is not a percentage and is
  not a direct quality measure. Validation AP measures unseen ranked detection,
  classification, and mask overlap. Epoch 05's lower training loss alongside
  flat validation AP indicates saturation under this data/schedule, not that
  epoch 05 is necessarily worse or that hundreds of unchanged epochs are useful.
- Product-architecture clarification: the current Mask R-CNN is an instance-
  segmentation component for locating, classifying, masking, and extracting
  garments. It cannot synthesize a photo of a person wearing a different
  garment. The stated end goal of person photo + desired garment/outfit photo
  -> try-on image requires a separate generative virtual-try-on model, normally
  a large pretrained diffusion-based system. Segmentation may remain useful for
  wardrobe ingestion, garment conditioning, preprocessing, and diagnostics,
  but it is supporting infrastructure rather than the core try-on generator.
- Do not expand this bounded experiment into training a try-on generator from
  scratch on one free T4. Complete the predefined epoch-04/05 segmentation
  improvement for its educational evidence, then evaluate pretrained virtual-
  try-on integration as a distinct project phase with its own data, metrics,
  privacy analysis, and compute plan.
- Pretrained VTON integration plan: begin with a public model that accepts both
  person images and flat-lay or model-worn garment references. Compare two
  fixed inputs for the same person/garment pairs: (A) the original garment
  reference and (B) our segmenter's selected/cropped garment asset. Judge the
  final try-on outputs for garment-detail preservation, person identity, pose,
  artifacts, and usability. This is an end-to-end preprocessing ablation, not
  necessarily a direct contest between two segmentation masks: a maskless VTON
  model may condition on the full garment reference through learned features or
  attention without exposing or even computing an equivalent garment mask.
- Expected role of our segmenter in that pipeline: select a requested garment
  from crowded, multi-person, or multi-garment source photos; assign a practical
  category; create a reusable transparent wardrobe asset; and provide explicit
  diagnostics or user correction. For a clean single-garment product image,
  segmentation may be unnecessary or harmful if it removes useful context or
  garment detail, so routing should be conditional rather than mandatory.
- Verified FASHN VTON v1.5 preprocessing detail: for a model-worn garment
  reference, its public pipeline runs DWPose and its 18-class SegFormer human
  parser on both the target-person and garment-reference images, then calls
  `create_garment_image` with the garment parse. For a flat-lay reference it
  disables garment masking and uses dummy garment pose. Therefore the cleanest
  ablation is not merely raw photo versus a standalone transparent PNG: preserve
  FASHN's target-person preprocessing and garment-pose path, but compare its
  garment-parser mask with our selected instance mask when constructing the
  garment-conditioning image. This tests whether explicit instance selection
  helps, especially in multi-person, multi-garment, or layered references.
- Prior-art clarification: external garment extraction and mask/mask-free VTON
  comparisons are established research topics, so this project should not claim
  novelty for that idea. Closest work includes a 2022 VTON clothing-extraction
  module, TryOnDiffusion's person-to-person formulation, UP-VTON's unified
  mask/mask-free guidance, and TryOffDiff/MGT's reconstruction of standardized
  garments from clothed people before VTON. MGT explicitly reports using its
  reconstructed garments with a VTON model for person-to-person try-on.
- Revised practical pipeline candidate: our instance segmenter selects the
  requested garment and category from an uncontrolled source photo; a pretrained
  virtual-try-off model may then reconstruct a canonical garment image; a
  pretrained VTON model renders it on the target person. Compare this against
  direct person-reference VTON and FASHN's built-in parser. Our contribution is
  a reproducible, user-correctable integration and evaluation for messy source
  photos, not a new VTON architecture. TryOffDiff/MGT is suitable for academic
  prototyping but its SSPL release is not a straightforward commercial license.

## Run notes

### 2026-08-20 - CVAT human correction started

- CVAT Free/Solo project and task were created successfully with 20 images and
  the 13 main-garment labels. The reliable import route was to create the task
  from `wardrobe_20_images.zip`, then upload the COCO proposal JSON separately;
  project-level dataset import had failed with `ValueError: No media data found`.
- Frame 0 / `photo_001.png` is complete and saved. The imported epoch-05 pants
  proposal was reviewed/repaired, and the missed black upper garment was added
  manually as a sweater mask. The final frame contains two instances: pants and
  sweater. Screenshot evidence is retained privately as
  `C:\Users\minse\Downloads\cvat3.png`.
- Annotation convention: create filled masks for all visible garment pixels,
  exclude genuine occluders such as hair, hands, skin, shoes, and background,
  and do not hallucinate invisible fabric beneath occlusion. Prioritize correct
  instance/category and materially accurate coverage over individual ambiguous
  boundary pixels.
- CVAT mask-brush editing was noticeably laggy while the surrounding site stayed
  responsive, consistent with browser-side rendering and recomputation of dense
  1,024-pixel masks. Lock completed objects, use broad tools for interiors and a
  small brush only near boundaries, save between frames, and avoid unnecessary
  pixel-level polishing.
- Resume point: open the existing job, confirm frame 0 is saved, click the single
  right-chevron beside Play to open frame 1 / `photo_002.png`, then continue the
  same review-correct-add-save procedure through frames 1-19.

### 2026-08-19 - 20-photo domain-annotation package prepared

- User-directed scope change: repurpose the already inspected 20 mixed natural
  photos as a small domain-adaptation dataset rather than retaining them as an
  external test. The historical epoch-03 applicability result remains valid as
  a record of what occurred, but these images can no longer support a future
  unseen-test claim.
- Added stage 09 to create CVAT-compatible, recall-oriented COCO preannotations
  from the locked epoch-05 model. Proposals use confidence 0.3 so annotators can
  delete false positives while seeing more plausible low-confidence garments.
  Training on uncorrected proposals is prohibited because it would reproduce the
  model's known omissions and errors.
- Local stage-09 run PASS on CPU in 83.76 seconds: 20 resized images, 59 proposed
  main-garment instances, 13 categories, maximum side 1,024. The COCO file loaded
  successfully with pycocotools and every referenced image exists.
- Private outputs remain outside Git:
  `C:\Users\minse\Downloads\wardrobe_domain_annotations\` and import archive
  `C:\Users\minse\Downloads\wardrobe_domain_annotations.zip` (7,946,731 bytes).
- Required human step: import the images and COCO preannotations into CVAT;
  inspect every image; delete false instances; correct categories and mask
  boundaries; add every missed garment; then export corrected COCO 1.0. Only
  that corrected export is eligible for supervised domain adaptation.
- Evaluation constraint: because the same 20 images become development/training
  material, a final all-20 model has no independent natural-photo test here. Use
  image-level cross-validation for internal evidence and retain frozen
  Fashionpedia validation evaluation as a catastrophic-forgetting guard.
- Annotation-tool decision: CVAT Free/Solo is sufficient for this 20-image job.
  One project, one task, less than 1 GB, annotation-only export, manual review,
  and the listed 100 monthly AI calls all cover the requirement. Images are
  already retained locally, so paid export-with-images is unnecessary. CVAT's
  documented COCO import/export supports masks and RLE. Prefer local self-hosted
  CVAT only if uploading private images to CVAT Online is unacceptable; Label
  Studio Community Edition is a viable local alternative but would require
  adapting the prepared preannotation import workflow.
- Cross-validation rationale: the epoch-03 model achieved only 12/20 on the
  earlier coarse catastrophic-miss assessment, demonstrating an applicability
  problem on these mixed natural images, although epoch 05 itself has not been
  independently tested on them. Once the 20 images become adaptation data,
  evaluating a model on the same images it learned would mostly measure
  memorization. Five image-level folds train on 16 and assess the remaining four,
  rotating until each image has one out-of-fold prediction. This is limited
  internal evidence, not a substitute for a new external test.
- Report-artifact audit: notebook-1.5 `results.zip` is downloaded, but the full
  stage-08 finalization output is not currently present in Downloads. Its pasted
  final policy is recorded, but the archive containing the complete epoch-04/05
  confidence reports and qualitative triptychs should be downloaded separately
  as `results_stage08.zip` for the final PDF evidence package.

### 2026-08-19 - final segmentation operating policy PASS

- Stage 08 evaluated epochs 04 and 05 on the same frozen 500-image validation
  set and optimized confidence independently for main garment classes 1-13.
  Both checkpoints selected confidence 0.6.
- Epoch 04 at 0.6: 560 true positives, 257 false positives, 352 false negatives;
  precision 0.685435, recall 0.614035, and micro F1 0.647773.
- Epoch 05 at 0.6: 558 true positives, 242 false positives, 354 false negatives;
  precision 0.697500, recall 0.611842, and micro F1 0.651869.
- Decision: select epoch 05 at confidence 0.6. Relative to epoch 04 it gives up
  two true positives but removes 15 false positives, producing the better F1
  and a cleaner application output. This operational result also confirms that
  the tiny global-AP difference did not establish epoch 04 as the better product
  checkpoint.
- Final segmentation policy: epoch 05; classes 1-13; confidence 0.6; mask
  probability threshold 0.5; maximum image side 1,024. Training and validation
  selection are now closed.
- Local active checkpoint:
  `C:\Users\minse\Downloads\wardrobe_model\checkpoint_epoch_05.pt`,
  368,543,911 bytes, SHA-256
  `2842DEA6CC6D5D372AC5219CE70C604665EE82BBE17711C39752AB07C5490FF4`.
  CPU load verified epoch 5 with 432 model-state tensors. The model remains
  outside Git.

### 2026-08-19 - notebook 1.5 artifacts secured locally

- Source archive: `C:\Users\minse\Downloads\results.zip`, retained unchanged as
  the original downloaded notebook-output bundle (778,618,704 bytes).
- Extracted the selected epoch-04 checkpoint, training report and manifest, both
  evaluation reports, and frozen validation manifest under
  `C:\Users\minse\Downloads\wardrobe_model\notebook_1_5\`. Copied the active
  model to `C:\Users\minse\Downloads\wardrobe_model\checkpoint_epoch_04.pt`.
- Active checkpoint verification: 368,543,911 bytes; SHA-256
  `2EAB060EE51CAADBF20B2FCBFDCC9702B719ACB13FFE4FD5EB7E4FD12F292FE3`;
  PyTorch CPU load succeeded, reported epoch 4, and contained 432 model-state
  tensors. Weights and reports remain outside Git as required.
- Training provenance from the downloaded report: PASS on a Tesla T4 using the
  fixed 4,000 images and batch size 2. Epochs 04 and 05 took 3,357.8 seconds in
  total; mean training loss was 0.670704 and 0.620183 respectively. Lower epoch-
  05 training loss did not override its slightly worse frozen validation AP.

### 2026-08-19 - single improvement evaluation PASS; epoch 04 selected

- Epochs 04 and 05 were evaluated on the same deterministic 500-image validation
  subset containing 4,133 references, with batch size 1 and maximum side 1,024.
- Epoch 04: mask AP 0.222325, AP50 0.339174, AP75 0.235880, AR100 0.345195;
  17,670 predictions; runtime 327.88 seconds. Size AP was 0.100 small, 0.195
  medium, and 0.256 large.
- Epoch 05: mask AP 0.222213, AP50 0.338248, AP75 0.235111, AR100 0.345974;
  17,289 predictions; runtime 333.29 seconds. Size AP was 0.108 small, 0.194
  medium, and 0.255 large.
- Baseline epoch 03 had mask AP 0.2170, AP50 0.3305, AP75 0.2328, AR100
  0.3425, and 19,337 predictions on the identical evaluation population.
- Primary-metric record: epoch 04 is the nominal winner and gains approximately
  0.0053 absolute mask AP (2.45% relative) over epoch 03. Epoch 05 is lower by
  only 0.000112 AP while having slightly higher AR100, better small-object AP,
  fewer predictions, and lower training loss. Engineering interpretation: the
  two final checkpoints are practically tied, so global AP alone does not show
  which users would experience as better.
- Interpretation: the intervention produced a real but small held-out gain and
  then plateaued. It does not demonstrate that the earlier 60% applicability
  result has improved, and it does not justify additional epochs or another
  intervention. The old epoch-03 confidence threshold is not automatically
  valid for epoch 04.
- Required continuation: compare both checkpoints after independently choosing
  their main-garment validation thresholds, then freeze the checkpoint/threshold
  pair with the best validation micro F1. Do not use personal photos or training
  loss for this final application-policy comparison.

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
