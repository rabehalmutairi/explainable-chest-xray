# Grad-CAM and IoBB Localization Summary

## Overview

Grad-CAM explanations for the final ConvNeXt-Tiny 320x320 BCE model, evaluated against the
radiologist-drawn bounding boxes in `BBox_List_2017.csv`.

Notebook: `notebooks/06_convnext_gradcam.ipynb`
Run on Kaggle as `rabehalmutire/02-convnext-grad-cam` (GPU T4).

The classification model is unchanged and was not retrained. The checkpoint is loaded read-only
and verified before use: strict `load_state_dict` with zero missing and zero unexpected keys,
saved epoch 3, `val_macro_auc` 0.8120631850102562.

This is an educational research prototype and is not clinically validated.

---

## Target layer

`torchvision` ConvNeXt-Tiny is `features` (8 stages) -> `avgpool` -> `classifier`. Two candidate
layers were hooked in a **single** forward/backward pass and compared on `loc_tune`:

| Layer | Activation at 320x320 | Best mean IoU on `loc_tune` |
|---|---|---:|
| `features[5]` | 20x20 x 384 | 0.1711 |
| **`features[7]`** | **10x10 x 768** | **0.2344** |

`features[7]`, the last convolutional stage, was selected. The finer 20x20 grid did not help.

Grad-CAM backpropagates from the **raw logit** of one selected pathology, not the sigmoid output:
sigmoid saturates near 0 and 1 and flattens the gradients the method depends on.

---

## Choosing the CAM binarization threshold

A heatmap becomes a box by thresholding it and taking the bounding box of the largest connected
component. That threshold, `T`, is tuned on `loc_tune` only.

**`T` cannot be tuned on IoBB.** IoBB divides by the ground-truth box area alone, so enlarging the
prediction can never lower the score. "Maximize mean IoBB" is therefore won by the loosest possible
threshold regardless of where the heat actually is. The `loc_tune` sweep confirms this: mean IoBB
falls monotonically from 0.747 at `T=0.05` to 0.034 at `T=0.90`, with no interior optimum.

`T` is selected on **mean IoU** instead, whose union term penalizes an oversized box and produces a
real interior peak (0.190 -> **0.234** -> 0.031).

**Selected: `T* = 0.15`**, single global threshold.

Per-class thresholds were tested and **rejected**: macro IoBB@0.25 dropped from 0.746 to 0.638,
overfitting on 24-67 tuning images per class. They scored better on mean IoU (0.240 vs 0.205), the
criterion they were tuned on, by choosing tighter boxes (8% vs 16% of image area) - tighter boxes
help IoU and hurt IoBB. Results kept in `results/gradcam_iobb_results_perclass_T.csv` for reference.

---

## Protocol

- Tuned only on `loc_tune` (336 annotated image-class pairs).
- Reported only on `loc_report` (332 pairs). Verified at runtime: zero shared patients, zero shared images.
- Only the annotated classes are evaluated. **Infiltration is excluded** - `BBox_List_2017.csv`
  contains zero boxes for it.
- An image may carry several boxes for one class; a prediction counts against whichever it matches best.
- Classification and localization metrics are kept separate.

---

## Results on `loc_report`

| Pathology | n | mean IoBB | mean IoU | box area | IoBB@0.1 | IoBB@0.25 | IoBB@0.5 | centre control@0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 81 | 0.7330 | 0.5522 | 0.179 | 1.0000 | 0.9630 | 0.8148 | **0.9877** |
| Mass | 30 | 0.7413 | 0.1621 | 0.109 | 0.8333 | 0.8000 | 0.7333 | 0.6000 |
| Effusion | 44 | 0.6404 | 0.2193 | 0.154 | 0.7955 | 0.7500 | 0.7045 | 0.2500 |
| Pneumonia | 53 | 0.5955 | 0.2287 | 0.186 | 0.7547 | 0.6981 | 0.6038 | 0.6604 |
| Pneumothorax | 33 | 0.6082 | 0.1336 | 0.213 | 0.7273 | 0.6970 | 0.5758 | 0.4242 |
| Atelectasis | 57 | 0.5934 | 0.1069 | 0.148 | 0.6842 | 0.6667 | 0.5965 | 0.5439 |
| Nodule | 34 | 0.6296 | 0.0297 | 0.135 | 0.6471 | 0.6471 | 0.6176 | 0.1765 |
| **MACRO** | **332** | **0.6488** | **0.2046** | **0.161** | **0.7774** | **0.7460** | **0.6638** | **0.5204** |

95% bootstrap CIs (1000 draws, seed 42) on IoBB@0.25:

| Pathology | IoBB@0.25 | 95% CI |
|---|---:|---|
| Cardiomegaly | 0.963 | [0.914, 1.000] |
| Mass | 0.800 | [0.667, 0.933] |
| Effusion | 0.750 | [0.614, 0.864] |
| Pneumonia | 0.698 | [0.566, 0.811] |
| Pneumothorax | 0.697 | [0.545, 0.848] |
| Atelectasis | 0.667 | [0.544, 0.789] |
| Nodule | 0.647 | [0.500, 0.794] |

---

## Controls - read these before quoting any IoBB number

**IoBB alone is not interpretable.** Predicting the *entire image* every time scores mean IoBB
1.0000 and IoBB@0.25 accuracy of 1.0000. Only IoU exposes it, at 0.0842. Every IoBB figure above is
therefore reported next to the predicted box area and a control.

**Centre control**: the same-sized box parked at the image centre, ignoring the heatmap entirely.

- **Cardiomegaly's 0.963 is essentially free.** The centre control scores **0.988** - better than the
  model. The heart is centred in every chest X-ray, so this number reflects anatomy, not explanation
  quality. It should not be cited as evidence the model localizes.
- **Pneumonia (0.698 vs 0.660 control) is barely above chance placement.**
- **Effusion (0.750 vs 0.250) and Nodule (0.647 vs 0.176) are the genuine successes**, +50 and +47
  points over the control.

Macro is 0.746 against a 0.520 control: real, but far smaller than the raw number suggests.

---

## Known weaknesses

- Grad-CAM at `features[7]` is a 10x10 grid upsampled to 320x320, so heat is inherently coarse.
  Small findings (Nodule mean IoU 0.030) are localized loosely even when the box is hit.
- Class-level attention errors exist: cases were observed where the model predicts a class positive
  while the heatmap sits on the opposite lung.
- `loc_report` per-class n is small (30-81), so CIs are wide.
- Infiltration cannot be evaluated at all.

---

## Artifacts

| File | Contents |
|---|---|
| `results/gradcam_iobb_results.csv` | Per-class + macro, global `T*` (primary) |
| `results/gradcam_iobb_results_perclass_T.csv` | Per-class thresholds variant (rejected) |
| `results/gradcam_iobb_per_image.csv` | Per image-class pair: IoBB, IoU, box area, control |
| `results/gradcam_tune_sweep.csv` | Full `loc_tune` sweep, both layers, all thresholds |
| `results/gradcam_iobb_config.json` | Selected layer, `T*`, per-class thresholds, split provenance |
| `results/gradcam_samples/` | Heatmap overlays; `iobb_*` show predicted (red) vs ground truth (green) |

Model checkpoints are not stored in the repository.
