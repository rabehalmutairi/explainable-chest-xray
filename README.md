# Explainable Multi-Label Chest X-Ray Diagnosis

Multi-label classification of 14 thoracic pathologies from chest radiographs, with Grad-CAM
explanations evaluated against radiologist-drawn bounding boxes rather than assumed to work.

Samsung Innovation Campus AI capstone — Team Core (6 members). The DenseNet-121 and ResNet50
baseline notebooks are teammates' work; everything else in this repository is mine.

> Research prototype. Not intended, validated, or suitable for clinical use.

---

## My contribution

I led the deep-learning workstream. In this repository that means:

- **Final model** — ConvNeXt-Tiny 320x320, trained and selected against the team's DenseNet-121
  and ResNet50 baselines (`notebooks/convnext-tiny-320-training.ipynb`)
- **Loss comparison** — BCE vs focal loss, with the selection reasoning written down
- **Threshold calibration** — per-class decision thresholds tuned on validation only
- **Robustness analysis** — performance across sex, age band, and view position
- **Grad-CAM localization evaluation** — the part most worth reading
  (`docs/gradcam_iobb_summary.md`)
- **Serving layer** — FastAPI inference API and a Streamlit interface (`app/`, `src/`, `streamlit_app/`)

Teammates contributed the dataset construction and patient-level splitting, the DenseNet-121 and
ResNet50 baselines, and their evaluation.

---

## Problem

Chest radiographs are among the most common imaging studies in medicine, and reading them is
bottlenecked by radiologist availability. A model that flags likely pathologies could triage that
queue — but only if a clinician can see *why* it flagged something. A probability alone is not
actionable, and a heatmap that looks plausible is not the same as a heatmap that is correct.

That second problem is what most of this repository is about.

---

## Data

NIH ChestX-ray14, restricted to a 25,895-image classification subset (31,077 images including the
localization splits) covering 11,907 patients.

Splitting is defined **over patients, not images**. One patient can contribute many studies, so
splitting on images lets the network identify the individual instead of the pathology and inflates
every metric.

| Split | Images | Patients | Purpose |
|---|---:|---:|---|
| `train` | 18,013 | 7,930 | Training |
| `val` | 3,978 | 1,699 | Hyperparameters and thresholds |
| `test` | 3,904 | 1,700 | Final classification results |
| `loc_tune` | 2,606 | 289 | Grad-CAM threshold selection |
| `loc_report` | 2,576 | 289 | Final localization results |

All ten pairwise split combinations return zero shared patients. Patients with radiologist-drawn
bounding boxes are withheld from the classification splits entirely and used only for localization,
split evenly so the CAM threshold is never selected on the data it is reported against.

The model has **14 outputs**. `No Finding` is the absence of the fourteen pathologies, not a class.

---

## Model

ConvNeXt-Tiny, ImageNet-pretrained, fully fine-tuned at 320x320.

| | |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Scheduler | `ReduceLROnPlateau` |
| Precision | Mixed (AMP) |
| Epochs | 15 max, early stopping patience 4 — best checkpoint at epoch 3 |
| Augmentation | Resize 320, rotation ±7°, mild brightness/contrast jitter |

No horizontal flip: chest X-rays have a fixed left/right anatomy, and mirroring them teaches the
model that laterality is irrelevant when it is not.

**BCE vs focal loss.** A second run used focal loss (gamma=2, alpha=0.25) and reached a marginally
higher validation macro AUROC — 0.8145 vs 0.8121. BCE was still selected: focal won on the
aggregate by +0.0024 but lost on more individual classes, and class-wise balance mattered more than
a third-decimal macro gain.

---

## Classification results

Evaluated on `test` (3,904 images, 1,700 patients), thresholds tuned on `val` only.

| Metric | Value |
|---|---:|
| Macro AUROC | **0.8158** |
| Macro F1 @ 0.50 | 0.3034 |
| Macro F1 @ tuned thresholds | **0.4236** |
| Macro ECE | 0.0258 |

Validation macro AUROC was 0.8121 against 0.8158 on test — close enough to suggest the
patient-level split is holding rather than leaking. Calibration error was already below 0.05
uncalibrated, so temperature scaling was not applied.

| Pathology | AUROC | Threshold | F1 | Test positives |
|---|---:|---:|---:|---:|
| Hernia | 0.9232 | 0.50 | 0.4390 | 30 |
| Edema | 0.9207 | 0.18 | 0.5560 | 310 |
| Emphysema | 0.9156 | 0.30 | 0.6685 | 380 |
| Cardiomegaly | 0.9039 | 0.67 | 0.5253 | 288 |
| Pneumothorax | 0.8460 | 0.29 | 0.4553 | 375 |
| Effusion | 0.8423 | 0.22 | 0.5682 | 770 |
| Mass | 0.8269 | 0.19 | 0.3981 | 374 |
| Fibrosis | 0.8101 | 0.18 | 0.2643 | 212 |
| Atelectasis | 0.7733 | 0.34 | 0.4564 | 612 |
| Nodule | 0.7580 | 0.20 | 0.3856 | 386 |
| Pleural_Thickening | 0.7565 | 0.23 | 0.2967 | 309 |
| Consolidation | 0.7407 | 0.12 | 0.2983 | 333 |
| Infiltration | 0.7081 | 0.17 | 0.4532 | 838 |
| Pneumonia | 0.6961 | 0.05 | 0.1649 | 163 |

Pneumonia and Infiltration remain the weakest classes — both are diffuse findings, and both are
labels NIH's text-mined ground truth is noisiest on.

### Robustness

Macro AUROC held across sex (F 0.8165 / M 0.8138) and view position (PA 0.8084 / AP 0.7989), and
degraded with age: 0.8249 for 18–39 down to 0.7235 for 80+. The 80+ figure covers only 54 images
across 13 evaluable classes and should not be over-read.

---

## Explainability — and how much of it is real

Grad-CAM heatmaps from the final model, scored against `BBox_List_2017.csv`. The classifier is
loaded read-only and not retrained.

**Target layer.** Two candidates were hooked in a single forward/backward pass and compared on
`loc_tune`. `features[7]` (10x10x768, last conv stage) beat `features[5]` (20x20x384) on mean IoU,
0.2344 vs 0.1711 — the finer grid did not help. Gradients are taken from the **raw logit**, not the
sigmoid, which saturates and flattens exactly the gradients the method depends on.

**Binarization threshold.** A heatmap becomes a box by thresholding it. That threshold cannot be
tuned on IoBB: IoBB divides by ground-truth box area alone, so a bigger prediction can never score
worse, and "maximize IoBB" is won by the loosest possible threshold. The sweep confirms it — mean
IoBB falls monotonically from 0.747 at T=0.05 to 0.034 at T=0.90, with no interior optimum. It was
tuned on mean IoU instead, whose union term punishes oversized boxes and produces a real peak.
Selected: **T\* = 0.15**, a single global threshold. Per-class thresholds were tested and rejected —
they overfit on 24–67 tuning images each.

### Results on `loc_report`

| Pathology | n | mean IoBB | mean IoU | box area | IoBB@0.25 | centre control@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 81 | 0.7330 | 0.5522 | 0.179 | 0.9630 | **0.9877** |
| Mass | 30 | 0.7413 | 0.1621 | 0.109 | 0.8000 | 0.6000 |
| Effusion | 44 | 0.6404 | 0.2193 | 0.154 | 0.7500 | 0.2500 |
| Pneumonia | 53 | 0.5955 | 0.2287 | 0.186 | 0.6981 | 0.6604 |
| Pneumothorax | 33 | 0.6082 | 0.1336 | 0.213 | 0.6970 | 0.4242 |
| Atelectasis | 57 | 0.5934 | 0.1069 | 0.148 | 0.6667 | 0.5439 |
| Nodule | 34 | 0.6296 | 0.0297 | 0.135 | 0.6471 | 0.1765 |
| **Macro** | **332** | **0.6488** | **0.2046** | **0.161** | **0.7460** | **0.5204** |

### Read the controls before quoting any of this

**IoBB alone is not interpretable.** Predicting the entire image every time scores a perfect mean
IoBB of 1.0000 and IoBB@0.25 of 1.0000. Only IoU exposes it, at 0.0842. So every figure above is
reported next to its predicted box area and a control: the same-sized box parked at the image
centre, ignoring the heatmap entirely.

- **Cardiomegaly's 0.963 is free.** The centre control scores 0.988 — better than the model. The
  heart is centred in every chest X-ray, so this number reflects anatomy, not explanation quality,
  and should not be cited as evidence the model localizes.
- **Pneumonia (0.698 vs 0.660) is barely above chance placement.**
- **Effusion (0.750 vs 0.250) and Nodule (0.647 vs 0.176) are the genuine results** — +50 and +47
  points over control.

Macro is 0.746 against a 0.520 control. Real, but far smaller than the headline number suggests.

**Known limits.** The CAM is a 10x10 grid upsampled to 320x320, so small findings are localized
loosely (Nodule mean IoU 0.030) even when the box is hit. Cases exist where the model calls a class
positive while the heat sits on the opposite lung. Per-class n is 30–81, so confidence intervals are
wide. Infiltration cannot be evaluated at all — `BBox_List_2017.csv` contains zero boxes for it.

---

## Serving

A FastAPI service loads the checkpoint once at startup and exposes prediction and Grad-CAM
endpoints; a Streamlit interface consumes them. All model logic lives in `src/` so the API stays
HTTP-only and the frontend never reimplements preprocessing or thresholds. Uploads are held in
memory for the duration of one request and never written to disk.

```bash
pip install -r requirements.txt
export CHEST_XRAY_CHECKPOINT=models/convnext_tiny_320_inference.pth
uvicorn app.main:app --reload          # API  -> :8000
streamlit run streamlit_app/app.py     # UI   -> :8501
```

The checkpoint is not committed (107 MB). Training notebooks run on Kaggle with a GPU.

---

## Repository

    notebooks/
      01_data_preparation.ipynb        Patient-level splitting                 [team]
      02_densenet_baseline.ipynb       DenseNet-121 baseline                   [team]
      03_evaluation.ipynb              Baseline metrics, CIs, thresholds       [team]
      04_resnet50_baseline.ipynb       ResNet50 baseline                       [team]
      05_resnet_evaluation.ipynb       ResNet50 on the same pipeline           [team]
      convnext-tiny-320-training.ipynb Final model: training and selection
      06_convnext_gradcam.ipynb        Grad-CAM + IoBB localization evaluation
    docs/
      convnext_experiment_summary.md   Training, calibration, robustness writeup
      gradcam_iobb_summary.md          Localization protocol, controls, caveats
    src/                               Inference and Grad-CAM (single source of truth)
    app/                               FastAPI service
    streamlit_app/                     Streamlit interface
    results/                           Metrics, thresholds, CAM sweeps, sample overlays

---

## Status

**In progress.** Model training, evaluation, calibration, robustness analysis, Grad-CAM
localization evaluation, and the serving layer are complete and reported above. The capstone
programme runs to September 2026.

## Next

- Per-class probability calibration rather than a single macro ECE check
- Higher-resolution CAMs — the 10x10 grid is the binding constraint on small findings
- Localization for Infiltration, which needs annotations NIH does not provide
