# Explainable Multi-Label Chest X-Ray Diagnosis

Multi-label classification of 14 thoracic pathologies from chest radiographs, with Grad-CAM
explanations checked against radiologist-drawn bounding boxes instead of just assumed to work.

Samsung Innovation Campus AI capstone, Team Core (6 members). The DenseNet-121 and ResNet50
baseline notebooks are teammates' work. Everything else in this repository is mine.

> Research prototype. Not intended, validated, or suitable for clinical use.

---

## My contribution

I led the deep-learning workstream. In this repository that means:

- **Final model**: ConvNeXt-Tiny 320x320, trained and selected against the team's DenseNet-121
  and ResNet50 baselines (`notebooks/convnext-tiny-320-training.ipynb`)
- **Loss comparison**: BCE vs focal loss, with the reasoning behind the final pick written down
- **Threshold calibration**: per-class decision thresholds tuned on validation only
- **Robustness checks**: performance broken down by sex, age band, and view position
- **Grad-CAM localization evaluation**: probably the most interesting part
  (`docs/gradcam_iobb_summary.md`)
- **Serving layer**: FastAPI inference API plus a Streamlit interface (`app/`, `src/`, `streamlit_app/`)

Teammates handled dataset construction, patient-level splitting, and the DenseNet-121 / ResNet50
baselines and their evaluation.

---

## Problem

Chest radiographs are one of the most common imaging studies in medicine, and reading them is
bottlenecked by radiologist availability. A model that flags likely pathologies could help with
that queue, but only if a clinician can actually see why it flagged something. A probability
number on its own isn't actionable, and a heatmap that looks plausible isn't the same thing as a
heatmap that's actually pointing at the right place. Most of this repository is about that second
part.

---

## Data

NIH ChestX-ray14, cut down to a 25,895-image classification subset (31,077 including the
localization splits), covering 11,907 patients.

The splits are defined over patients, not images. A single patient can have several studies, and
if those end up split across train and test the network can just learn to recognize the patient
instead of the pathology, which inflates every metric you'd report.

| Split | Images | Patients | Purpose |
|---|---:|---:|---|
| `train` | 18,013 | 7,930 | Training |
| `val` | 3,978 | 1,699 | Hyperparameters and thresholds |
| `test` | 3,904 | 1,700 | Final classification results |
| `loc_tune` | 2,606 | 289 | Grad-CAM threshold selection |
| `loc_report` | 2,576 | 289 | Final localization results |

Checked all ten pairwise split combinations: zero shared patients. Patients with radiologist-drawn
bounding boxes are held out of the classification splits entirely and used only for localization,
split evenly so the CAM threshold is never picked on the same data it gets reported on.

The model has 14 outputs. `No Finding` is the absence of the fourteen pathologies, not a class of
its own.

---

## Model

ConvNeXt-Tiny, pretrained on ImageNet, fully fine-tuned at 320x320.

| | |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Scheduler | `ReduceLROnPlateau` |
| Precision | Mixed (AMP) |
| Epochs | 15 max, early stopping patience 4, best checkpoint landed at epoch 3 |
| Augmentation | Resize 320, rotation ±7°, mild brightness/contrast jitter |

No horizontal flip. Chest X-rays have a fixed left/right anatomy and mirroring them would teach the
model that laterality doesn't matter, when it does.

I also ran a focal loss variant (gamma=2, alpha=0.25) alongside BCE. It edged out BCE slightly on
validation macro AUROC, 0.8145 vs 0.8121, but I kept BCE as the final model anyway because it did
better on more individual classes and gave a more even result across them. A 0.0024 gain in the
aggregate wasn't worth the class-wise tradeoff.

---

## Classification results

Evaluated on `test` (3,904 images, 1,700 patients), thresholds tuned on `val` only.

| Metric | Value |
|---|---:|
| Macro AUROC | **0.8158** |
| Macro F1 @ 0.50 | 0.3034 |
| Macro F1 @ tuned thresholds | **0.4236** |
| Macro ECE | 0.0258 |

Validation macro AUROC came out to 0.8121, test was 0.8158. Close enough that the patient-level
split seems to be doing its job rather than leaking information. Calibration error was already
under 0.05 without any calibration step, so I didn't bother with temperature scaling.

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

Pneumonia and Infiltration are the weakest classes. Both are diffuse findings, and both are labels
where NIH's text-mined ground truth is noisiest, so that's not entirely surprising.

### Robustness

Macro AUROC held up across sex (F 0.8165 / M 0.8138) and view position (PA 0.8084 / AP 0.7989). It
dropped off with age though: 0.8249 for 18-39 down to 0.7235 for 80+. That last number is only 54
images across 13 evaluable classes, so I wouldn't read too much into it on its own.

---

## Explainability

Grad-CAM heatmaps from the final model, checked against `BBox_List_2017.csv`. The classifier is
loaded read-only here and never retrained.

For the target layer, I hooked two candidates in a single forward/backward pass and compared them
on `loc_tune`: `features[7]` (10x10x768, the last conv stage) beat `features[5]` (20x20x384) on
mean IoU, 0.2344 vs 0.1711. The finer grid didn't actually help. Gradients come from the raw logit
rather than the sigmoid, since the sigmoid saturates near 0 and 1 and flattens out the gradients
Grad-CAM needs.

Turning a heatmap into a box means picking a threshold, and that threshold can't be tuned on IoBB.
IoBB only divides by the ground-truth box area, so a bigger predicted box can never score worse on
it, and "maximize IoBB" just rewards the loosest possible threshold. I confirmed this with a sweep:
mean IoBB falls monotonically from 0.747 at T=0.05 down to 0.034 at T=0.90, no interior optimum
anywhere. So I tuned on mean IoU instead, since its union term actually punishes an oversized box.
That gave a real peak, and the threshold I landed on is T* = 0.15, applied globally. I also tried
per-class thresholds and dropped them: they overfit on the 24-67 tuning images available per class.

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

None of this means much without a control, though. Predicting the entire image every time gets you
a perfect mean IoBB of 1.0000, since IoBB has no penalty for an oversized box. IoU is what actually
catches that (a whole-image guess scores 0.0842 on it), so I'm reporting every IoBB number next to
its box area and a control: the same-sized box just parked at the image center, ignoring the
heatmap completely.

That control is what makes Cardiomegaly's 0.963 mostly meaningless. The centre control alone scores
0.988 on it, better than the model. The heart sits in the middle of basically every chest X-ray, so
that number is measuring anatomy, not whether Grad-CAM found anything. Pneumonia's 0.698 against a
0.660 control is only barely above chance placement too. Effusion (0.750 vs 0.250) and Nodule
(0.647 vs 0.176) are where the model is actually doing something, 50 and 47 points over their
controls respectively. The macro number, 0.746 against a 0.520 control, is real but a lot smaller
than it looks at first glance.

A few limitations worth flagging: the CAM comes out of a 10x10 grid upsampled to 320x320, so small
findings stay coarse even when the box technically overlaps (Nodule's mean IoU is only 0.030). I
also found cases where the model predicts a class correctly but the heatmap sits on the wrong lung
entirely. Per-class n here is only 30-81 images, so the confidence intervals are wide, and
Infiltration can't be evaluated at all since `BBox_List_2017.csv` has no boxes for it.

---

## Serving

A FastAPI service loads the checkpoint once at startup and exposes prediction and Grad-CAM
endpoints. A Streamlit interface sits on top of it. All the model logic lives in `src/`, so the API
stays HTTP-only and the frontend doesn't reimplement preprocessing or thresholds on its own.
Uploads are held in memory for one request and never written to disk.

```bash
pip install -r requirements.txt
export CHEST_XRAY_CHECKPOINT=models/convnext_tiny_320_inference.pth
uvicorn app.main:app --reload          # API  -> :8000
streamlit run streamlit_app/app.py     # UI   -> :8501
```

The checkpoint itself isn't committed (107 MB). Training notebooks run on Kaggle with a GPU.

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

In progress. Training, evaluation, calibration, robustness checks, the Grad-CAM/IoBB evaluation,
and the serving layer are all done and reported above. The capstone program runs through
September 2026.

## Next

- Per-class probability calibration instead of a single macro ECE check
- Higher-resolution CAMs, since the 10x10 grid is the main thing capping small-finding localization
- Localization for Infiltration, which would need annotations NIH doesn't provide
