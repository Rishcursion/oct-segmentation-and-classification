# OCT Segmentation and Classification

> **Dual-stage deep learning for retinal disease diagnosis from OCT scans. First segment fluid biomarkers, then classify the pathology — so the model has to *show its work* before it gets to make a call.**

This is the codebase for our IEEE paper **"Dual-Stage Deep Learning for Explainable Retinal Disease Diagnosis via OCT Segmentation and Fluid Biomarker Classification"** (Mohan & Sankar, MIT Bengaluru). The PDF lives in this repo as `IEEEPaper.pdf` if you want the full write-up; this README is the practitioner's version.

## The problem

Standard OCT classifiers take a retinal scan in, spit a label out (CNV / DME / Drusen / Normal), and the ophthalmologist is supposed to trust it. That's a hard sell in clinical settings — when the model is wrong, there's no way to ask *why*.

The intuition we tested: clinicians don't diagnose by looking at the whole scan and gestalt-classifying. They look for **specific fluid biomarkers** — intraretinal fluid (IRF), subretinal fluid (SRF), pigment epithelial detachments (PED) — and reason from there. So the model should too.

## The pipeline

```
                        OCT scan
                            │
         ┌──────────────────▼──────────────────┐
         │   Segmentation (DeepLabV3 + R50)    │   Stage 1: find the biomarkers
         │   → IRF / SRF / PED fluid masks     │   95% Dice, 90% Jaccard
         └──────────────────┬──────────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │   Hybrid classifier (ResNet-50      │   Stage 2: classify the disease
         │   on raw scan + fluid masks)        │   94% accuracy, 92% F1
         │   → CNV / DME / Drusen / Normal     │
         └─────────────────────────────────────┘
```

The fluid masks aren't just an output — they're an **input** to the classifier. We feed the segmentation map alongside the raw scan, so the classifier has both the original signal and the model's own "what looks pathological" hint. The classifier then has somewhere to point when it's asked why it picked a label.

## Numbers

| Stage | Metric | Value |
|---|---|---|
| Segmentation | Dice | 95% |
| Segmentation | Jaccard / mIoU | 90% |
| Classification | Accuracy | 94% |
| Classification | Precision | 93% |
| Classification | F1-score | 92% |

Results from the validation split. Per-sample inference outputs (with class probabilities + visualized masks) are in `inference_results/`.

## What's in the repo

```
IEEEPaper.tex / .pdf       The paper, source and build
streamlit.py               Demo UI — upload a scan, see the prediction with mask overlay
test.py                    Headless eval harness, dumps per-sample JSON + viz to inference_results/
notebooks/                 EDA, training-loop scratch, ablation runs
models/
  ├── segmentation_model/  Trained DeepLabV3 weights
  ├── classification_model/ Trained classifier weights
  └── combined_model/      The end-to-end model used for the paper's reported numbers
dataset/scripts/           Data loaders for the OCT2017 / Kermany dataset
inference_results/         Sample predictions for CNV / DME / DRUSEN scans
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchmetrics streamlit numpy pillow tqdm

# Headless evaluation on a sample
python test.py

# Streamlit demo
streamlit run streamlit.py
```

You'll need the OCT2017 dataset (Kermany et al.) under `dataset/` to retrain. The trained checkpoints under `models/` are sufficient to reproduce the inference results without re-training.

## Why I'm proud of this one

The "explainability" framing isn't a marketing layer bolted on after the fact — the architecture is *literally constrained* to produce an interpretable intermediate (the fluid map) before it's allowed to classify. If the segmentation looks wrong to a clinician, that's their cue to discount the classification. That's the kind of model behavior I want to keep building.

Co-authored with [Nived S Mohan](mailto:nivroh2016@gmail.com).
