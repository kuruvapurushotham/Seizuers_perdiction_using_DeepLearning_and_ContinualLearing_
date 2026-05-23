# Release Notes – v4.0

Current state of the Real-Time Seizure Prediction System.

## Overview

This version focuses on:
- Multiple deep learning architectures for EEG seizure prediction
- Upgraded preprocessing
- Flexible training pipeline
- Continual learning via Elastic Weight Consolidation (EWC)
- Streamlit dashboard + FastAPI server
- Python 3.11 compatibility

---

## New Features

### 1. Multiple Model Architectures
- Added `create_model` factory in `models/network.py`
- Implemented architectures:
  - `eegnet` – compact 1D CNN baseline
  - `googlenet` – Inception-style 1D network
  - `densenet` – simplified DenseNet-style 1D CNN
  - `vgg` – VGG-style deep 1D CNN
  - `resnet` – residual 1D CNN
  - `rnn` – CNN + recurrent layer hybrid
  - `deepcnn` – deeper pure 1D CNN

### 2. Training Pipeline Upgrades (`training/train.py`)
- Model selection via `--model {eegnet,googlenet,densenet,vgg,resnet,rnn,deepcnn}`
- Hyperparameters:
  - Optimizers: `adam`, `sgd`, `adamw`
  - Schedulers: `plateau`, `cosine`, `none`
  - Weight decay, gradient clipping, early stopping (patience)
- Saves best checkpoints to `models/checkpoints/{model_name}_best.pt`

### 3. Continual Learning – EWC (`models/ewc.py`)
- `EWC` class for Fisher Information & penalty
- `EWCTrainer` for sequential task training
- Support for saving EWC-aware models as `{model_name}_ewc_best.pt`

### 4. Evaluation Utilities (`evaluation/evaluate_models.py`)
- Modes: `single`, `compare`, `cv`, `cl`
- Metrics:
  - Accuracy, precision, recall, F1, AUC-ROC
  - Sensitivity, specificity
  - Confusion matrices and plots
- Simple cross-validation support

### 5. Streamlit Dashboard (`app/app.py`)
- Modern dashboard for offline EEG file analysis
- Features:
  - File upload (EDF/EEG/CNT/VHDR/NPZ)
  - Preprocessing + heuristic detector
  - Deep learning detector with selectable architecture
  - Continual learning (EWC) mode
  - Probability timelines, top suspicious windows, spectrograms
  - CSV + PDF report export

### 6. FastAPI Server (`main.py`, `src/api/server.py`)
- `main.py --mode api` starts FastAPI server
- `src.api.server.create_app` builds the app
- `/predict` endpoint for programmatic access

### 7. Preprocessing Improvements (`app/preprocess.py`)
- Hybrid filters and SNR-based selection (targeted for seizure-like patterns)
- Consistent windowing suitable for all models

### 8. Python 3.11 Compatibility
- Updated `requirements.txt` to work on Python 3.11
- Resolved issues with:
  - `torch`
  - `pyedflib`
  - `sphinx` and other tooling

---

## Breaking / Important Changes

- Models are now always created through `create_model` instead of direct class imports.
- Checkpoints store `model_kwargs` including:
  - `model_name`
  - `in_channels`
  - `num_classes`
- The Streamlit app and training code **infer channel count from data**, so old checkpoints with mismatched channels may not load cleanly.
- DenseNet implementation was simplified to remove BatchNorm channel mismatch issues.

---

## Known Limitations

- Demo checkpoints created during setup are randomly initialized (no real training) and intended only for integration testing.
- EWC features are implemented but require a proper experimental pipeline (datasets/tasks) for meaningful CL experiments.
- Streamlit UI still logs future deprecation warnings about `use_container_width` (layout-related, not functional).

---

## Files Touched in v4.0

- **Core**
  - `models/network.py` – new architectures + `create_model`
  - `training/train.py` – flexible training loop
  - `models/ewc.py` – EWC + `EWCTrainer`
  - `evaluation/evaluate_models.py` – evaluation CLI
- **UI / API**
  - `app/app.py` – Streamlit dashboard
  - `src/api/server.py` – API server setup
  - `main.py` – entrypoint for modes (download/train/realtime/api/dashboard/evaluate)
- **Config & Env**
  - `config.yaml` – unified config
  - `requirements.txt` – Python 3.11 support

---

## Version Tag

This documentation refers to the **current working version (v4.0)** as represented by:
- New model architectures
- EWC support
- Updated training/evaluation scripts
- Working Streamlit + FastAPI paths on Python 3.11.
