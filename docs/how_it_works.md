# How the System Works (High-Level Overview)

This document explains the end-to-end flow of the Real-Time Seizure Prediction System in its current version.

## 1. Data and Preprocessing

### Input Data
- EEG recordings from clinical datasets (e.g. CHB-MIT)
- Typical formats: EDF/EEG/CNT/VHDR, plus internal NPZ window files

### Loading
- **Training / evaluation**: `training/train.py` uses `EEGDataset` + `load_recording` from `app/io_utils.py`.
- **Streamlit app**: `app/app.py` uses `load_recording` directly for uploaded files.

### Preprocessing (`app/preprocess.py`)
Key steps (simplified):
1. **Bandpass / Notch filtering** to remove artifacts and line noise
2. **Standardization / normalization** per channel
3. **Windowing**
   - Splits continuous EEG into overlapping / fixed-length windows
   - Produces 3D tensor: `(num_windows, channels, samples)`
4. **Heuristic features**
   - Functions like `simple_heuristic_score` compute a classical detector score
   - Used both for quick threshold-based detection and as a reference timeline

All deep learning models consume the same standardized window representation.

---

## 2. Model Architectures (`models/network.py`)

Models are implemented as PyTorch `nn.Module` classes and created via:

```python
from models.network import create_model

model = create_model(name, in_channels, num_classes)
```

Available models:
- `eegnet`, `googlenet`, `densenet`, `vgg`, `resnet`, `rnn`, `deepcnn`

### Factory Design
- Centralizes model creation logic
- Makes it easy to select architectures from CLI/Streamlit
- Encapsulates architecture details inside `network.py`

---

## 3. Training Pipeline (`training/train.py`)

### Dataset
- `EEGDataset`:
  - Reads a CSV of `{file, label}` pairs
  - Loads recordings with `load_recording`
  - Applies `preprocess_for_model` to get windows
  - Returns `(windows, label)` per recording
- `collate_batch`:
  - Stacks variable-length windows from different recordings
  - Repeats labels so each window has a label

### Training Loop
1. **Setup**
   - Detects device (GPU/CPU)
   - Infers `in_channels` from a sample batch
   - Creates model via `create_model` with chosen architecture
2. **Optimization**
   - Loss: `CrossEntropyLoss`
   - Optimizers: `adam`, `sgd`, `adamw`
   - Schedulers: `ReduceLROnPlateau`, `CosineAnnealingLR`, or none
   - Optional gradient clipping and early stopping
3. **Checkpointing**
   - Saves best model to `models/checkpoints/{model}_best.pt` with:
     - `state_dict`
     - `model_kwargs` (name, `in_channels`, `num_classes`)

---

## 4. Continual Learning with EWC (`models/ewc.py`)

### EWC Concept
- When training on Task B after Task A, we want to avoid forgetting Task A.
- Elastic Weight Consolidation (EWC) penalizes changes to parameters that are important for previous tasks.

### Implementation
- `EWC`:
  - Computes Fisher Information per parameter on a dataset
  - Stores parameter means (from previous tasks)
  - Returns an additional EWC loss term

- `EWCTrainer`:
  - Wraps a base model from `create_model`
  - Trains sequential tasks:
    - Standard loss + EWC penalty
  - Can evaluate all tasks and compute summary stats

EWC-trained models are saved as `{model_name}_ewc_best.pt`.

---

## 5. Evaluation (`evaluation/evaluate_models.py`)

CLI entrypoint for offline evaluation.

### Modes
- `single` – evaluate one trained checkpoint
- `compare` – compare multiple checkpoints
- `cv` – k-fold cross-validation on a given architecture
- `cl` – hooks for continual learning evaluation (via `EWCTrainer`)

### Metrics & Outputs
- Accuracy, precision, recall, F1-score
- AUC-ROC, sensitivity, specificity
- Confusion matrices
- JSON summaries and plots saved to an output directory

---

## 6. Streamlit Dashboard (`app/app.py`)

The dashboard provides a clinician-friendly UI.

### Flow
1. **Upload EEG file** (EDF/EEG/CNT/VHDR/NPZ)
2. **Recording loaded** via `load_recording`
3. **Preprocessing** with `preprocess_for_model`
4. **Heuristic detection** via `simple_heuristic_score`
5. **Deep learning inference**:
   - User selects:
     - Model type: Standard vs EWC
     - Architecture: `eegnet`, `googlenet`, `densenet`, `vgg`, `resnet`, `rnn`, `deepcnn`
   - App loads checkpoint from `models/checkpoints/`
   - Runs model over windows and builds probability timeline
6. **Visualization**:
   - Session summary + detector metrics
   - EEG viewer (stacked waves)
   - Probability timelines + heatmap
   - Top suspicious windows
   - Spectrogram for selected window/channel
7. **Export**:
   - CSV of probabilities
   - PDF report combining timeline + top windows + notes

---

## 7. FastAPI Server (`main.py`, `src/api/server.py`)

For programmatic access in production or integration scenarios.

- `main.py --mode api`:
  - Loads config from `config.yaml`
  - Calls `src.api.server.create_app(config)`
  - Runs a Uvicorn server

- Typical endpoints:
  - `/health` – health check
  - `/predict` – accepts EEG input (or references) and returns seizure predictions

The API shares the same preprocessing and model logic as training/Streamlit.

---

## 8. Configuration and Environment

### Config (`config.yaml`)
- Contains global parameters used by `main.py` modes
- API host/port, data settings, etc.

### Dependencies (`requirements.txt`)
- Locked to versions compatible with Python 3.11
- Includes:
  - PyTorch, scikit-learn, MNE, pyedflib
  - FastAPI, Uvicorn, Streamlit
  - Plotting, logging, and utilities

---

## 9. High-Level Data Flow Summary

1. **Raw EEG** → `load_recording`
2. **Preprocessed windows** → `(N, C, T)`
3. **Heuristic and DL models** → probability per window
4. **Aggregation**:
   - Timelines, metrics, and visualizations
5. **Outputs**:
   - Streamlit dashboard
   - API JSON responses
   - Checkpoints, evaluation reports, and exported CSV/PDF

This architecture keeps preprocessing, model definition, training, evaluation, and serving **modular but shared**, so all entrypoints (CLI, API, dashboard) use the same core logic.
