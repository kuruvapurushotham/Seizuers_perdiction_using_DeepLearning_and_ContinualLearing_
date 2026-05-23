# Project Workflow – Step by Step

This document explains how the current version of the project works **end-to-end**, step by step.

---

## 1. Setup and Environment

1. **Clone / open the repo** on your machine (here assumed at `T:\suezier_p`).
2. **Create and activate a virtual environment** (Python 3.11 recommended):
   ```bash
   python -m venv .venv311
   .venv311\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) **Adjust `config.yaml`** for API host/port, data paths, etc.

At this point the codebase and environment are ready.

---

## 2. Data Ingestion

### 2.1 Supported Formats
- EDF/EEG/CNT/VHDR (via MNE + `pyedflib`)
- NPZ window files (internal format: `(N, C, T)` or `(N, T, C)`)

### 2.2 Loading Functions
- All entrypoints rely on **one place** for loading:
  - `app/io_utils.py` → `load_recording(file_like, filename)`

This function returns a dictionary with at least:
- `data`: 2D NumPy array `(channels, samples)`
- `sfreq`: sampling rate (Hz)
- `ch_names`: optional channel name list

---

## 3. Preprocessing

All pipelines use the same preprocessing logic defined in:
- `app/preprocess.py` → `preprocess_for_model(X, sfreq)`

### 3.1 Steps
1. **Input**: raw EEG `X` with shape `(channels, samples)` and sampling rate `sfreq`.
2. **Filtering**:
   - Bandpass (keep relevant EEG frequency bands)
   - Notch filter (remove 50/60 Hz line noise)
3. **Standardization / scaling** per channel.
4. **Windowing**:
   - Split the continuous signal into fixed-length windows
   - Output shape: `(num_windows, channels, samples_per_window)`

### 3.2 Heuristic Detector
- `simple_heuristic_score(windows, sfreq, return_per_window=True)` returns:
  - A scalar heuristic score
  - A per-window score array
- Used for:
  - Simple threshold-based detection
  - Reference timeline in UI/evaluation

---

## 4. Model Architectures

Implemented in `models/network.py` and created via:

```python
from models.network import create_model
model = create_model(name, in_channels, num_classes)
```

Available `name` values:
- `eegnet`, `googlenet`, `densenet`, `vgg`, `resnet`, `rnn`, `deepcnn`

Key points:
- All models expect input shape `(batch, channels, samples)`.
- `in_channels` is inferred from data (e.g. 19–23 EEG channels).
- `num_classes` is usually 2 (seizure / non-seizure).

See `docs/models_and_architectures.md` for per-model details.

---

## 5. Training Pipeline

Main script: `training/train.py`.

### 5.1 Dataset and DataLoader
1. **EEGDataset**:
   - Reads a CSV (`--labels_csv`) containing `file` and `label` columns.
   - For each row:
     - Builds full path: `data_dir / file`.
     - Calls `load_recording` → gets raw EEG.
     - Calls `preprocess_for_model` → gets windows `(N, C, T)`.
2. **collate_batch**:
   - Takes a batch of `(windows, label)` pairs with variable `N`.
   - Stacks all windows into a single batch `x`.
   - Repeats labels for each window to get `y`.

### 5.2 Model Creation
1. Take one batch from DataLoader to infer `in_channels` from `x.shape[1]`.
2. Use CLI argument `--model` to pick architecture.
3. Call `create_model(model_name, in_channels, num_classes=2)`.

### 5.3 Optimization
- Loss: `CrossEntropyLoss`.
- Optimizer: chosen via `--optimizer {adam,sgd,adamw}`.
- LR scheduler: `--scheduler {plateau,cosine,none}`.
- Optional:
  - Weight decay (`--weight_decay`)
  - Gradient clipping (`--grad_clip`)
  - Early stopping (`--patience`).

### 5.4 Checkpointing
- During training, best model by accuracy is saved to:
  - `models/checkpoints/{model_name}_best.pt`
- Checkpoint contents:
  - `state_dict` (weights)
  - `model_kwargs` (e.g. `model_name`, `in_channels`, `num_classes`).

---

## 6. Continual Learning (EWC)

Implemented in `models/ewc.py`.

### 6.1 EWC Core
- `EWC`:
  - Given a trained model & dataset, computes Fisher Information.
  - Stores parameter means and importances.
  - Provides EWC penalty to add to loss when training on new tasks.

### 6.2 EWCTrainer
- Wraps a base model from `create_model`.
- For each new task:
  1. Train on task data.
  2. Compute Fisher Information.
  3. Add EWC penalty for future tasks.
- Supports evaluation on all tasks and summary metrics.

EWC-trained checkpoints are named like `models/checkpoints/{model_name}_ewc_best.pt`.

---

## 7. Evaluation

Script: `evaluation/evaluate_models.py`.

### 7.1 Modes
- `--mode single`: evaluate one checkpoint.
- `--mode compare`: evaluate several checkpoints and compare.
- `--mode cv`: run cross-validation for a given architecture.
- `--mode cl`: hooks for evaluating continual learning setups.

### 7.2 Metrics and Outputs
- Accuracy, precision, recall, F1, AUC-ROC.
- Sensitivity and specificity.
- Confusion matrices.
- JSON + PNG plots saved to an `--output_dir` (default: `evaluation_results`).

---

## 8. Streamlit Dashboard

File: `app/app.py`.

### 8.1 User Flow
1. User opens the app:
   ```bash
   streamlit run app\app.py
   ```
2. In the sidebar:
   - Select **Model Type**: Standard vs EWC.
   - Select **Model Architecture**: `eegnet`, `densenet`, etc.
   - Upload an EEG file.
   - (Optionally) add clinician notes.
3. The app:
   - Loads the recording (`load_recording`).
   - Preprocesses EEG (`preprocess_for_model`).
   - Runs heuristic detector (`simple_heuristic_score`).
   - Infers channel count and loads the corresponding checkpoint.
   - Runs deep learning inference over windows.
4. The UI displays:
   - Session summary.
   - Threshold vs deep learning metrics.
   - Probability timeline + heatmap.
   - Top suspicious windows.
   - Spectrogram for a chosen window/channel.
5. Exports:
   - Predictions as CSV.
   - Clinical PDF report.

---

## 9. FastAPI Server

Main entry: `main.py --mode api`.

### 9.1 Startup
1. `main.py` reads `config.yaml`.
2. Calls `src.api.server.create_app(config)`.
3. Starts Uvicorn with the configured host/port.

### 9.2 Serving Predictions
- `/health`: health check.
- `/predict`: accepts input (depending on implementation) and returns model predictions.
- Internally reuses:
  - `load_recording`
  - `preprocess_for_model`
  - `create_model` + checkpoint loading

---

## 10. Overall Flow Summary

1. **Data** → `load_recording` → `(C, T)` EEG.
2. **Preprocessing** → `preprocess_for_model` → windows `(N, C, T_win)`.
3. **Heuristic** → `simple_heuristic_score` → baseline scores.
4. **Deep Model** → `create_model` + checkpoint → probabilities per window.
5. **Aggregation & Visualization** → timelines, maps, metrics.
6. **Serving** via Streamlit or FastAPI, or evaluation via CLI.

This is the complete high-level, step-by-step picture of how the current project operates.
