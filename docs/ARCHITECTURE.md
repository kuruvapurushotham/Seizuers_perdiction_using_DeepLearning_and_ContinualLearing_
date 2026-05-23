# Project Architecture

This document describes the high-level architecture of the Seizure Prediction System.

---

## 1. Top-Level Components

- **Streamlit UI (v3.1)**
  - File: `app/app.py`
  - Purpose: Interactive web UI for seizure detection using absolute thresholds.
  - Input: EEG files (EDF, BrainVision `.vhdr`, Neuroscan `.cnt`).
  - Output: Seizure/no-seizure decision, confidence, EEG plots.

- **FastAPI Web / API App**
  - File: `src/api/web_app.py`
  - Purpose: Web UI + REST API for seizure prediction using a deep CNN model.
  - Input: EDF EEG files or PDF reports.
  - Output: JSON with probabilities, abnormalities, and visualization-ready data.

- **Model Training Pipelines**
  - `train.py`: CNN training on CHB-MIT EDFs from `dataset/training` and `dataset/validation`.
  - `train_kaggle_chbmit.py`: CNN training on KaggleHub CHB-MIT dataset.
  - `train_ds006519.py`: Feature-based classifier on OpenNeuro ds006519 iEEG (prestim vs poststim).

- **Core Data & Models**
  - `src/data/edf_reader.py`: EDF reader, preprocessing, and CNN-based prediction.
  - `src/data/feature_extractor.py`: Feature extraction (filterbank, spatial, temporal).
  - `src/models/deep_learning_models.py`: CNN / LSTM / CNN-LSTM architectures.

---

## 2. Data Flow: Streamlit v3.1 App

1. **Upload EEG file** in `app/app.py`.
2. `load_recording` (from `app/io_utils.py`):
   - Detects format (`.edf`, `.eeg`/`.vhdr` BrainVision, `.cnt`).
   - Uses MNE / pyedflib to read raw EEG into a NumPy array (`data`) and metadata (`sfreq`, `ch_names`).
3. **Threshold-based detector** (`detect_seizure_absolute` in `app/app.py`):
   - Splits the signal into 10s windows (5s overlap).
   - Computes amplitude standard deviation, line length, and peak-to-peak amplitude.
   - Normalizes against absolute thresholds calibrated from CHB-MIT.
   - Combines features into a score and checks for consecutive high-activity windows.
4. **Visualization**:
   - `plot_eeg_waves` uses Plotly to display multi-channel EEG.
   - Streamlit widgets show metrics, confidence, and detection results.

The Streamlit app does **not** use the deep CNN model; it is standalone and tuned for interpretable threshold-based detection.

---

## 3. Data Flow: FastAPI Seizure Prediction Server

File: `src/api/web_app.py`

### 3.1 Initialization

- Creates `FastAPI` app.
- Initializes `ModelConfig` and a CNN model (`create_model('cnn', ModelConfig())`).
- Tries to load weights from `data/models/trained_seizure_model.pth`.
- Creates a shared `EEGFeatureExtractor` and `EDFReader` instance.

### 3.2 EDF File Path

1. Client uploads EDF to `/analyze`.
2. Server writes bytes to a temporary `.edf` file.
3. `edf_reader.read_edf` (in `src/data/edf_reader.py`):
   - Uses `pyedflib.EdfReader` to read:
     - Multi-channel signals
     - Channel labels, sampling rates
     - Duration and header metadata
4. `edf_reader.preprocess_eeg`:
   - Selects a subset of channels (first 18 or requested names).
   - Resamples each channel to 256 Hz.
   - Pads/truncates to exactly 18 channels.
5. `edf_reader.predict_seizures`:
   - Splits the preprocessed EEG into 2s epochs.
   - For each epoch:
     - Computes feature vector via `EEGFeatureExtractor.extract_epoch_features`.
     - Feeds the raw epoch into the CNN model (`CNN1D_EEG`).
   - Aggregates per-epoch probabilities and predictions.
   - Builds an `abnormalities` list for suspicious epochs.
6. The API wraps this into JSON, along with downsampled signals for plotting.

### 3.3 PDF Path

- For PDF uploads, the current implementation uses `extract_eeg_from_pdf`:
  - This is a **placeholder** that generates synthetic EEG for demo purposes.
  - The same analysis pipeline is used on this synthetic signal.

---

## 4. Training Pipelines

### 4.1 CHB-MIT CNN Training (`train.py`)

- Uses real CHB-MIT EDF files stored locally.
- `CHBMITDataset`:
  - Reads EDF via `EDFReader`.
  - Preprocesses to 18 channels at 256 Hz.
  - Segments into 2s epochs.
  - Labels epochs as seizure / non-seizure using manually curated `SEIZURE_ANNOTATIONS` (per-file time intervals).
  - Optionally generates synthetic seizure epochs if none are found.
- `balance_dataset` downsamples the majority class.
- `train_model`:
  - Builds CNN1D model (`CNN1D_EEG`).
  - Trains with weighted cross-entropy.
  - Tracks accuracy, sensitivity, specificity, and F1.
  - Saves best weights and final weights to `data/models/`.

### 4.2 KaggleHub CHB-MIT Training (`train_kaggle_chbmit.py`)

- Uses KaggleHub to download `adibadea/chbmitseizuredataset`.
- Recursively finds all `.edf` files.
- Splits into train/validation sets.
- Reuses `CHBMITDataset`, `balance_dataset`, and `train_model` exactly like `train.py`.
- Saves a Kaggle-specific checkpoint and updates `trained_seizure_model.pth`.

### 4.3 ds006519 iEEG Training (`train_ds006519.py`)

- Uses OpenNeuro ds006519 iEEG dataset at `dataset/ds006519-main`.
- Loads BrainVision recordings via `.vhdr` using MNE.
- Preprocessing:
  - Band-pass (0.5–100 Hz)
  - Notch filtering at 50 Hz and harmonics
  - Resampling to 256 Hz
- Uses BIDS `*_events.tsv` files to define **prestim** and **poststim** segments.
- Windowing:
  - 2s windows with 1s step.
  - Windows fully inside event intervals.
- Feature extraction:
  - Uses `EEGFeatureExtractor` with filterbank and spatial variance features.
- Classification:
  - XGBoost (if installed) or scikit-learn GradientBoosting.
  - The target task is prestim vs poststim, not seizure detection.

---

## 5. Core Libraries & Responsibilities

- **MNE-Python**
  - Reading multi-format EEG (EDF, BrainVision, CNT).
  - Preprocessing: filtering, notch, resampling.

- **pyedflib**
  - Low-level EDF reading in `EDFReader`.

- **Streamlit**
  - Rapid UI for v3.1 threshold-based detector.

- **FastAPI + Uvicorn**
  - Production-style API and HTML UI for CNN-based prediction.

- **PyTorch**
  - Deep learning (CNN, LSTM, hybrid models) for seizure classification.

- **scikit-learn / XGBoost**
  - Classical ML classifiers for feature-based tasks (e.g., ds006519 prestim vs poststim).

- **Plotly**
  - Interactive EEG waveform visualization in the Streamlit app.

---

## 6. Existing Documentation

Additional project docs in the root:

- `README.md` – high-level overview and quick start.
- `RELEASE_v3.1.md` – v3.1 release notes and changes.
- `EEG_FORMAT_SUPPORT.md` – details on supported EEG file formats.
- `FEATURE_EEG_SUPPORT.md` – feature and capability overview.

The `docs/` files (including this one) are meant as a deeper technical reference.
