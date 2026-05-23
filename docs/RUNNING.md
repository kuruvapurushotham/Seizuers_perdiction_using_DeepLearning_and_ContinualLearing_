# How to Run the Seizure Prediction System

This document explains how to run the different parts of the project:

- Streamlit v3.1 web app (threshold-based detector)
- FastAPI web/API server (deep-learning-based EDF/PDF analysis)
- Training scripts for CHB-MIT and ds006519 datasets

> Always activate the virtual environment before running anything.

---

## 1. Prerequisites

- Python 3.8 or newer
- `pip`
- Recommended: a virtual environment

From the project root `t:\suezier_p`:

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you see `(venv)` in your terminal prompt, the environment is active.

---

## 2. Run the v3.1 Streamlit App (UI shown in README)

This is the "Fixed v3.1" Seizure Detection System using **absolute thresholds**.

From the project root, with venv activated:

```bash
cd app
python -m streamlit run app.py
```

Streamlit will print a URL such as:

- Local: `http://localhost:8501`

Open that URL in your browser.

### Supported file formats in v3.1 UI

- EDF (`.edf`)
- BrainVision (`.vhdr` + `.eeg` + `.vmrk`) — upload the **`.vhdr`** file
- Neuroscan CNT (`.cnt`)

The app will:

1. Load the recording (`app/io_utils.py`).
2. Run the **absolute threshold** seizure detector.
3. Show EEG plots, metrics, and detection results.

To stop the app, press `Ctrl + C` in the terminal.

---

## 3. Run the FastAPI Web / API Server

This server is implemented in `src/api/web_app.py` using FastAPI + Uvicorn. It loads the deep-learning model (CNN) from `data/models/trained_seizure_model.pth`.

From the project root, with venv active:

```bash
python src/api/web_app.py
```

You should see logs like:

- `Uvicorn running on http://0.0.0.0:8000`

### Endpoints

- UI: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- API (file upload): `POST /analyze` with `file` form field (EDF or PDF)

The server:

1. For EDF: uses `EDFReader` to preprocess EEG and run the CNN model.
2. For PDF: uses a synthetic EEG generator (placeholder) for demo.

Stop with `Ctrl + C` in the terminal.

> Note: `main.py --mode api` references a missing `src/api/server.py`, so the direct `python src/api/web_app.py` entrypoint is the currently working option.

---

## 4. Train on Local CHB-MIT EDF Files (CNN)

Script: `train.py`

This trains the CNN on EDF files organized as:

- `T:/suezier_p/dataset/training/`  (training EDFs)
- `T:/suezier_p/dataset/validation/` (validation EDFs)

Each folder should contain `.edf` recordings from the CHB-MIT dataset.

Run from project root:

```bash
python train.py
```

What it does:

1. Loads EDF files from the training and validation directories.
2. Builds `CHBMITDataset` instances with 2s epochs and seizure labels from `SEIZURE_ANNOTATIONS`.
3. Balances the dataset.
4. Trains the CNN model (`CNN1D_EEG`).
5. Saves model weights:
   - `data/models/trained_seizure_model_chbmit.pth`
   - `data/models/trained_seizure_model.pth` (default used by API/EDFReader)

---

## 5. Train on CHB-MIT via KaggleHub

Script: `train_kaggle_chbmit.py`

This script downloads the **Kaggle version** of CHB-MIT:

```python
import kagglehub
path = kagglehub.dataset_download("adibadea/chbmitseizuredataset")
```

and then:

- Finds all `*.edf` recursively.
- Splits them into train/validation in code.
- Reuses `CHBMITDataset`, `balance_dataset`, and `train_model`.

Run from project root (venv active):

```bash
python train_kaggle_chbmit.py
```

It will:

- Download (or reuse cached) Kaggle dataset.
- Train the CNN.
- Save:
  - `data/models/trained_seizure_model_chbmit_kaggle.pth`
  - `data/models/trained_seizure_model.pth` (default for API/EDFReader)

---

## 6. Train on ds006519 iEEG Dataset

Script: `train_ds006519.py`

Target dataset:

- BIDS iEEG dataset at: `dataset/ds006519-main` (OpenNeuro ds006519)

This script:

1. Searches `dataset/ds006519-main/sub-*/ses-*/ieeg/*_task-dcs_ieeg.vhdr`.
2. Loads the BrainVision recordings with MNE.
3. Filters, notch-filters, resamples to 256 Hz.
4. Uses events (`*_events.tsv`) to define **prestim** vs **poststim** intervals.
5. Extracts 2s sliding windows and features using `EEGFeatureExtractor`.
6. Trains a simple classifier (XGBoost if installed, otherwise GradientBoosting).

Run from project root:

```bash
python train_ds006519.py
```

This is a **prestim vs poststim classifier**, not a classic seizure vs non-seizure model.

---

## 7. Where Models Are Stored and Used

- Deep CNN model weights:
  - `data/models/trained_seizure_model.pth` (main default)
  - Other variants: `trained_seizure_model_chbmit.pth`, `trained_seizure_model_chbmit_kaggle.pth`.
- Used by:
  - `src/api/web_app.py` (FastAPI server)
  - `src/data/edf_reader.py` (`EDFReader` class)

The Streamlit v3.1 app uses **absolute thresholds** only and does not load these models.
