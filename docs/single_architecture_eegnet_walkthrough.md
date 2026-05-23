# End-to-End Walkthrough with a Single Architecture (EEGNet)

This document shows the **full process of the project** using one concrete architecture: **EEGNet1D (`eegnet`)**.

> The same flow applies to other architectures (`densenet`, `resnet`, etc.), only the `model_name` changes.

---

## 1. Choose the Architecture: EEGNet

EEGNet is a compact 1D CNN designed for EEG.

Creation:
```python
from models.network import create_model

model = create_model("eegnet", in_channels=23, num_classes=2)
```

In practice you rarely call this directly; it’s done by:
- `training/train.py` during training
- `app/app.py` when loading checkpoints for inference

---

## 2. Training EEGNet

### 2.1 Prepare Data
1. Collect EEG recordings into a folder, e.g. `data/windows/`.
2. Create a CSV file, e.g. `data/labels.csv` with columns:
   ```csv
   file,label
   rec1.edf,0
   rec2.edf,1
   ...
   ```

### 2.2 Run Training
From the project root:

```bash
python training\train.py \
  --data_dir data\windows \
  --labels_csv data\labels.csv \
  --out_dir models\checkpoints \
  --model eegnet \
  --epochs 50 \
  --batch_files 4 \
  --lr 1e-3
```

**What happens internally:**
1. `EEGDataset` reads each `file,label` pair.
2. For each file:
   - `load_recording` loads raw EEG `(C, T)`.
   - `preprocess_for_model` returns windows `(N, C, T_win)`.
3. `collate_batch` stacks all windows from a batch into `(B, C, T_win)` and repeats labels.
4. The script infers `in_channels = C` from a batch.
5. It calls:
   ```python
   model = create_model("eegnet", in_channels=C, num_classes=2)
   ```
6. Model is trained using Adam (by default) and CrossEntropy loss.
7. The best checkpoint is saved as:
   - `models/checkpoints/eegnet_best.pt`
   - Contains `state_dict` and `model_kwargs` (`model_name`, `in_channels`, `num_classes`).

---

## 3. Using EEGNet in the Streamlit App

### 3.1 Start Streamlit
```bash
streamlit run app\app.py
```

### 3.2 Select EEGNet
In the sidebar:
1. Set **Model Type** = `Standard Models`.
2. Set **Model Architecture** = `eegnet`.
3. Upload an EEG file (e.g. `rec_test.edf`).

### 3.3 Inference Flow for EEGNet
Once you upload a file:
1. `load_recording` loads `raw_data` with shape `(C, T)` and sampling rate `sfreq`.
2. `preprocess_for_model(raw_data, sfreq)` produces `windows` `(N, C, T_win)`.
3. `simple_heuristic_score(windows, sfreq, return_per_window=True)`
   - Computes a classical heuristic score per window.
   - Provides a reference timeline.
4. Channel count is determined by:
   ```python
   actual_channels = raw_data.shape[0]
   ```
5. The app calls:
   ```python
   dl_result = run_dl_inference(windows, device, "eegnet", actual_channels)
   ```
   which internally:
   - Loads `eegnet_best.pt` via `load_dl_checkpoint`.
   - Recreates EEGNet with `in_channels=actual_channels` and `num_classes=2`.
   - Loads `state_dict` and sets model to `.eval()`.
   - Runs `model(windows_tensor)` to get logits.
   - Applies `softmax` to get seizure probabilities per window.
6. The output is a dict like:
   ```python
   {
       "probs": np.array([...]),
       "max": float(...),
       "mean": float(...),
       "count": int(...),
       "model_name": "eegnet",
       "model_type": "Standard"
   }
   ```

### 3.4 What You See in the UI
- **Session panel**:
  - File name, number of channels, duration, number of windows.
- **Threshold Detector**:
  - Result (seizure / no seizure), confidence, max heuristic score.
- **Deep Learning (EEGNet)**:
  - Model type = `Standard`.
  - Model = `Eegnet`.
  - Max probability, mean probability, number of high-risk windows.
- **Probability Timeline**:
  - Line plot of per-window probabilities from EEGNet.
  - Optional heuristic timeline for comparison.
- **Top Suspicious Windows**:
  - Windows with highest EEGNet probabilities.
  - Mini-plots of EEG segments per window.
- **Spectrogram**:
  - Time–frequency view of a selected window/channel.
- **Exports**:
  - CSV file with per-window probabilities.
  - PDF report summarizing EEGNet results + heuristic + clinician notes.

---

## 4. Using EEGNet via the API

### 4.1 Start API Server
```bash
python main.py --mode api
```

This will:
1. Read config from `config.yaml`.
2. Create FastAPI app using `src.api.server.create_app(config)`.
3. Start Uvicorn on the configured host/port (e.g. `http://0.0.0.0:8000`).

### 4.2 Prediction Endpoint (Conceptual)
While the exact request/response schema depends on your API implementation, a typical flow is:
1. Client sends EEG data (or file reference) to `/predict`.
2. Server:
   - Loads EEG (like Streamlit) via `load_recording` or equivalent.
   - Preprocesses via `preprocess_for_model`.
   - Loads EEGNet checkpoint (`eegnet_best.pt`).
   - Runs inference to generate per-window probabilities.
   - Aggregates into a summary prediction and returns JSON.

This reuses the **same EEGNet model and preprocessing** as training and Streamlit.

---

## 5. Evaluating EEGNet Offline

### 5.1 Single-Model Evaluation
```bash
python evaluation\evaluate_models.py \
  --mode single \
  --model_path models\checkpoints\eegnet_best.pt \
  --model_name eegnet \
  --data_dir data\windows \
  --labels_csv data\labels.csv \
  --output_dir evaluation_results
```

### 5.2 Outputs
- Metrics (accuracy, precision, recall, F1, AUC-ROC).
- Confusion matrix.
- JSON + PNG plots saved under `evaluation_results/`.

---

## 6. Summary of the EEGNet Path

1. **Training**:
   - `training/train.py` + `--model eegnet` → `eegnet_best.pt`.
2. **Streamlit**:
   - User selects `Standard` + `eegnet` → app loads `eegnet_best.pt` and shows interactive results.
3. **API**:
   - `main.py --mode api` → `/predict` uses `eegnet_best.pt` internally.
4. **Evaluation**:
   - `evaluation/evaluate_models.py --mode single --model_name eegnet` → offline metrics.

This single-architecture walkthrough (EEGNet) mirrors the **full project process** and can be used as a template for other models by changing only `model_name` and checkpoints.
