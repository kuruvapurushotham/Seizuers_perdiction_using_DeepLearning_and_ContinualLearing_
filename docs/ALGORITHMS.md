# Algorithms and Models

This document explains the key algorithms used in the Seizure Prediction System.

- Hybrid seizure detection dashboard (Streamlit v4.0: thresholds + deep learning models)
- Feature extraction pipeline
- Deep learning models (CNN, LSTM, hybrid)
- Training strategy for CHB-MIT
- Feature-based classifier for ds006519

---

## 1. Seizure Detection Dashboard (v4.0)

Implementation: `app/app.py` in the function `detect_seizure_absolute`.

### 1.1 Windowing

- Input: multi-channel EEG `data` with sampling frequency `sfreq`.
- Windows:
  - Window length: **10 seconds**
  - Step: **5 seconds** (50% overlap)
- For each window:
  - Extracts `window = data[:, start:start + win_samples]`.

### 1.2 Features per Window

For each window:

1. **Amplitude standard deviation (amp_std)**
   - `amp_std = np.std(window)`
   - Measures average variability of the signal.
   - Seizure segments show significantly higher variance than baseline.

2. **Line length (line_length)**
   - `line_length = np.mean(np.abs(np.diff(window, axis=1)))`
   - Captures overall activity and complexity of the waveform.
   - Higher during seizures due to rapid, high-amplitude changes.

3. **Peak-to-peak amplitude (p2p)**
   - `p2p = np.mean(np.ptp(window, axis=1))`
   - Average peak-to-peak over channels.
   - Seizures tend to have higher peak-to-peak values.

### 1.3 Absolute Thresholds

Based on calibration on CHB-MIT (as comments in `app.py`):

- Normal windows:
  - `amp_std ≈ 3–4e-5`
  - `line_length ≈ 6–9e-6`
- Seizure windows:
  - `amp_std ≈ 8–10e-5`
  - `line_length ≈ 1.5–1.8e-5`

Strict thresholds are chosen to avoid false positives:

```python
AMPLITUDE_STD_THRESHOLD = 9.0e-5
LINE_LENGTH_THRESHOLD = 1.8e-5
COMBINED_SCORE_THRESHOLD = 0.75
```

Each feature is normalized relative to its threshold and clipped, producing scores in [0, 1.5] which are then scaled:

- `amp_score`
- `ll_score`
- `p2p_score`

The final **combined score** is a weighted sum:

```python
combined_score = 0.4 * amp_score + 0.4 * ll_score + 0.2 * p2p_score
```

### 1.4 Decision Logic

1. Mark windows with `combined_score > COMBINED_SCORE_THRESHOLD` as **high-activity**.
2. Compute:
   - `high_ratio` = high-activity windows / total windows.
   - `max_score`, `percentile_95` of scores.
   - Longest run of consecutive high-activity windows (`max_consecutive`).
3. Decision rules:

- **Seizure detected (high confidence)** if:
  - `max_consecutive >= 3` and `high_ratio > 0.003`
- **Seizure detected (low confidence)** if:
  - `max_consecutive >= 2` and `max_score > 0.85`
- Otherwise: no seizure.

A probability and a human-readable confidence string (`"High"`, `"Medium"`, `"Low"`) are derived from the scores.

This logic emphasizes **sustained high activity** over multiple windows, reducing isolated false alarms.

---

## 2. Feature Extraction Pipeline

Implementation: `src/data/feature_extractor.py`.

The `EEGFeatureExtractor` combines spectral, spatial, and temporal features.

### 2.1 Configuration (`FeatureConfig`)

Key parameters:

- `sampling_rate` (default 256 Hz)
- `epoch_length` (default 2.0 s)
- `num_channels` (default 18)
- `num_filters` (filterbank count, default 8)
- `freq_min`, `freq_max` (spectral range, usually 0.5–25 Hz)
- `window_size` (temporal stacking length)

### 2.2 Spectral Features (`FilterBank` + `SpectralFeatureExtractor`)

Per channel, a **filterbank** of bandpass filters is applied:

- Frequency bands are automatically created between `freq_min` and `freq_max`.
- For each band (per channel):
  - Design a 4th-order Butterworth bandpass filter.
  - Apply to the epoch.
  - Compute **energy** as sum of squared samples.

This produces a matrix `[channels, num_filters]`, flattened to a vector.

### 2.3 Spatial Features (`SpatialFeatureExtractor`)

For each epoch:

- **Channel variance**: variance of each channel.
- (Optionally) inter-channel correlation and spatial gradients (difference between adjacent channels) can be computed.

These capture spatial distribution of power across electrodes.

### 2.4 Temporal Features (`TemporalFeatureExtractor`)

The extractor keeps a short history of past epochs and can create a stacked feature vector across multiple consecutive epochs (size `window_size`). This allows the model to see short-term temporal context in addition to single-epoch features.

### 2.5 Combined Features

`EEGFeatureExtractor.extract_epoch_features`:

1. Spectral filterbank energy features.
2. Spatial variance features.
3. Concatenates into one feature vector.

`extract_features_from_signal` segments continuous EEG into epochs and returns a feature matrix over time.

These features are used in:

- `EDFReader.predict_seizures` (alongside the CNN).
- `train_ds006519.py` (classical ML classifier).

---

## 3. Deep Learning Models

Implementation: `src/models/deep_learning_models.py`.

### 3.1 CNN1D_EEG (Default CNN)

- Input shape: `[batch, channels=18, time=512]` (2 seconds at 256 Hz).
- Architecture:
  1. Conv1D: 18 → 64 filters, kernel 7, padding 3.
  2. BatchNorm1d(64), ReLU, MaxPool1d(2).
  3. Conv1D: 64 → 128 filters, kernel 5, padding 2.
  4. BatchNorm1d(128), ReLU, MaxPool1d(2).
  5. Conv1D: 128 → 256 filters, kernel 3, padding 1.
  6. BatchNorm1d(256), ReLU, MaxPool1d(2).
  7. Flatten.
  8. FC: 256 * (sequence_length / 8) → 256, ReLU, Dropout.
  9. FC: 256 → 2 (binary output: seizure / non-seizure).

- Training:
  - Loss: CrossEntropyLoss.
  - Optimizer: Adam.
  - Learning rate scheduler: ReduceLROnPlateau on validation F1.
  - Class imbalance handled via dataset balancing + emphasis on F1.

### 3.2 LSTM_EEG

- Input: `[batch, channels, time]` → transposed to `[batch, time, channels]`.
- Bidirectional LSTM with attention:
  - LSTM output: sequence of hidden states.
  - Attention layer computes weights over time and aggregates into a context vector.
  - FC layers map context to 2-class prediction.

This model focuses on temporal patterns across time steps.

### 3.3 CNN-LSTM Hybrid

- CNN front-end extracts spatial features and compresses time.
- LSTM back-end models temporal dynamics of these spatial features.
- Combines benefits of both spatial and temporal modeling.

Currently, the **CNN1D_EEG** is the primary model used in training scripts and the API.

---

## 4. Training Strategy (CHB-MIT)

Implementation: `train.py` and `train_kaggle_chbmit.py`.

### 4.1 CHBMITDataset

Per EDF file:

1. Read and preprocess EDF via `EDFReader` (18 channels, 256 Hz).
2. Segment the signal into 2s epochs.
3. Label each epoch using known seizure intervals (`SEIZURE_ANNOTATIONS`):
   - For a file like `chb01_03.edf`, seizure times might be `(2996, 3036)` in seconds.
   - Any epoch overlapping a seizure interval is labeled `1`, otherwise `0`.
4. If no seizure epochs are found, synthetic seizure-like patterns can be injected into some normal epochs to enable training.

### 4.2 Dataset Balancing

- Many more non-seizure than seizure epochs.
- `balance_dataset` downsamples normal epochs to a maximum ratio (e.g., 5:1 normal:seizure).
- This prevents the model from simply predicting "no seizure".

### 4.3 Training Loop

- Iterates over epochs:
  - Tracks training loss & accuracy.
  - Evaluates on validation set each epoch.
  - Computes sensitivity (recall of seizure class), specificity, and F1.
- Best model is chosen based primarily on F1 (to balance precision/recall).

The resulting model is saved and used by the FastAPI server and `EDFReader`.

---

## 5. Feature-Based Classifier (ds006519)

Implementation: `train_ds006519.py`.

- Dataset: OpenNeuro `ds006519-main` (intracranial EEG during cortical stimulation).
- Goal: classify **prestim** vs **poststim** stimulation windows, not seizure vs non-seizure.

### 5.1 Data Handling

- Finds `sub-*/ses-*/ieeg/*_task-dcs_ieeg.vhdr` BrainVision header files.
- Uses MNE to read the corresponding `.eeg`/`.vmrk` data.
- Preprocessing:
  - Band-pass filter 0.5–100 Hz.
  - Notch filter at 50 Hz and harmonics.
  - Resample to 256 Hz.
  - Use first 18 channels (for consistent feature dimension).

### 5.2 Labels from Events

- Each recording has an events file like `*_events.tsv` with rows:
  - `prestim`: baseline interval before stimulation.
  - `poststim`: interval during/after stimulation.
- For each event interval, sliding windows are extracted:
  - 2s window, 1s step, fully inside the interval.
- Labels:
  - `prestim` → 0
  - `poststim` → 1

### 5.3 Features and Model

- Feature extractor: `EEGFeatureExtractor` (spectral + spatial variance).
- A classical ML model is trained:
  - XGBoost (`binary:logistic`) if available.
  - Fallback: `GradientBoostingClassifier` (scikit-learn).
- Evaluation:
  - Train/test split with stratification.
  - Classification report + AUC.

This classifier is separate from the CNN seizure detector and is tailored to this specific stimulation dataset.

---

## 6. Summary

- **Streamlit v4.0**: hybrid dashboard combining calibrated threshold-based detection and deep-learning models for EEG-based seizure analysis.
- **FastAPI + EDFReader + CNN**: more advanced, model-based seizure prediction with epoch-level probabilities.
- **Feature extraction**: robust spectral + spatial features for both classical ML and DL.
- **Training scripts**: flexible pipelines for local CHB-MIT EDFs, Kaggle CHB-MIT, and ds006519 stimulation data.

These components together form a modular system that can be extended with new models or datasets while reusing the same core utilities.
