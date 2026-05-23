# How to Run the Current Version (v4)

This guide explains how to run the main components of the project on your machine.

## 0. Prerequisites

- **Python**: 3.11 (recommended)
- **Virtual environment** (recommended):
  ```bash
  python -m venv .venv311
  .venv311\Scripts\activate  # Windows
  ```
- **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

All commands below assume you run them from the project root: `T:\suezier_p`.

---

## 1. Streamlit Dashboard (Recommended UI)

The Streamlit app is defined in `app/app.py`.

### Start the Dashboard
```bash
streamlit run app\app.py
```

Then open the URL shown in the terminal, typically:
- `http://localhost:8501`

### Using the Dashboard
1. Choose **Model Type**:
   - `Standard Models` – normal training
   - `Continual Learning (EWC)` – uses EWC checkpoints
2. Choose **Model Architecture**:
   - `eegnet`, `googlenet`, `densenet`, `vgg`, `resnet`, `rnn`, `deepcnn`
3. Upload an EEG file (EDF/EEG/CNT/VHDR/NPZ)
4. Wait for preprocessing and inference
5. Explore:
   - Threshold vs Deep Learning summaries
   - Probability timeline + heatmap
   - Top suspicious windows
   - Spectrogram
6. Export:
   - Predictions CSV
   - PDF report

---

## 2. API Server (FastAPI)

The entrypoint is `main.py`.

### Start the API Server
```bash
python main.py --mode api
```

Typical config (from `config.yaml`):
- Host: `0.0.0.0`
- Port: `8000`

### Check API
- Health: `GET http://localhost:8000/health`
- Docs (Swagger): `http://localhost:8000/docs`

### Example Predict Request
(Actual payload depends on your API contract; typically JSON or file upload)

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"example": "payload"}'
```

---

## 3. Training Models

Use `training/train.py` to train models on prepared datasets.

### Required Inputs
- `--data_dir`: directory with EEG files
- `--labels_csv`: CSV with columns like `file,label`

### Basic Example
```bash
python training\train.py \
  --data_dir data\windows \
  --labels_csv data\labels.csv \
  --out_dir models\checkpoints \
  --model eegnet
```

### Important Arguments
- `--model {eegnet,googlenet,densenet,vgg,resnet,rnn,deepcnn}`
- `--epochs N`
- `--batch_files N`
- `--lr LR`
- `--weight_decay WD`
- `--optimizer {adam,sgd,adamw}`
- `--scheduler {plateau,cosine,none}`
- `--grad_clip VALUE`
- `--patience N` (early stopping)

Trained checkpoints are saved to `models/checkpoints/{model}_best.pt`.

---

## 4. Evaluation

Use `evaluation/evaluate_models.py` for offline evaluation.

### Help
```bash
python evaluation\evaluate_models.py --help
```

### Single Model Evaluation
```bash
python evaluation\evaluate_models.py \
  --mode single \
  --model_path models\checkpoints\eegnet_best.pt \
  --model_name eegnet \
  --data_dir data\windows \
  --labels_csv data\labels.csv \
  --output_dir evaluation_results
```

### Model Comparison
```bash
python evaluation\evaluate_models.py \
  --mode compare \
  --model_paths \
    models\checkpoints\eegnet_best.pt \
    models\checkpoints\densenet_best.pt \
    models\checkpoints\resnet_best.pt \
  --data_dir data\windows \
  --labels_csv data\labels.csv \
  --output_dir evaluation_results
```

---

## 5. Main Entry Script (`main.py`)

`main.py` orchestrates several modes.

### Show Help
```bash
python main.py --help
```

### Modes
- `--mode download` – dataset download (if implemented)
- `--mode train` – training per patient / all patients
- `--mode realtime` – real-time prediction (streaming)
- `--mode api` – run FastAPI server
- `--mode dashboard` – run frontend dashboard (if configured)
- `--mode evaluate` – evaluation per patient

Example:
```bash
python main.py --mode train --patient chb01
```

---

## 6. Notes on Checkpoints and Channels

- Models are saved with `model_kwargs` that include:
  - `model_name`
  - `in_channels`
  - `num_classes`
- The Streamlit app and training script **infer** `in_channels` from the data.
- Ensure your data has consistent channel counts across train/eval/deploy.

---

## 7. Quick Sanity Check

After installation, you can quickly verify core imports:

```bash
python -c "from models.network import create_model; m = create_model('eegnet', in_channels=23); print(sum(p.numel() for p in m.parameters()))"
```

If this runs without error, the model code and dependencies are likely set up correctly.
