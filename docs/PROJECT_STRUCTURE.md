# Project Structure Overview

This document explains the purpose of every top-level folder and key files in the repository. It highlights how each component fits into the EEG seizure prediction pipeline, why it exists, and when you should touch it.

> **Legend**
> - **Purpose** – Why the folder/file exists.
> - **Contents / Key Files** – Notable items inside.
> - **Usage Notes** – How or when to interact with it.

---

## Root-Level Files & Directories

| Path | Purpose | Usage Notes |
|------|---------|-------------|
| `.bidsignore` | Patterns ignored when packaging EEG datasets using the BIDS standard. | Leave as-is unless you add/remove dataset files that should be skipped in BIDS exports. |
| `.gitattributes` | Git text/binary handling rules (e.g., line-endings). | Extending the repo? Update if you add new file types needing special treatment. |
| `.gitignore` | Files and folders ignored by Git (virtualenvs, checkpoints, etc.). | Add entries here (e.g., new virtual environments or generated artifacts) before committing. |
| `.venv311/`, `venv/` | Local virtual environment folders. | Development only; keep out of Git commits. |
| `.zed/` | Editor-specific settings (Zed). | Safe to leave untouched unless refactoring editor configs. |
| `app/` | Core application package (training/inference entrypoints, IO helpers). | See [Detailed Folder Descriptions](#detailed-folder-descriptions). |
| `basepaper/` | Reference paper PDF supporting model ideas. | Use for documentation or citations. |
| `checkpoints/` | Default location for saved model checkpoints. | Generated during training; typically gitignored. |
| `config/` & `config.yaml` | Central configuration templates (YAML). | Edit to change global training/inference defaults. |
| `data/`, `dataset/`, `dataset_sample/`, `eeg_data/` | Raw and prepared EEG datasets. | Large files; not committed. Maintain structure for reproducibility. |
| `docs/` | Comprehensive documentation set (how-to guides, release notes). | Author or update guides when workflows change. |
| `evaluation/` | Evaluation scripts and utilities. | Run post-training comparison or benchmarking code from here. |
| `logs/`, `results/` | Output artifacts (logs, evaluation summaries). | Generated during training/evaluation. Usually ignored by Git. |
| `main.py` | High-level script orchestrating training/evaluation pipelines. | Entry point when running the project as a script. |
| `models/` | Torch model definitions (classical CNN/RNN architectures, EWC). | Source of reusable model components. |
| `networks_and_models.md` | Documentation describing implemented networks/algorithms. | Reference when comparing architectures. |
| `ref/` | Reference assets (e.g., label maps, metadata). | Keep curated supporting materials here. |
| `requirements.txt` | Python dependencies. | Update when adding/removing packages. |
| `scripts/` | Auxiliary automation scripts (data prep, batch jobs). | Run specialized maintenance or preprocessing tasks. |
| `src/` | Application code not inside `app/` (API server, datasets, advanced models). | For modular development; contains subpackages. |
| `training/` | Training CLI, dataset loader, utilities. | Use to launch training jobs with CLI arguments. |

---

## Detailed Folder Descriptions

### `app/`

| File | Purpose | Why/When to Use |
|------|---------|-----------------|
| `app.py` | Main FastAPI/Flask (depending on framework) entrypoint to expose training/inference services. | Launch to serve inference APIs or UI integrations. |
| `inference.py` | Utilities to load trained checkpoints and run model predictions on new EEG data. | Use when deploying or testing models on unseen recordings. |
| `io_utils.py` | Handles reading EEG files, metadata parsing, serialization. | Centralized IO logic ensures consistent preprocessing across scripts. |
| `preprocess.py` | Signal preprocessing pipeline (filtering, windowing). | Imported by training/inference to transform raw EEG into model-ready tensors. |

### `basepaper/`
- **Contents**: `Robust_Seizure_Prediction_Model_Using_EEG_Signal.pdf` – the academic reference underpinning the architecture choices.
- **Usage**: cite or consult when explaining model rationale or reproducing methodology.

### `config/`
- **`config.yaml`**: Template for global settings (data paths, training hyperparameters, logging). Override values here instead of hard-coding them in scripts.

### `data/`, `dataset/`, `dataset_sample/`, `eeg_data/`
- **Purpose**: Store raw and processed EEG data (BIDS-format dataset in `dataset/`, samples for quick tests in `dataset_sample/`, extracted numpy arrays in `eeg_data/`, etc.).
- **Usage Notes**: These folders are typically large and should remain untracked; keep structure consistent for reproducible experiments.

### `docs/`
Contains all user-facing documentation:
- `how_it_works.md`, `how_to_run_v4.md`, `step_by_step_workflow.md`: process walkthroughs.
- `models_and_architectures.md`, `single_architecture_eegnet_walkthrough.md`: architectural deep dives.
- `release_notes_v4.md`: highlights of the latest release.
- (Older release note `RELEASE_v3.1.md` removed in favor of updated docs.)
- **Usage**: Update when instructions or features change so team members have a single source of truth.

### `evaluation/`
- `evaluate_models.py`: Script to compare trained models on validation/test splits, compute metrics, and optionally generate plots.
- **Usage**: Run after training to quantify improvements or regression-test new architectures.

### `models/`
| File | Purpose |
|------|---------|
| `network.py` | Collection of core EEG architectures (EEGNet, GoogLeNet-style, DenseNet-style, VGG, ResNet, RNN hybrids, DeepCNN). Provides `create_model()` factory used by training scripts. |
| `ewc.py` | Implements Elastic Weight Consolidation (EWC) for continual learning, plus `EWCTrainer` wrapper. Use when sequentially training on multiple tasks while avoiding catastrophic forgetting. |

### `scripts/`
- Generally contains automation/utility scripts (check subfolders for dataset conversion, data sanity checks, etc.). Customize as you add workflows.

### `src/`
Breaks into modular packages:
- `src/api/`
  - `server.py`: Lightweight API layer (e.g., FastAPI) for exposing inference/training endpoints separate from `app/`.
- `src/data/`
  - Dataset helpers (e.g., PyTorch Dataset implementations, data augmentation). Use when building new loaders.
- `src/models/`
  - Advanced architectures (`deep_learning_models.py` etc.) and supporting classes (configs, datasets). Suitable for experimenting with hybrid CNN-LSTM, Transformer, ResNet1D, etc.

### `training/`
| File | Purpose |
|------|---------|
| `train.py` | CLI training script: loads data via `EEGDataset`, instantiates models from `models.network`, trains with Adam/SGD/AdamW, handles schedulers, checkpoints, and early stopping. |
| `__init__.py`, support files | (If present) Additional utilities shared by training scripts. |

### `main.py`
- Umbrella script to stitch together preprocessing, training, evaluation, or inference flows. Use it as a CLI entry point or orchestrator in notebooks.

### `networks_and_models.md`
- Documentation summarizing neural network architectures, algorithms (optimizers, schedulers), and continual-learning components. Handy for onboarding.

### `requirements.txt`
- Pin list of Python packages (PyTorch, numpy, loguru, etc.). Keep updated when adding dependencies so Contributors can reproduce environments.

### `config.yaml`
- Mirror of `config/config.yaml` at the root for convenience—set default directories, hyperparameters, device configs used throughout scripts.

### `ref/`
- Store lookup tables, electrode maps, metadata JSON/CSV that remain relatively static but are needed during preprocessing and evaluation.

### `logs/`, `results/`, `checkpoints/`
- Generated artifacts. Keep directories in place so scripts have known output targets; contents are typically gitignored.

---

## How to Use This Document
1. **New Contributor Onboarding** – Hand them this file first so they can understand what each component does without reading the entire codebase.
2. **Housekeeping & Refactors** – When renaming/moving folders, update this document so the structure stays accurate.
3. **Audit** – Quickly confirm whether a given functionality already exists before writing new code.

> **Tip:** Pair this file with `networks_and_models.md` when presenting the project: this file explains *where* things live, while that one explains *what* the models are.
