# Models and Architectures (v4)

This project implements several deep learning architectures for EEG-based seizure prediction, all adapted to 1D time-series input.

## Shared Conventions
- **Input shape**: `(batch, channels, samples)` (PyTorch `N, C, T`)
- **Channels**: Typically 19–23 EEG channels, inferred at runtime from data
- **Outputs**: Binary classification (`seizure` vs `non-seizure`, `num_classes=2`)
- **Factory**: All models are created via `models.network.create_model(name, in_channels, num_classes)`
- **Checkpoints**: Stored in `models/checkpoints/{model_name}_best.pt`

Available model names for `create_model` and CLI/Streamlit:
- `eegnet`
- `googlenet`
- `densenet`
- `vgg`
- `resnet`
- `rnn`
- `deepcnn`

---

## EEGNet1D (`eegnet`)
Lightweight 1D CNN baseline for EEG.

- Stack of Conv1d + BatchNorm1d + ReLU + MaxPool1d blocks
- Ends with AdaptiveAvgPool1d + Linear classifier
- Designed for fast training and low parameter count

Typical use:
- Good default model
- Robust starting point for experimentation

---

## EEGGoogLeNet (`googlenet`)
GoogLeNet-style architecture using 1D Inception modules (`InceptionModule1D`).

- Initial Conv1d stem
- Several Inception 1D blocks with parallel branches:
  - 1×1 conv
  - 1×1 → 3×3 conv
  - 1×1 → 5×5 conv
  - MaxPool → 1×1 conv
- Concatenation along channel dimension
- AdaptiveAvgPool1d + Linear classifier

Pros:
- Multi-scale temporal receptive fields
- Captures both short and long EEG patterns

---

## EEGDenseNet (`densenet`)
Simplified DenseNet-inspired CNN for 1D EEG.

- Initial Conv1d + BatchNorm1d + ReLU + MaxPool1d
- Two Dense-like blocks (no complex concatenation across many layers):
  - Block1: Conv1d→BN→ReLU→Conv1d→BN→ReLU (64→128 channels)
  - Block2: Conv1d→BN→ReLU→Conv1d→BN→ReLU (128→256 channels)
- AdaptiveAvgPool1d + Linear classifier (256 → 2)

Notes:
- Architecture simplified to avoid BatchNorm channel mismatches
- Still increases channel depth progressively like DenseNet

---

## EEGVGG (`vgg`)
VGG-style deep CNN using 1D convolutions.

- Multiple "VGG blocks" of:
  - Conv1d→ReLU
  - Conv1d→ReLU
  - MaxPool1d
- Channel depth increases per block (e.g. 64 → 128 → 256 → 512)
- Final AdaptiveAvgPool1d + Linear classifier

Pros:
- Strong baseline for vision-style deep CNN behavior on 1D EEG
- Deeper than `eegnet`, good capacity for complex patterns

---

## EEGResNet (`resnet`)
Residual CNN for 1D signals, using `ResidualBlock1D`.

- Initial Conv1d + BN + ReLU
- Several residual blocks with skip connections:
  - `y = F(x);  out = y + shortcut(x)`
  - Optional downsampling in shortcut when channel/stride changes
- Global pooling + Linear classifier

Pros:
- Easier optimization in deep networks due to residual connections
- Good for long training and larger datasets

---

## EEGRNN (`rnn`)
Recurrent model for temporal modeling over EEG windows.

- Convolutional feature extractor front-end (1D convs)
- Followed by an RNN layer (e.g. GRU/LSTM-like) over temporal dimension
- Final Linear layer for classification

Pros:
- Explicit temporal modeling beyond fixed receptive fields
- Potentially better for long-range preictal patterns

---

## DeepCNN1D (`deepcnn`)
Deeper pure CNN baseline.

- Multiple Conv1d + BN + ReLU + Pool blocks
- Gradual increase in channels, similar to deep computer-vision CNNs
- AdaptiveAvgPool1d + Linear classifier

Use cases:
- More capacity than `eegnet` but simpler than ResNet/VGG

---

## Continual Learning (EWC)
Continual learning is implemented in `models/ewc.py`:

- `EWC` class:
  - Computes Fisher Information Matrix on a task
  - Stores parameter means and importance
  - Adds EWC regularization term to the loss

- `EWCTrainer` class:
  - Wraps a base model from `create_model`
  - Trains sequential tasks with EWC penalty
  - Provides utilities for evaluation and summary statistics

EWC models are saved as:
- `models/checkpoints/{model_name}_ewc_best.pt`

These can be selected in the Streamlit app under:
- **Model Type** → "Continual Learning (EWC)"

---

## Model Creation Summary

```python
from models.network import create_model

model = create_model(
    name="resnet",        # one of: eegnet, googlenet, densenet, vgg, resnet, rnn, deepcnn
    in_channels=23,        # inferred from data in training / app
    num_classes=2,
)
```

The training script and Streamlit app automatically infer `in_channels` from the EEG data and pass the correct value into `create_model` when loading checkpoints.
