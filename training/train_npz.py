from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.network import EEGNet1D


def _load_npz(npz_path: Path, signal_key: str, label_key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if signal_key not in data or label_key not in data:
        raise KeyError(
            f"NPZ file {npz_path} is missing required keys '{signal_key}' and/or '{label_key}'. "
            f"Available keys: {list(data.files)}"
        )
    X = data[signal_key].astype(np.float32)
    y = data[label_key].astype(np.int64)
    if X.ndim != 3:
        raise ValueError(f"Signals in {npz_path} must have shape (N, C, T); got {X.shape} instead")
    return X, y


class NPZEEGDataset(Dataset):
    def __init__(self, npz_path: Path, signal_key: str, label_key: str):
        self.signals, self.labels = _load_npz(npz_path, signal_key, label_key)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int):
        x = self.signals[idx]
        y = self.labels[idx]
        # Ensure tensor shape (C, T)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def train_model(train_path: Path, val_path: Path, signal_key: str, label_key: str,
                out_dir: Path, epochs: int, batch_size: int, lr: float, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = NPZEEGDataset(train_path, signal_key, label_key)
    val_ds = NPZEEGDataset(val_path, signal_key, label_key)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    in_channels = train_ds.signals.shape[1]
    model = EEGNet1D(in_channels=in_channels, num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    checkpoint_path = out_dir / "best_npz.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        val_acc = evaluate(model, val_loader, device)
        avg_loss = train_loss / max(train_total, 1)
        avg_acc = train_correct / max(train_total, 1)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.3f} - val_acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "model_kwargs": {"in_channels": in_channels, "num_classes": 2}
            }, checkpoint_path)
            print(f"Saved new best checkpoint to {checkpoint_path}")

    print(f"Training complete. Best val accuracy: {best_val_acc:.3f}")


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEGNet1D on NPZ datasets")
    parser.add_argument("--train_npz", type=Path, required=True, help="Path to training NPZ file")
    parser.add_argument("--val_npz", type=Path, required=True, help="Path to validation NPZ file")
    parser.add_argument("--signal_key", type=str, default="train_signals", help="Key for signal tensor inside NPZ")
    parser.add_argument("--label_key", type=str, default="train_labels", help="Key for labels inside NPZ")
    parser.add_argument("--out_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_model(
        train_path=args.train_npz,
        val_path=args.val_npz,
        signal_key=args.signal_key,
        label_key=args.label_key,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    main()
