from __future__ import annotations
import os
import csv
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.io_utils import load_recording
from app.preprocess import preprocess_for_model
from models.network import create_model

class EEGDataset(Dataset):
    def __init__(self, data_dir: Path, labels_csv: Path):
        self.items = []
        with open(labels_csv, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                fpath = data_dir / row['file']
                y = int(row['label'])
                self.items.append((fpath, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, y = self.items[idx]
        with open(fpath, 'rb') as fb:
            rec = load_recording(fb, fpath.name)
        X, sfreq = rec['data'], rec['sfreq']
        wins = preprocess_for_model(X, sfreq)
        if wins.size == 0:
            # fabricate empty
            wins = np.zeros((1, X.shape[0], min(1000, X.shape[1])), dtype=np.float32)
        return wins, y


def collate_batch(batch):
    # batch: list of (wins, y) with variable number of windows, stack windows and repeat labels
    xs, ys = [], []
    for wins, y in batch:
        xs.append(torch.from_numpy(wins).float())
        ys.append(torch.full((wins.shape[0],), y, dtype=torch.long))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data_dir)
    labels_csv = Path(args.labels_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Training model: {args.model}")

    ds = EEGDataset(data_dir, labels_csv)
    dl = DataLoader(ds, batch_size=args.batch_files, shuffle=True, collate_fn=collate_batch)

    # Peek to get channel count
    x0, y0 = next(iter(dl))
    in_ch = x0.shape[1]

    # Create model
    model = create_model(args.model, in_channels=in_ch, num_classes=2).to(device)
    
    # Optimizer with paper hyperparameters
    if args.optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss function
    crit = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None

    best = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))
        
        acc = correct / max(1, total)
        avg_loss = loss_sum / total
        
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} acc={acc:.3f}")
        
        # Learning rate scheduling
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(acc)
            else:
                scheduler.step()
        
        # Early stopping
        if acc > best:
            best = acc
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'model_kwargs': {'model_name': args.model, 'in_channels': in_ch, 'num_classes': 2},
                'optimizer_state': opt.state_dict(),
                'epoch': epoch,
                'accuracy': acc,
                'loss': avg_loss
            }, out_dir / f'{args.model}_best.pt')
            print(f"Saved best model with accuracy {acc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
    
    print(f"Training completed. Best accuracy: {best:.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Train EEG Seizure Prediction Models")
    ap.add_argument('--data_dir', required=True, help='Directory containing EEG data files')
    ap.add_argument('--labels_csv', required=True, help='CSV file with labels')
    ap.add_argument('--out_dir', default='models/checkpoints', help='Output directory for checkpoints')
    
    # Model selection
    ap.add_argument('--model', default='eegnet', 
                   choices=['eegnet', 'googlenet', 'densenet', 'vgg', 'resnet', 'rnn', 'deepcnn'],
                   help='Model architecture to train')
    
    # Training hyperparameters (from paper)
    ap.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    ap.add_argument('--batch_files', type=int, default=4, help='Number of recordings per batch')
    ap.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    ap.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'adamw'], help='Optimizer')
    ap.add_argument('--scheduler', default='plateau', choices=['plateau', 'cosine', 'none'], help='LR scheduler')
    ap.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold (0 to disable)')
    ap.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    args = ap.parse_args()
    train(args)
