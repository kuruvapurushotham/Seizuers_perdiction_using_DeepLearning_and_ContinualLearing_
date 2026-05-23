from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def online_finetune_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         bx: Optional[torch.Tensor], by: Optional[torch.Tensor], device: str='cpu') -> bool:
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    # Combine with rehearsal batch if provided
    if bx is not None and by is not None:
        x = torch.cat([x, bx.to(device)], dim=0)
        y = torch.cat([y, by.to(device)], dim=0)
    for _ in range(5):
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
    model.eval()
    return True


@dataclass
class RehearsalBuffer:
    capacity: int = 16
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)

    def add_batch(self, windows: np.ndarray, labels: np.ndarray):
        for w, y in zip(windows, labels):
            if len(self.xs) >= self.capacity:
                self.xs.pop(0); self.ys.pop(0)
            self.xs.append(w.astype('float32'))
            self.ys.append(int(y))

    def sample(self, k: int = 8) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.xs:
            return None, None
        idx = np.random.choice(len(self.xs), size=min(k, len(self.xs)), replace=False)
        X = torch.from_numpy(np.stack([self.xs[i] for i in idx], axis=0)).float()
        y = torch.tensor([self.ys[i] for i in idx], dtype=torch.long)
        return X, y

    def save(self, path):
        np.savez_compressed(path, X=np.array(self.xs, dtype=object), y=np.array(self.ys))

    def load(self, path):
        d = np.load(path, allow_pickle=True)
        self.xs = [x.astype('float32') for x in d['X'].tolist()]
        self.ys = [int(v) for v in d['y'].tolist()]
