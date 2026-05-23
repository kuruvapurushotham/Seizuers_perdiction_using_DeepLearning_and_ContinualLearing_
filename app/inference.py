from __future__ import annotations
import os
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from preprocess import simple_heuristic_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.network import EEGNet1D
from training.continual import online_finetune_step, RehearsalBuffer

CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "best.pt"
BUFFER_PATH = Path(__file__).resolve().parents[1] / "models" / "buffer.npz"

class InferenceEngine:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EEGNet1D(in_channels=19)  # will adapt at runtime
        self.model_loaded = False
        self.use_heuristic = True  # Force heuristic mode for now
        self.buffer = RehearsalBuffer(capacity=8)
        if BUFFER_PATH.exists():
            try:
                self.buffer.load(BUFFER_PATH)
            except Exception:
                pass
        # self._try_load_checkpoint()  # Disable model loading

    def _try_load_checkpoint(self):
        if CHECKPOINT.exists():
            try:
                ckpt = torch.load(CHECKPOINT, map_location=self.device)
                model_kwargs = ckpt.get("model_kwargs", {})
                in_channels = model_kwargs.get("in_channels", 19)
                self.model = EEGNet1D(**model_kwargs)
                self.model.load_state_dict(ckpt["state_dict"])
                self.model.to(self.device).eval()
                self.model_loaded = True
                self.trained_channels = in_channels
                print(f"[INFO] Model loaded: {in_channels} channels, model_loaded={self.model_loaded}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                self.model_loaded = False
                self.trained_channels = None

    def _ensure_channel_match(self, X: np.ndarray) -> bool:
        """Check if channels match trained model. Return True if OK to use model."""
        c = X.shape[1]
        if hasattr(self, 'trained_channels') and self.trained_channels is not None:
            if c == self.trained_channels:
                return True  # Perfect match, use model
            else:
                print(f"[WARNING] Channel mismatch: data has {c}, model trained on {self.trained_channels}")
                return False  # Don't use model with wrong channels
        return False  # No trained model

    def predict(self, windows: np.ndarray, sfreq: float) -> Dict[str, Any]:
        # windows: (N, C, T)
        can_use_model = self._ensure_channel_match(windows)
        
        if self.model_loaded and can_use_model:
            print(f"[INFO] Using trained model for prediction")
            with torch.no_grad():
                x = torch.from_numpy(windows).float().to(self.device)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            # Use max probability (any window with seizure) instead of mean
            # This is better for detecting seizures in long recordings
            prob_max = float(np.clip(probs.max(), 0.0, 1.0))
            prob_mean = float(np.clip(probs.mean(), 0.0, 1.0))
            
            # If ANY window has high seizure probability, flag as seizure
            # Count how many windows exceed threshold
            high_prob_windows = np.sum(probs > 0.5)
            
            # Use adaptive threshold: if >5% of windows show seizure, flag it
            prob = prob_max if high_prob_windows > len(probs) * 0.05 else prob_mean
            label = int(prob >= 0.5)
            
            print(f"[INFO] Model prediction: max_prob={prob_max:.3f}, mean_prob={prob_mean:.3f}, final_prob={prob:.3f}, label={label}, high_prob_windows={high_prob_windows}/{len(probs)}")
            return {"prob": prob, "label": label, "model_loaded": True}
        
        # Fallback heuristic
        print(f"[WARNING] Using heuristic (model_loaded={self.model_loaded}, can_use_model={can_use_model})")
        score = simple_heuristic_score(windows, sfreq)
        prob = float(np.clip(score, 0.0, 1.0))
        label = int(prob >= 0.65)
        print(f"[INFO] Heuristic prediction: prob={prob:.3f}, label={label}")
        return {"prob": prob, "label": label, "model_loaded": False}

    def online_update(self, windows: np.ndarray, label: int) -> bool:
        # Only possible if we already have a trained checkpoint
        if not self.model_loaded:
            return False
        x = torch.from_numpy(windows).float().to(self.device)
        y = torch.full((windows.shape[0],), int(label), dtype=torch.long, device=self.device)
        # Add a rehearsal sample if available
        bx, by = self.buffer.sample()
        updated = online_finetune_step(self.model, x, y, bx, by, device=self.device)
        # Update buffer with a small subset
        self.buffer.add_batch(windows[::max(1, len(windows)//2)], np.full((len(windows[::max(1, len(windows)//2)]),), label))
        try:
            self.buffer.save(BUFFER_PATH)
        except Exception:
            pass
        # Save updated weights
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "model_kwargs": {"in_channels": windows.shape[1], "num_classes": 2}
        }, CHECKPOINT)
        self.model_loaded = True
        return updated
