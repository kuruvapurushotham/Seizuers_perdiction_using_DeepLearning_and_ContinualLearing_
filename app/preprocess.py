from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple

import mne
import numpy as np
import pywt
from scipy import signal as sp_signal


HybridFilterName = Literal[
    "butterworth_wavelet",
    "chebyshev1_wavelet",
    "chebyshev2_wavelet",
    "elliptic_wavelet",
    "savgol_wavelet",
]


@dataclass
class HybridFilterResult:
    name: HybridFilterName
    data: np.ndarray
    snr: float


def _wavelet_denoise(data: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Channel-wise wavelet denoising with soft thresholding."""
    denoised = np.zeros_like(data)
    for ch, signal in enumerate(data):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-8
        threshold = sigma * np.sqrt(2 * np.log(signal.size + 1))
        coeffs_thresh = [pywt.threshold(c, threshold, mode="soft") if i else c for i, c in enumerate(coeffs)]
        denoised[ch] = pywt.waverec(coeffs_thresh, wavelet)[: signal.size]
    return denoised


def _bandpass(data: np.ndarray, sfreq: float, order: int, band: Tuple[float, float], kind: str) -> np.ndarray:
    low, high = band
    if kind == "butter":
        sos = sp_signal.butter(order, [low, high], btype="band", fs=sfreq, output="sos")
    elif kind == "cheby1":
        sos = sp_signal.cheby1(order, 0.5, [low, high], btype="band", fs=sfreq, output="sos")
    elif kind == "cheby2":
        sos = sp_signal.cheby2(order, 20, [low, high], btype="band", fs=sfreq, output="sos")
    elif kind == "ellip":
        sos = sp_signal.ellip(order, 0.5, 20, [low, high], btype="band", fs=sfreq, output="sos")
    else:
        raise ValueError(f"Unsupported filter kind: {kind}")
    return sp_signal.sosfiltfilt(sos, data, axis=-1)


def _apply_savgol(data: np.ndarray, sfreq: float, polyorder: int = 3) -> np.ndarray:
    window = max(5, int(0.25 * sfreq) | 1)
    return sp_signal.savgol_filter(data, window_length=window, polyorder=polyorder, axis=-1)


def _compute_snr(original: np.ndarray, filtered: np.ndarray) -> float:
    signal_power = np.mean(filtered**2, axis=-1) + 1e-12
    noise_power = np.mean((original - filtered) ** 2, axis=-1) + 1e-12
    snr = 10.0 * np.log10(signal_power / noise_power)
    return float(np.mean(snr))


def _apply_hybrid_filter(
    X: np.ndarray,
    sfreq: float,
    name: HybridFilterName,
    band: Tuple[float, float],
    notch: Optional[float],
) -> HybridFilterResult:
    data = np.asarray(X, dtype=np.float64)
    if notch:
        Q = 30.0
        w0 = notch / (sfreq / 2.0)
        b, a = sp_signal.iirnotch(w0=w0, Q=Q)
        data = sp_signal.filtfilt(b, a, data, axis=-1)

    if name == "butterworth_wavelet":
        filtered = _bandpass(data, sfreq, order=4, band=band, kind="butter")
    elif name == "chebyshev1_wavelet":
        filtered = _bandpass(data, sfreq, order=4, band=band, kind="cheby1")
    elif name == "chebyshev2_wavelet":
        filtered = _bandpass(data, sfreq, order=4, band=band, kind="cheby2")
    elif name == "elliptic_wavelet":
        filtered = _bandpass(data, sfreq, order=4, band=band, kind="ellip")
    elif name == "savgol_wavelet":
        filtered = _apply_savgol(data, sfreq)
    else:
        raise ValueError(f"Unknown hybrid filter: {name}")

    filtered = _wavelet_denoise(filtered)
    snr = _compute_snr(np.asarray(X, dtype=np.float64), filtered)
    return HybridFilterResult(name=name, data=filtered, snr=snr)


def run_hybrid_filterbank(
    X: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    notch: Optional[float],
    filters: Optional[Iterable[HybridFilterName]] = None,
) -> Dict[HybridFilterName, HybridFilterResult]:
    to_run = list(filters) if filters else [
        "butterworth_wavelet",
        "chebyshev1_wavelet",
        "chebyshev2_wavelet",
        "elliptic_wavelet",
        "savgol_wavelet",
    ]
    results: Dict[HybridFilterName, HybridFilterResult] = {}
    for name in to_run:
        results[name] = _apply_hybrid_filter(X, sfreq, name, band, notch)
    return results


def preprocess_for_model(
    X: np.ndarray,
    sfreq: float,
    band: Tuple[float, float] = (0.5, 40.0),
    notch: Optional[float] = 50.0,
    win_sec: float = 10.0,
    step_sec: float = 5.0,
    filter_mode: Literal["auto", "butterworth_wavelet", HybridFilterName] = "auto",
    return_filter_report: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """Hybrid preprocessing with SNR-based filter selection."""

    filterbank = run_hybrid_filterbank(X, sfreq, band=band, notch=notch)

    if filter_mode == "auto":
        best = max(filterbank.values(), key=lambda res: (res.snr, res.name == "butterworth_wavelet"))
    else:
        name: HybridFilterName = filter_mode  # type: ignore[assignment]
        best = filterbank[name]

    Xf = best.data
    Xf = (Xf - Xf.mean(axis=1, keepdims=True)) / (Xf.std(axis=1, keepdims=True) + 1e-6)

    win = int(win_sec * sfreq)
    step = int(step_sec * sfreq)
    if Xf.shape[1] < win:
        windows = np.empty((0, Xf.shape[0], 0), dtype=np.float32)
    else:
        starts = np.arange(0, Xf.shape[1] - win + 1, step)
        windows = np.stack([Xf[:, s : s + win] for s in starts], axis=0).astype(np.float32)

    if return_filter_report:
        filter_report = {name: res.snr for name, res in filterbank.items()}
        return windows, filter_report
    return windows


def spectral_entropy(x: np.ndarray, sfreq: float) -> float:
    psd, freqs = mne.time_frequency.psd_array_welch(x, sfreq=sfreq, fmin=0.5, fmax=40.0, n_fft=1024, verbose=False)
    psd = psd.sum(axis=0)
    p = psd / (psd.sum() + 1e-9)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def line_length(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(x, axis=-1))))


def simple_heuristic_score(windows: np.ndarray, sfreq: float, return_per_window: bool = False):
    """Return [0,1] score indicating seizure likelihood from windows.
    Uses line-length and spectral entropy. Looks for ANY window with seizure-like activity."""
    if windows.size == 0:
        if return_per_window:
            return 0.0, np.array([], dtype=float)
        return 0.0
    
    scores = []
    for w in windows:  # (C, T)
        ll = line_length(w)
        se = spectral_entropy(w, sfreq)
        
        # Seizures have: high line length (activity) AND low entropy (rhythmic)
        # Normalize line length: typical normal ~0.5-2.0, seizure ~5-15
        ll_score = np.clip(ll / 10.0, 0.0, 1.0)
        
        # Low entropy (rhythmic) is seizure-like
        # Normal entropy ~0.7-0.9, seizure ~0.3-0.6
        se_score = np.clip((0.9 - se) / 0.6, 0.0, 1.0)
        
        # Combined score: both must be high
        window_score = ll_score * 0.7 + se_score * 0.3
        scores.append(window_score)

    scores_arr = np.array(scores, dtype=float)

    # Use percentile instead of mean - if top 10% of windows show seizure, flag it
    top_percentile = np.percentile(scores_arr, 90)

    # Also check if many windows exceed threshold
    high_score_count = np.sum(scores_arr > 0.3)
    high_score_ratio = high_score_count / len(scores_arr)

    # Final score: max of (top percentile, high score ratio)
    final_score = max(top_percentile, high_score_ratio)
    final_score = float(np.clip(final_score, 0.0, 1.0))

    if return_per_window:
        return final_score, scores_arr
    return final_score
