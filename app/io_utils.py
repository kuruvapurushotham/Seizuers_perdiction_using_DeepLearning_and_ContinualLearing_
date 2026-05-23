from __future__ import annotations
from typing import Optional, Dict, Any
import io
import os
from pathlib import Path
import numpy as np
import mne
import pyedflib
from tempfile import NamedTemporaryFile


def _save_temp_file(file_like, suffix: str) -> str:
    data = file_like.read() if hasattr(file_like, 'read') else file_like
    tf = NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(data)
    tf.flush()
    tf.close()
    return tf.name


def _is_git_annex_placeholder(path: str) -> bool:
    """Detect git-annex pointer files (no actual data)."""
    try:
        with open(path, 'rb') as fh:
            head = fh.read(4096)
        return b'.git/annex' in head or b'annex/objects' in head
    except Exception:
        return False


def load_recording(file_like, filename: str) -> Optional[Dict[str, Any]]:
    name = filename.lower()
    if name.endswith('.edf'):
        tmp = _save_temp_file(file_like, '.edf')
        raw = mne.io.read_raw_edf(tmp, preload=True, verbose=False)
        X = raw.get_data()  # shape (ch, samples)
        return {"data": X, "sfreq": float(raw.info['sfreq']), "ch_names": raw.ch_names}
    
    if name.endswith('.eeg'):
        # Support for .eeg files (BrainVision, Neuroscan, etc.)
        tmp = None
        try:
            tmp = _save_temp_file(file_like, '.eeg')
            
            # Method 1: Try MNE's BrainVision reader (best for .eeg/.vhdr/.vmrk triplets)
            # Note: This needs the .vhdr file which user might not upload
            # We'll handle single .eeg file as best as possible
            
            # Method 2: Try reading as EDF format (some .eeg files are actually EDF)
            try:
                f = pyedflib.EdfReader(tmp)
                n = f.signals_in_file
                if n > 0:
                    sigbufs = [f.readSignal(i) for i in range(n)]
                    sfreq = f.getSampleFrequency(0)
                    labels = [f.getLabel(i) for i in range(n)]
                    f.close()
                    X = np.vstack(sigbufs)
                    return {"data": X, "sfreq": float(sfreq), "ch_names": labels}
            except Exception:
                pass
            
            # Method 3: Try MNE's auto-detection
            try:
                # MNE can sometimes infer format
                raw = mne.io.read_raw(tmp, preload=True, verbose=False)
                X = raw.get_data()
                return {"data": X, "sfreq": float(raw.info['sfreq']), "ch_names": raw.ch_names}
            except Exception:
                pass
            
            # Method 4: Try reading as raw binary (Neuroscan CNT format)
            try:
                raw = mne.io.read_raw_cnt(tmp, preload=True, verbose=False)
                X = raw.get_data()
                return {"data": X, "sfreq": float(raw.info['sfreq']), "ch_names": raw.ch_names}
            except Exception:
                pass
            
            raise RuntimeError(
                ".eeg file format not recognized. "
                "Please ensure it's in EDF, BrainVision, or Neuroscan format. "
                "For BrainVision files, you may need to upload the .vhdr and .vmrk files together."
            )
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    
    # Support for other formats
    if name.endswith('.cnt'):
        tmp = _save_temp_file(file_like, '.cnt')
        if _is_git_annex_placeholder(tmp):
            raise RuntimeError(
                "Uploaded Neuroscan .cnt is a git-annex placeholder. "
                "Download the real EEG file before uploading."
            )
        raw = mne.io.read_raw_cnt(tmp, preload=True, verbose=False)
        X = raw.get_data()
        return {"data": X, "sfreq": float(raw.info['sfreq']), "ch_names": raw.ch_names}
    
    if name.endswith('.vhdr'):
        tmp = _save_temp_file(file_like, '.vhdr')
        if _is_git_annex_placeholder(tmp):
            raise RuntimeError(
                "Uploaded BrainVision .vhdr is a git-annex placeholder. "
                "Download the real EEG files (vhdr/vmrk/eeg) before uploading."
            )
        raw = mne.io.read_raw_brainvision(tmp, preload=True, verbose=False)
        X = raw.get_data()
        return {"data": X, "sfreq": float(raw.info['sfreq']), "ch_names": raw.ch_names}
    
    return None


def parse_pdf(file_like) -> str:
    # We do not extract raw EEG from PDFs; treat as report text only
    try:
        import PyPDF2  # optional dependency not in requirements by default
        reader = PyPDF2.PdfReader(io.BytesIO(file_like.read()))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        snippet = (text[:500] + "...") if len(text) > 500 else text
        return "Parsed PDF text (not signals). Please upload .edf/.eeg for signal-based prediction.\n" + snippet
    except Exception:
        return "PDF treated as report; unable to parse text. Please upload .edf/.eeg for prediction."
