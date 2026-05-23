"""Streamlit v4.0 hybrid seizure prediction dashboard."""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import load_recording
from models.network import create_model
from models.ewc import EWCTrainer
from preprocess import preprocess_for_model, simple_heuristic_score


APP_VERSION = "v4.0"
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best.pt"


st.set_page_config(page_title="Seizure Prediction Dashboard", page_icon="", layout="wide")

CSS = """
<style>
main .block-container { max-width: 1150px; }
.card { background:#ffffff; border-radius:14px; padding:18px 22px; box-shadow:0 8px 24px rgba(11,31,53,0.08); border:1px solid rgba(11,31,53,0.05); }
.title { color:#0b0f14; font-size:40px; font-weight:800; text-align:center; margin-bottom:0; }
.subtitle { text-align:center; color:#5d6c78; font-size:15px; margin-bottom:6px; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#3b4bff; font-size:13px; font-weight:600; }
.metric-pill { border-radius:12px; background:#f5f7fb; padding:10px 14px; margin-bottom:8px; }
.muted { color:#55606c; font-size:0.9rem; }
.section-title { font-size:20px; font-weight:700; margin-bottom:8px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


NPZ_SIGNAL_KEYS = [
    "signals",
    "data",
    "train_signals",
    "val_signals",
    "windows",
    "eeg",
]


def _load_npz_recording(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=True) as npz:
        key = next((k for k in NPZ_SIGNAL_KEYS if k in npz.files), None)
        if key is None:
            key = next((k for k in npz.files if npz[k].ndim >= 2), None)
        if key is None:
            raise ValueError("NPZ file does not contain any array with at least 2 dimensions.")
        arr = np.array(npz[key])
        sfreq = float(npz.get("sfreq", npz.get("sampling_rate", 256.0)))
        ch_names = list(npz.get("ch_names", []))

    if arr.ndim == 3:
        n, a, b = arr.shape
        # Heuristic to decide which axis is channels
        if a <= 64 and b >= 64:
            raw = arr.transpose(1, 0, 2).reshape(a, n * b)
            channels = a
        elif b <= 64 and a >= 64:
            raw = arr.transpose(2, 0, 1).reshape(b, n * a)
            channels = b
        else:
            raw = arr.transpose(1, 0, 2).reshape(a, n * b)
            channels = a
    elif arr.ndim == 2:
        # Interpret as (channels, samples)
        channels = arr.shape[0]
        raw = arr
    else:
        raise ValueError(f"Unsupported NPZ array shape: {arr.shape}")

    raw = raw.astype(np.float32)
    if not ch_names or len(ch_names) != channels:
        ch_names = None

    return {"data": raw, "sfreq": sfreq, "ch_names": ch_names}


def detect_seizure_absolute(data: np.ndarray, sfreq: float) -> dict:
    if data.size == 0 or data.shape[1] < sfreq * 10:
        return {"label": 0, "prob": 0.0, "confidence": "N/A", "seizure_windows": 0, "total_windows": 0}

    win_sec = 10.0
    step_sec = 5.0
    win_samples = int(win_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    starts = np.arange(0, data.shape[1] - win_samples + 1, step_samples)

    seizure_scores = []
    high_activity_windows = 0

    AMP_TH = 9.0e-5
    LL_TH = 1.8e-5
    COMBINED_TH = 0.75

    for start in starts:
        window = data[:, start:start + win_samples]
        amp_std = np.std(window)
        line_length = np.mean(np.abs(np.diff(window, axis=1)))
        p2p = np.mean(np.ptp(window, axis=1))

        amp_score = min(amp_std / AMP_TH, 1.5) / 1.5
        ll_score = min(line_length / LL_TH, 1.5) / 1.5
        p2p_score = min(p2p / 5e-4, 1.5) / 1.5
        combined_score = 0.4 * amp_score + 0.4 * ll_score + 0.2 * p2p_score

        seizure_scores.append(combined_score)
        if combined_score > COMBINED_TH:
            high_activity_windows += 1

    scores_array = np.array(seizure_scores)
    high_mask = scores_array > COMBINED_TH
    max_consecutive = 0
    current = 0
    for flag in high_mask:
        if flag:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0

    high_ratio = high_activity_windows / max(len(scores_array), 1)
    max_score = float(np.max(scores_array))
    percentile_95 = float(np.percentile(scores_array, 95))

    if max_consecutive >= 3 and high_ratio > 0.003:
        label = 1
        prob = min(percentile_95 * 1.2, 1.0)
        confidence = "High" if max_consecutive >= 5 else "Medium"
    elif max_consecutive >= 2 and max_score > 0.85:
        label = 1
        prob = max_score * 0.9
        confidence = "Low"
    else:
        label = 0
        prob = min(max_score * 0.5, 0.5)
        confidence = "High"

    return {
        "label": label,
        "prob": float(np.clip(prob, 0.0, 1.0)),
        "confidence": confidence,
        "seizure_windows": int(high_activity_windows),
        "total_windows": int(len(scores_array)),
        "max_consecutive": int(max_consecutive),
        "max_score": max_score,
        "percentile_95": percentile_95,
        "timeline": scores_array
    }


def load_ewc_model(device: torch.device, model_name: str = "eegnet", in_channels: int = 19):
    """Load EWC-trained model if available"""
    ewc_checkpoint_path = CHECKPOINT_DIR / f"{model_name}_ewc_best.pt"
    if not ewc_checkpoint_path.exists():
        return None
    
    try:
        ckpt = torch.load(ewc_checkpoint_path, map_location=device, weights_only=False)
        model_kwargs = ckpt.get("model_kwargs", {})
        actual_in_channels = model_kwargs.get("in_channels", in_channels)
        num_classes = model_kwargs.get("num_classes", 2)
        model = create_model(model_name, in_channels=actual_in_channels, num_classes=num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return {"model": model, "kwargs": model_kwargs, "model_name": f"{model_name}_ewc"}
    except Exception as e:
        st.error(f"Error loading EWC model: {e}")
        return None


def load_dl_checkpoint(device: torch.device, model_name: str = "eegnet", in_channels: int = 19):
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        # Fallback to default checkpoint
        if not DEFAULT_CHECKPOINT.exists():
            return None
        checkpoint_path = DEFAULT_CHECKPOINT
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_kwargs = ckpt.get("model_kwargs", {})
        
        # Get model name from checkpoint if available
        if "model_kwargs" in ckpt and "model_name" in ckpt["model_kwargs"]:
            model_name = ckpt["model_kwargs"]["model_name"]
        
        # Create model with the correct architecture and actual channels
        actual_in_channels = model_kwargs.get("in_channels", in_channels)
        num_classes = model_kwargs.get("num_classes", 2)
        model = create_model(model_name, in_channels=actual_in_channels, num_classes=num_classes)
        
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return {"model": model, "kwargs": model_kwargs, "model_name": model_name}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def run_ewc_inference(windows: np.ndarray, device: torch.device, model_name: str = "eegnet", in_channels: int = 19) -> dict | None:
    """Run inference with EWC-trained model"""
    if windows.size == 0:
        return None
    try:
        ewc_data = load_ewc_model(device, model_name, in_channels)
        if ewc_data is None:
            return None
        model = ewc_data["model"]
        with torch.no_grad():
            x = torch.from_numpy(windows).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        summary = {
            "probs": probs,
            "max": float(probs.max(initial=0.0)),
            "mean": float(probs.mean()) if probs.size else 0.0,
            "count": int(np.sum(probs > 0.5)),
            "model_name": ewc_data.get("model_name", f"{model_name}_ewc"),
            "model_type": "EWC"
        }
        return summary
    except Exception as e:
        st.error(f"Error running EWC inference with {model_name}: {e}")
        return None


def run_dl_inference(windows: np.ndarray, device: torch.device, model_name: str = "eegnet", in_channels: int = 19) -> dict | None:
    if windows.size == 0:
        return None
    try:
        ckpt_data = load_dl_checkpoint(device, model_name, in_channels)
        if ckpt_data is None:
            return None
        model = ckpt_data["model"]
        with torch.no_grad():
            x = torch.from_numpy(windows).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        summary = {
            "probs": probs,
            "max": float(probs.max(initial=0.0)),
            "mean": float(probs.mean()) if probs.size else 0.0,
            "count": int(np.sum(probs > 0.5)),
            "model_name": ckpt_data.get("model_name", model_name),
            "model_type": "Standard"
        }
        return summary
    except Exception as e:
        st.error(f"Error running inference with {model_name}: {e}")
        return None


def smooth_curve(values: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or values.size == 0:
        return values
    return uniform_filter1d(values, size=win)


def build_probability_chart(timelines: dict, threshold: float):
    fig = go.Figure()
    for name, data in timelines.items():
        fig.add_trace(go.Scatter(y=data["values"], name=name, mode="lines"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold {threshold:.2f}")
    fig.update_layout(template="plotly_white", height=260, margin=dict(t=20, b=20))
    return fig


def compute_accuracy_precision(reference: np.ndarray, candidate: np.ndarray, threshold: float) -> tuple[float, float] | None:
    n = min(len(reference), len(candidate))
    if n == 0:
        return None
    ref_bin = reference[:n] > threshold
    cand_bin = candidate[:n] > threshold
    tp = np.sum(np.logical_and(cand_bin, ref_bin))
    tn = np.sum(np.logical_and(~cand_bin, ~ref_bin))
    fp = np.sum(np.logical_and(cand_bin, ~ref_bin))
    fn = np.sum(np.logical_and(~cand_bin, ref_bin))
    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    return float(accuracy), float(precision)


def plot_eeg_waves(data: np.ndarray, sfreq: float, ch_names: list[str] | None, duration: float = 10.0):
    n_channels, n_samples = data.shape
    plot_samples = int(min(duration, n_samples / sfreq) * sfreq)
    time_axis = np.arange(plot_samples) / sfreq
    fig = make_subplots(rows=min(n_channels, 8), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=[ch_names[i] if ch_names else f"Ch {i+1}" for i in range(min(n_channels, 8))])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for idx in range(min(n_channels, 8)):
        fig.add_trace(go.Scatter(x=time_axis, y=data[idx, :plot_samples], line=dict(color=colors[idx % len(colors)], width=1),
                                 showlegend=False), row=idx + 1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=min(n_channels, 8), col=1)
    fig.update_layout(height=720, template='plotly_white')
    return fig


def make_pdf(buffer: io.BytesIO, filename: str, sfreq: float, probs: np.ndarray, threshold: float,
             windows: np.ndarray, timeline_name: str, notes: str):
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.suptitle("Seizure Prediction Clinical Report", fontsize=14)
        plt.subplot(3, 1, 1)
        plt.text(0.01, 0.85, f"File: {filename}", fontsize=10)
        plt.text(0.01, 0.77, f"Timeline: {timeline_name}", fontsize=10)
        plt.text(0.01, 0.69, f"Windows analyzed: {len(probs)}", fontsize=10)
        plt.text(0.01, 0.61, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", fontsize=10)

        plt.subplot(3, 1, 2)
        plt.plot(probs, color='tab:blue')
        plt.axhline(threshold, linestyle='--', color='tab:red')
        plt.ylabel('Probability')
        plt.title('Probability Timeline')

        plt.subplot(3, 1, 3)
        top_idx = np.argsort(probs)[-6:][::-1]
        for i, idx in enumerate(top_idx):
            if idx < len(windows):
                plt.plot(windows[idx, 0], label=f"Window {idx}")
        plt.title('Top suspicious windows (channel 0 preview)')
        plt.legend(fontsize=7)
        pdf.savefig(fig)
        plt.close(fig)

        fig2 = plt.figure(figsize=(8.27, 11.69))
        plt.suptitle('Clinician Notes', fontsize=12)
        plt.axis('off')
        plt.text(0.03, 0.9, notes or "No notes provided.", fontsize=11)
        pdf.savefig(fig2)
        plt.close(fig2)


st.markdown('<div class="card"><div class="title">SEIZURE PREDICTION DASHBOARD</div></div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Controls")
    
    # Only EWC models are available in the UI (removed standard model option)
    model_types = ["Continual Learning (EWC)"]
    model_type = st.selectbox("Model Type", model_types)

    # Available model architectures (EWC selection)
    available_models = ['eegnet', 'googlenet', 'densenet', 'vgg', 'resnet', 'rnn', 'deepcnn']
    selected_model = st.selectbox("Select EWC Model", available_models, index=0)
    use_ewc = True
    
    uploaded = st.file_uploader(
        "Upload EEG (EDF/EEG/CNT/VHDR/NPZ)",
        type=["edf", "eeg", "cnt", "vhdr", "npz"],
        accept_multiple_files=False,
        help="NPZ files should contain windows shaped (N, C, T) or (N, T, C)",
    )
    clinician_notes = st.text_area("Clinician notes", height=140)

if uploaded is None:
    st.info("Upload an EEG file to begin analysis.")
    st.stop()

try:
    if uploaded.name.lower().endswith('.npz'):
        recording = _load_npz_recording(uploaded.getvalue())
    else:
        recording = load_recording(uploaded, uploaded.name)
except Exception as exc:
    st.error(f"Unable to read file: {exc}")
    st.stop()

if recording is None:
    st.error("Unsupported file format. Please upload EDF/EEG/CNT/VHDR.")
    st.stop()

raw_data = recording["data"]
sfreq = float(recording["sfreq"])
ch_names = recording.get("ch_names")

with st.spinner("Preprocessing signal and running detectors..."):
    abs_result = detect_seizure_absolute(raw_data, sfreq)
    windows = preprocess_for_model(raw_data, sfreq)
    heuristic_score, heuristic_windows = simple_heuristic_score(windows, sfreq, return_per_window=True)
    heuristic_windows = heuristic_windows if heuristic_windows.size else np.zeros(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get actual number of channels from the data
    actual_channels = raw_data.shape[0]
    
    # Use EWC or standard model based on selection
    if use_ewc:
        dl_result = run_ewc_inference(windows, device, selected_model, actual_channels)
    else:
        dl_result = run_dl_inference(windows, device, selected_model, actual_channels)

timeline_source = None
timeline_values = None
if dl_result:
    timeline_source = "Deep Learning"
    timeline_values = dl_result["probs"]
else:
    timeline_source = "Heuristic"
    timeline_values = heuristic_windows

seizure_detected = bool(abs_result["label"] or (dl_result and dl_result["count"] > 0))

st.markdown('<div class="card">', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("### Session")
    st.write(f"**File:** `{uploaded.name}`")
    duration = raw_data.shape[1] / sfreq
    st.write(f"Channels: {raw_data.shape[0]} · Duration: {duration/60:.1f} min")
    st.write(f"Windows (10s): {abs_result['total_windows']}")
with col_b:
    st.markdown("### Threshold Detector")
    st.write(f"Result: {'⚠️ Seizure' if abs_result['label'] else '✔️ No seizure'}")
    st.write(f"Confidence: {abs_result['confidence']}")
    st.write(f"Max score: {abs_result['max_score']:.3f}")
with col_c:
    st.markdown("### Deep Learning")
    if dl_result:
        model_type = dl_result.get("model_type", "Standard")
        st.write(f"**Type:** {model_type}")
        st.write(f"**Model:** {dl_result.get('model_name', selected_model).title()}")
        st.write(f"Max prob: {dl_result['max']:.3f}")
        st.write(f"Mean prob: {dl_result['mean']:.3f}")
        st.write(f"Risk windows (>0.5): {dl_result['count']}")
    else:
        # Only EWC mode is available in the UI now
        st.warning(f"EWC model checkpoint not found for {selected_model}. Showing heuristic timeline only.")
st.markdown('</div>', unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("EEG Viewer")
    show_waves = st.checkbox("Show stacked waves", value=True)
    if show_waves:
        st.plotly_chart(plot_eeg_waves(raw_data, sfreq, ch_names), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Probability Timeline")
    timelines = {timeline_source: {"values": timeline_values}}
    if dl_result and heuristic_windows.size:
        timelines["Heuristic"] = {"values": heuristic_windows}
    st.plotly_chart(build_probability_chart(timelines, 0.5), use_container_width=True)
    hm = timeline_values[np.newaxis, :]
    heatmap = px.imshow(hm, aspect='auto', color_continuous_scale='Reds')
    heatmap.update_layout(template='plotly_white', height=90, margin=dict(t=4, b=4))
    heatmap.update_yaxes(showticklabels=False)
    st.plotly_chart(heatmap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

reference_timeline = heuristic_windows if heuristic_windows.size else abs_result['timeline']
metrics_rows = []
threshold_metrics = compute_accuracy_precision(reference_timeline, abs_result['timeline'], 0.5)
if threshold_metrics:
    metrics_rows.append({"Detector": "Threshold", "Accuracy": threshold_metrics[0], "Precision": threshold_metrics[1]})
if dl_result:
    dl_metrics = compute_accuracy_precision(reference_timeline, dl_result['probs'], 0.5)
    if dl_metrics:
        metrics_rows.append({"Detector": "Deep Learning", "Accuracy": dl_metrics[0], "Precision": dl_metrics[1]})
if metrics_rows:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Detector Accuracy & Precision")
    metrics_df = pd.DataFrame(metrics_rows)
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(x=metrics_df['Detector'], y=metrics_df['Accuracy'], name='Accuracy'))
    fig_metrics.add_trace(go.Bar(x=metrics_df['Detector'], y=metrics_df['Precision'], name='Precision'))
    fig_metrics.update_layout(barmode='group', template='plotly_white', height=320, yaxis=dict(tickformat='.0%'))
    st.plotly_chart(fig_metrics, use_container_width=True)
    st.dataframe(metrics_df.style.format({'Accuracy': '{:.2%}', 'Precision': '{:.2%}'}), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Top Suspicious Windows")
if timeline_values.size:
    topk = min(8, timeline_values.size)
    top_idx = np.argsort(timeline_values)[-topk:][::-1]
    cols = st.columns(4)
    for i, idx in enumerate(top_idx):
        with cols[i % 4]:
            st.markdown(f"**Window {idx}** · p={timeline_values[idx]:.3f}")
            if idx < len(windows):
                fig = go.Figure()
                for ch in range(min(windows.shape[1], 3)):
                    fig.add_trace(go.Scatter(y=windows[idx, ch] + ch * 3, mode='lines', showlegend=False))
                fig.update_layout(height=120, margin=dict(l=0, r=0, t=4, b=4), template='plotly_white', xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No windows available for visualization.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Spectrogram")
window_idx = st.number_input("Window index", min_value=0, max_value=max(len(windows) - 1, 0), value=min(len(windows) // 2, len(windows) - 1), step=1)
channel_idx = st.number_input("Channel", min_value=0, max_value=max(raw_data.shape[0] - 1, 0), value=0, step=1)
if len(windows):
    fig_sp, ax = plt.subplots(figsize=(10, 3))
    ax.specgram(windows[window_idx, channel_idx], NFFT=128, Fs=sfreq, noverlap=96, cmap='magma')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    st.pyplot(fig_sp)
    plt.close(fig_sp)
else:
    st.info("Spectrogram unavailable without windows.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Clinician Notes & Export")
df_predictions = pd.DataFrame({"window_index": np.arange(timeline_values.size), "probability": timeline_values})
st.download_button("Download predictions (CSV)", df_predictions.to_csv(index=False).encode(), file_name=f"predictions_{int(time.time())}.csv", mime="text/csv")

if st.button("Generate PDF report"):
    with st.spinner("Composing PDF..."):
        pdf_buffer = io.BytesIO()
        make_pdf(pdf_buffer, uploaded.name, sfreq, timeline_values, 0.5, windows, timeline_source, clinician_notes)
        pdf_buffer.seek(0)
        st.download_button("Download PDF", pdf_buffer.read(), file_name=f"seizure_report_{int(time.time())}.pdf", mime="application/pdf")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
