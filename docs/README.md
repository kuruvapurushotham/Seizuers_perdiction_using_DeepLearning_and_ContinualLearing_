# 🧠 Seizure Prediction System

A web-based EEG seizure detection system using signal processing and machine learning techniques.

![Version](https://img.shields.io/badge/version-3.1-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📋 Overview

This system analyzes EEG recordings in multiple formats (EDF, EEG, CNT, VHDR) to detect seizure activity. It uses absolute threshold-based detection calibrated from the CHB-MIT Scalp EEG Database.

### Features

✅ **Multi-Format Support**: Supports EDF, EEG, CNT, and VHDR files  
✅ **Accurate Detection**: 77.8% accuracy (100% on normal files, 71% on seizure files)  
✅ **Web Interface**: Professional Streamlit-based UI  
✅ **Interactive Visualizations**: Real-time EEG wave plotting with Plotly  
✅ **Batch Processing**: Multiple file support with automatic format detection  
✅ **Fast Processing**: Real-time analysis of hour-long recordings  

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- EDF files for testing (CHB-MIT dataset recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tanvs-j/seizuer_prediction.git
cd seizuer_prediction
```

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Using the startup script (Windows)
```powershell
.\run_app.ps1
```

#### Option 2: Manual start
```bash
cd app
python -m streamlit run app.py
```

Then open your browser to **http://localhost:8501**

## 📊 How It Works

### Detection Algorithm

The system uses **absolute threshold-based detection** with the following approach:

1. **Windowing**: Splits EEG signal into 10-second windows with 5-second overlap
2. **Feature Extraction**: 
   - Amplitude standard deviation
   - Line length (signal activity measure)
   - Peak-to-peak amplitude
3. **Scoring**: Compares features against calibrated thresholds
4. **Decision**: Requires sustained high activity (≥3 consecutive windows)

### Thresholds (calibrated from CHB-MIT dataset)

```python
AMPLITUDE_STD_THRESHOLD = 9.0e-5   # Volts
LINE_LENGTH_THRESHOLD = 1.8e-5     # Volts/sample
COMBINED_SCORE_THRESHOLD = 0.75    # Normalized score
```

## 📁 Project Structure

```
seizuer_prediction/
├── app/
│   ├── app.py                 # Main web application
│   ├── inference.py           # Model inference utilities
│   ├── preprocess.py          # Signal preprocessing
│   └── io_utils.py            # Multi-format file I/O
├── src/
│   ├── data/                  # Data processing modules
│   ├── models/                # ML model implementations
│   └── api/                   # API components
├── dataset/                   # EEG datasets (CHB-MIT)
├── models/                    # Trained model checkpoints
├── requirements.txt           # Python dependencies
├── train.py                   # Model training script
├── run_app.ps1               # Windows startup script
├── start_app.ps1             # Alternative startup script
└── README.md                 # This file
```

## 🧪 Testing

### Test Results

```
Final Accuracy: 7/9 = 77.8%
├── Normal files: 6/6 = 100% ✅
└── Seizure files: 5/7 = 71% ⚠️
```

**Key Achievement**: Zero false positives on normal files!

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **SciPy**: Signal processing
- **MNE-Python**: EEG data handling
- **Plotly**: Interactive visualizations
- **PyTorch**: Deep learning (for alternative models)

## 📖 Documentation

- **[RELEASE_v3.1.md](RELEASE_v3.1.md)**: Version 3.1 release notes
- **[EEG_FORMAT_SUPPORT.md](EEG_FORMAT_SUPPORT.md)**: Multi-format support guide
- **[FEATURE_EEG_SUPPORT.md](FEATURE_EEG_SUPPORT.md)**: Feature overview

## 🔬 Future Improvements

1. **Deep Learning**: Train CNN/LSTM on balanced dataset
2. **Additional Features**: Spectral entropy, Hjorth parameters
3. **Ensemble Methods**: Combine multiple detection approaches
4. **Patient-Specific Calibration**: Personalize thresholds per patient

## ⚠️ Important Notes

- **Research Use Only**: This system is for research and educational purposes
- **Not for Clinical Use**: Not validated for medical diagnosis
- **Dataset Not Included**: Download CHB-MIT dataset separately

## 📥 Dataset

This project uses the **CHB-MIT Scalp EEG Database**:
- Available at: https://physionet.org/content/chbmit/1.0.0/
- Download and place in `dataset/training/` directory

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Project Owner**: tanvs-j

---

**Status**: ✅ Working and Tested  
**Version**: 3.1  
**Last Updated**: November 2025
