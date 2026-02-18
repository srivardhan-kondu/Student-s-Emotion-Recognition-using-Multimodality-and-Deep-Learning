# Student's Emotion Recognition using Multimodality and Deep Learning

An AI-powered system that detects student emotions using facial expressions, speech signals, and textual data through multimodal deep learning.

## 🎯 Overview

This system combines three emotion recognition modalities:
- **Facial Recognition** - CNN-based emotion detection from images
- **Speech Analysis** - LSTM/CNN emotion detection from audio
- **Text Analysis** - BERT-based sentiment analysis from text

**Supported Emotions:** Happy, Sad, Angry, Neutral, Fear, Surprise

## ✨ Features

- ✅ Multi-modal emotion detection (facial + speech + text)
- ✅ Adaptive fusion strategies (weighted, adaptive, voting)
- ✅ Interactive web dashboard with real-time visualization
- ✅ Individual and combined modality predictions
- ✅ Comprehensive training and evaluation tools
- ✅ Docker support for easy deployment
- ✅ Complete documentation and guides

## 🚀 Quick Start

```bash
# Automated setup
python setup_environment.py

# Activate environment
source activate.sh  # or activate.bat on Windows

# Download datasets
python download_datasets.py

# Train models
python src/facial_recognition/train.py
python src/speech_analysis/train.py
python src/text_analysis/train.py

# Run dashboard
python run_dashboard.py
```

**📖 See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.**

## 📁 Project Structure

```
.
├── data/                          # Dataset storage
│   ├── facial/                    # Facial emotion datasets
│   ├── speech/                    # Speech emotion datasets
│   └── text/                      # Text emotion datasets
├── models/                        # Model architectures
│   ├── facial_model/              # CNN models for facial recognition
│   ├── speech_model/              # Speech emotion models
│   ├── text_model/                # NLP models for text
│   └── fusion_model/              # Multimodal fusion models
├── src/                           # Source code
│   ├── facial_recognition/        # Face detection and emotion classification
│   ├── speech_analysis/           # ASR and speech emotion analysis
│   ├── text_analysis/             # NLP-based text emotion detection
│   ├── fusion/                    # Multimodal fusion algorithms
│   ├── dashboard/                 # Web dashboard interface
│   └── utils/                     # Utility functions
├── saved_models/                  # Trained model weights
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
├── notebooks/                     # Jupyter notebooks for experimentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

### Functional Requirements
- ✅ **FR1-FR5:** Multi-input support (image, video, audio, text)
- ✅ **FR6-FR8:** Facial emotion recognition with automatic face detection
- ✅ **FR9-FR10:** Speech emotion analysis with ASR
- ✅ **FR11-FR12:** Text emotion analysis using NLP
- ✅ **FR13-FR14:** Multimodal fusion with adaptive weighting
- ✅ **FR15-FR17:** Web dashboard with visualization
- ✅ **FR19-FR20:** Model training and performance evaluation
- ✅ **FR22-FR23:** Comprehensive documentation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Student's Emotion Recognition using Multimodality and Deep Learning"
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models and datasets**
```bash
# Instructions for dataset download will be added
```

## Usage

### Training Models

```bash
# Train facial emotion model
python src/facial_recognition/train.py

# Train speech emotion model
python src/speech_analysis/train.py

# Train text emotion model
python src/text_analysis/train.py
```

### Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Making Predictions

```python
from src.fusion.multimodal_predictor import MultimodalPredictor

predictor = MultimodalPredictor()
result = predictor.predict(
    image_path="path/to/image.jpg",
    audio_path="path/to/audio.wav",
    text="Sample text input"
)
print(f"Detected emotion: {result['emotion']}")
```

## Datasets

The system uses the following datasets:
- **Facial:** FER2013, CK+, AffectNet
- **Speech:** RAVDESS, TESS, SAVEE
- **Text:** GoEmotions, EmoContext

## Model Performance

Performance metrics will be updated after training:
- Accuracy
- Precision
- Recall
- F1-Score

## Technology Stack

- **Deep Learning:** TensorFlow, PyTorch
- **Computer Vision:** OpenCV, MTCNN
- **Audio Processing:** librosa, Whisper
- **NLP:** NLTK, spaCy, Transformers
- **Web Framework:** Streamlit/Flask
- **Visualization:** Matplotlib, Plotly, Seaborn

## Development Status

🚧 **Project is currently under development**

Check `TASK_CHECKLIST.md` for current progress.

## Documentation

- [Plan of Action](PLAN_OF_ACTION.md) - Detailed implementation plan
- [Task Checklist](TASK_CHECKLIST.md) - Progress tracking

## License

[Add license information]

## Contributors

[Add contributor information]

## Acknowledgments

- Dataset providers
- Open-source libraries and frameworks used in this project
