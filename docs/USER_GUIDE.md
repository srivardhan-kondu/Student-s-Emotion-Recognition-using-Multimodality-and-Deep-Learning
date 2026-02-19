# Multimodal Emotion Recognition System - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Using the Dashboard](#using-the-dashboard)
5. [Training Models](#training-models)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

## Introduction

The Multimodal Emotion Recognition System is an AI-powered platform that detects emotions from facial expressions, speech, and text. It combines three deep learning models using adaptive fusion to provide accurate emotion predictions.

**Supported Emotions:**
- Happy 😊
- Sad 😢
- Angry 😠
- Neutral 😐
- Fear 😨
- Surprise 😲

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "Student's Emotion Recognition using Multimodality and Deep Learning"
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Quick Start

### Running the Dashboard
```bash
python run_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### Making a Quick Prediction

**Python API:**
```python
from src.fusion.multimodal_predictor import MultimodalEmotionPredictor

# Initialize predictor
predictor = MultimodalEmotionPredictor()
predictor.load_models()

# Predict from image
result = predictor.predict_from_image("path/to/image.jpg")
print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']:.2%}")

# Predict from text
result = predictor.predict_from_text("I am so happy today!")
print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']:.2%}")

# Multimodal prediction
result = predictor.predict_multimodal(
    image_path="path/to/image.jpg",
    audio_path="path/to/audio.wav",
    text="This is amazing!"
)
print(f"Fused Emotion: {result['emotion']}, Confidence: {result['confidence']:.2%}")
```

## Using the Dashboard

### 1. Multimodal Tab
Upload image, audio, and/or text simultaneously for comprehensive emotion analysis.

**Steps:**
1. Upload an image (JPG, PNG)
2. Upload an audio file (WAV, MP3)
3. Enter text in the text area
4. Click "Analyze All Modalities"
5. View fused emotion result with individual modality breakdowns

### 2. Image Tab
Analyze facial emotions from images or camera.

**Steps:**
1. Select "Upload" or "Camera"
2. Upload image or take a picture
3. Click "Analyze Facial Emotion"
4. View detected emotion with confidence scores

### 3. Audio Tab
Analyze emotions from speech audio.

**Steps:**
1. Upload audio file (WAV, MP3, OGG)
2. Click "Analyze Speech Emotion"
3. View detected emotion from speech patterns

### 4. Text Tab
Analyze emotions from text input.

**Steps:**
1. Enter text in the text area
2. Click "Analyze Text Emotion"
3. View detected emotion from text sentiment

### Fusion Configuration
Adjust fusion strategy and modality weights in the sidebar:

- **Fusion Strategy:** Weighted, Adaptive, or Voting
- **Modality Weights:** Adjust importance of each modality (0.0 - 1.0)

## Training Models

### Training Facial Emotion Model

**Dataset:** FER2013 (Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013))

```bash
# Place dataset in data/facial/fer2013/fer2013.csv
python src/facial_recognition/train.py
```

**Configuration:** Edit `src/config.py` to adjust:
- Image size
- Batch size
- Number of epochs
- Learning rate
- Model architecture (custom_cnn, vgg16, resnet50)

### Training Speech Emotion Model

**Dataset:** RAVDESS (Download from [Kaggle](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio))

```bash
# Place dataset in data/speech/ravdess/
python src/speech_analysis/train.py
```

**Configuration:** Edit `src/config.py` to adjust:
- Sample rate
- MFCC coefficients
- Batch size
- Number of epochs
- Model type (LSTM, CNN, Hybrid)

### Training Text Emotion Model

**Dataset:** GoEmotions (Download from [GitHub](https://github.com/google-research/google-research/tree/master/goemotions))

```bash
# Place dataset in data/text/goemotions/goemotions.csv
python src/text_analysis/train.py
```

**Configuration:** Edit `src/config.py` to adjust:
- BERT model (bert-base-uncased, roberta-base)
- Max sequence length
- Batch size
- Number of epochs
- Learning rate

## API Reference

### MultimodalEmotionPredictor

Main class for emotion prediction.

```python
from src.fusion.multimodal_predictor import MultimodalEmotionPredictor

predictor = MultimodalEmotionPredictor()
predictor.load_models()
```

**Methods:**

- `predict_from_image(image_path: str) -> Dict`
  - Predict emotion from image file
  - Returns: `{'emotion': str, 'confidence': float, 'probabilities': np.ndarray}`

- `predict_from_audio(audio_path: str) -> Dict`
  - Predict emotion from audio file
  - Returns: `{'emotion': str, 'confidence': float, 'probabilities': np.ndarray}`

- `predict_from_text(text: str) -> Dict`
  - Predict emotion from text
  - Returns: `{'emotion': str, 'confidence': float, 'probabilities': np.ndarray}`

- `predict_multimodal(image_path=None, audio_path=None, text=None) -> Dict`
  - Predict emotion using multiple modalities
  - Returns: Fused result with individual modality results

### MultimodalFusion

Fusion engine for combining modality outputs.

```python
from src.fusion.multimodal_fusion import MultimodalFusion

fusion = MultimodalFusion(
    fusion_type='weighted',
    facial_weight=0.4,
    speech_weight=0.3,
    text_weight=0.3
)
```

**Methods:**

- `fuse(facial_result, speech_result, text_result) -> Dict`
  - Fuse emotion predictions from multiple modalities
  
- `set_weights(facial_weight, speech_weight, text_weight)`
  - Update fusion weights

## Troubleshooting

### Issue: Models not found
**Solution:** Train models first or download pre-trained models and place in `saved_models/` directory.

### Issue: NLTK data not found
**Solution:** Run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: Camera not working in dashboard
**Solution:** Grant camera permissions to your browser. Use HTTPS or localhost.

### Issue: Audio file not supported
**Solution:** Convert audio to WAV format using:
```bash
ffmpeg -i input.mp3 output.wav
```

### Issue: Out of memory during training
**Solution:** Reduce batch size in `src/config.py`

### Issue: Low accuracy
**Solution:** 
- Ensure datasets are properly formatted
- Increase number of training epochs
- Try different model architectures
- Adjust learning rate

## Performance Metrics

Expected model performance (after training on full datasets):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Facial | 65-70% | 0.66-0.71 | 0.65-0.70 | 0.65-0.70 |
| Speech | 60-65% | 0.61-0.66 | 0.60-0.65 | 0.60-0.65 |
| Text | 70-75% | 0.71-0.76 | 0.70-0.75 | 0.70-0.75 |
| Multimodal | 72-78% | 0.73-0.79 | 0.72-0.78 | 0.72-0.78 |

## Support

For issues and questions:
- Check the [PLAN_OF_ACTION.md](PLAN_OF_ACTION.md) for implementation details
- Review the [TASK_CHECKLIST.md](TASK_CHECKLIST.md) for feature status
- Consult the source code documentation

## License

This project is for educational purposes.
