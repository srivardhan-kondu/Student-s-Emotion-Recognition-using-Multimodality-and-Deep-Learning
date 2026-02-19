# 🚀 Quick Start Guide

Get the Multimodal Emotion Recognition System up and running in minutes!

## Setup

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 3: Download Datasets

Download the following datasets from Kaggle and place them in the `data/` directory:

- **FER2013** → `data/facial/fer2013/` ([Kaggle link](https://www.kaggle.com/datasets/msambare/fer2013))
- **RAVDESS** → `data/speech/ravdess/` ([Kaggle link](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio))
- **GoEmotions** → `data/text/goemotions/` ([Kaggle link](https://www.kaggle.com/datasets/debarshichanda/goemotions))

See [DATASET_INSTRUCTIONS.md](DATASET_INSTRUCTIONS.md) for detailed download steps.

### Step 4: Train Models
```bash
# Train facial model (~30 minutes)
python src/facial_recognition/train.py

# Train speech model (~20 minutes)
python src/speech_analysis/train.py

# Train text model (~40 minutes with BERT)
python src/text_analysis/train.py
```

### Step 5: Run Dashboard
```bash
python run_dashboard.py
```

Open your browser to: **http://localhost:8501**

## 🎯 What's Next?

After setup, you can:

1. **Test the Dashboard:**
   - Upload an image to detect facial emotions
   - Upload audio to analyze speech emotions
   - Enter text to detect sentiment
   - Try multimodal prediction with all three!

2. **Customize Settings:**
   - Edit `src/config.py` to adjust model parameters
   - Change fusion weights in the dashboard sidebar

3. **Deploy:**
   - See `docs/DEPLOYMENT_GUIDE.md` for deployment options

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Python version too old | Install Python 3.8+ |
| Dependencies fail to install | `pip install --upgrade pip` |
| Models not found | Train models first using the scripts above |
| Port 8501 already in use | `lsof -i :8501` then `kill -9 <PID>` |

## 🎓 Example Usage

```python
from src.fusion.multimodal_predictor import MultimodalPredictor

predictor = MultimodalPredictor()

# Multimodal prediction
result = predictor.predict_multimodal(
    image_path="photo.jpg",
    audio_path="speech.wav",
    text="I'm feeling great today!"
)
print(f"Emotion: {result['emotion']} ({result['confidence']:.1%})")
```

---

**Ready to go!** 🎉
