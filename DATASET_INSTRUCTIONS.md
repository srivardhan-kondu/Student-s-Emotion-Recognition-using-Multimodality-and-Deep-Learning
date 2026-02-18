# Dataset Download Instructions

## 📥 Required Datasets for Training

You need to download 3 datasets to train the emotion recognition models:

### 1. FER2013 - Facial Emotion Dataset
**What:** Grayscale images of faces with emotion labels  
**Size:** ~60 MB  
**Samples:** 35,887 images  
**URL:** https://www.kaggle.com/datasets/msambare/fer2013

**Steps:**
1. Go to the URL above
2. Sign in to Kaggle (create free account if needed)
3. Click "Download" button
4. Extract the downloaded ZIP file
5. Place `fer2013.csv` in: `data/facial/fer2013/fer2013.csv`

### 2. RAVDESS - Speech Emotion Dataset
**What:** Audio recordings of actors expressing emotions  
**Size:** ~1 GB  
**Samples:** 1,440 audio files  
**URL:** https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio

**Steps:**
1. Go to the URL above
2. Sign in to Kaggle
3. Click "Download" button
4. Extract the downloaded ZIP file
5. Place all audio files in: `data/speech/ravdess/`

### 3. GoEmotions - Text Emotion Dataset
**What:** Reddit comments labeled with emotions  
**Size:** ~50 MB  
**Samples:** 58,000+ texts  
**URL:** https://github.com/google-research/google-research/tree/master/goemotions

**Steps:**
1. Go to the URL above
2. Download the dataset files (or use the direct link below)
3. Direct CSV: https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
4. Place in: `data/text/goemotions/goemotions.csv`

## 🚀 Quick Download Links

The dataset helper has opened these URLs in your browser. 

**Download them now, then run:**

```bash
# After downloading and placing files in correct locations

# Verify datasets
ls -lh data/facial/fer2013/
ls -lh data/speech/ravdess/
ls -lh data/text/goemotions/

# Then train models
python src/facial_recognition/train.py
python src/speech_analysis/train.py
python src/text_analysis/train.py
```

## ⏱️ Estimated Time
- Downloads: 30-60 minutes (depending on internet speed)
- File placement: 5 minutes
- Training: 90 minutes automated

**Total:** ~2-3 hours to fully trained system
