# Technical Documentation - Multimodal Emotion Recognition System

## System Architecture

### Overview
The system consists of three independent emotion recognition modules (facial, speech, text) that are combined using a multimodal fusion engine.

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Multimodal Emotion Predictor                    │
└─────────────────────────────────────────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Facial Module    │ │ Speech Module    │ │ Text Module      │
│ - Face Detection │ │ - ASR            │ │ - Preprocessing  │
│ - CNN Model      │ │ - Feature Extract│ │ - BERT/LSTM      │
└──────────────────┘ └──────────────────┘ └──────────────────┘
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Multimodal Fusion   │
                   │  - Weighted          │
                   │  - Adaptive          │
                   │  - Voting            │
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Final Emotion       │
                   │  Prediction          │
                   └──────────────────────┘
```

## Module Details

### 1. Facial Emotion Recognition

**Components:**
- `FaceDetector`: Detects faces using Haar Cascade or DNN
- `EmotionCNN`: CNN-based emotion classifier

**Architecture:**
```
Input (48x48 grayscale) → Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
                        → Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout
                        → Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout
                        → Flatten → Dense(256) → Dropout → Dense(128) → Dropout
                        → Dense(6, softmax)
```

**Training:**
- Dataset: FER2013 (35,887 images)
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Data Augmentation: Rotation, shift, flip, zoom

### 2. Speech Emotion Analysis

**Components:**
- `AudioFeatureExtractor`: Extracts MFCC, mel spectrogram, chroma, etc.
- `SpeechRecognizer`: Converts speech to text (Google API/Whisper)
- `SpeechEmotionModel`: LSTM/CNN for emotion classification

**Features Extracted:**
- MFCC (40 coefficients)
- Mel Spectrogram
- Chroma Features
- Spectral Contrast
- Zero Crossing Rate
- Pitch (F0)

**Architecture (LSTM):**
```
Input (640 features) → LSTM(128) → Dropout → LSTM(64) → Dropout → LSTM(32) → Dropout
                     → Dense(64) → Dropout → Dense(6, softmax)
```

**Training:**
- Dataset: RAVDESS (1,440 audio files)
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy

### 3. Text Emotion Analysis

**Components:**
- `TextPreprocessor`: Cleans and tokenizes text
- `BERTEmotionClassifier`: BERT-based emotion classifier
- `LSTMEmotionClassifier`: LSTM-based alternative

**Preprocessing:**
1. Lowercase conversion
2. URL/mention/hashtag removal
3. Special character removal
4. Tokenization
5. Lemmatization

**Architecture (BERT):**
```
Input Text → BERT Tokenizer → BERT Base (12 layers, 768 hidden)
          → Classification Head → Dense(6, softmax)
```

**Training:**
- Dataset: GoEmotions (58,000+ texts)
- Optimizer: AdamW (lr=2e-5)
- Loss: Categorical Crossentropy
- Max Length: 128 tokens

### 4. Multimodal Fusion

**Fusion Strategies:**

1. **Weighted Fusion:**
   ```
   P_fused = w_f * P_facial + w_s * P_speech + w_t * P_text
   where w_f + w_s + w_t = 1
   ```

2. **Adaptive Fusion:**
   ```
   w_i = confidence_i * base_weight_i
   P_fused = Σ(w_i * P_i) / Σ(w_i)
   ```

3. **Voting Fusion:**
   ```
   emotion_fused = mode([emotion_facial, emotion_speech, emotion_text])
   ```

**Default Weights:**
- Facial: 0.4
- Speech: 0.3
- Text: 0.3

## Configuration

All configuration is centralized in `src/config.py`:

```python
# Facial Recognition Settings
FACIAL_CONFIG = {
    "image_size": (48, 48),
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "model_architecture": "custom_cnn"
}

# Speech Analysis Settings
SPEECH_CONFIG = {
    "sample_rate": 22050,
    "n_mfcc": 40,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001
}

# Text Analysis Settings
TEXT_CONFIG = {
    "max_length": 128,
    "model_name": "bert-base-uncased",
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5
}

# Fusion Settings
FUSION_CONFIG = {
    "fusion_type": "weighted",
    "facial_weight": 0.4,
    "speech_weight": 0.3,
    "text_weight": 0.3
}
```

## Data Flow

### Training Phase
```
1. Load Dataset → 2. Preprocess → 3. Split (Train/Val/Test)
   → 4. Train Model → 5. Evaluate → 6. Save Model
```

### Inference Phase
```
1. Input (Image/Audio/Text) → 2. Preprocess
   → 3. Feature Extraction → 4. Model Prediction
   → 5. Fusion (if multimodal) → 6. Output Result
```

## Performance Optimization

### Model Optimization
- Early stopping with patience=10
- Learning rate reduction on plateau
- Batch normalization for faster convergence
- Dropout for regularization

### Inference Optimization
- Model caching in session state
- Batch prediction for multiple inputs
- GPU acceleration (if available)

## Error Handling

### Graceful Degradation
- If one modality fails, system continues with available modalities
- Missing modalities are handled by fusion engine
- Default to neutral emotion if all modalities fail

### Input Validation
- Image: Check format, size, face detection
- Audio: Check format, duration, sample rate
- Text: Check length, encoding

## Security Considerations

### Data Privacy
- No data stored permanently without consent
- Temporary files deleted after processing
- User authentication for dashboard access

### Input Sanitization
- Text input sanitized to prevent injection
- File upload validation (type, size)
- Rate limiting on API endpoints

## Deployment

### Local Deployment
```bash
python run_dashboard.py
```

### Docker Deployment
```dockerfile
FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_dashboard.py"]
```

### Cloud Deployment
- Streamlit Cloud
- AWS EC2 + Docker
- Google Cloud Run
- Azure App Service

## Testing

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Model Evaluation
```bash
python src/facial_recognition/train.py --evaluate
python src/speech_analysis/train.py --evaluate
python src/text_analysis/train.py --evaluate
```

## Monitoring

### Metrics to Track
- Prediction latency
- Model accuracy over time
- Error rates by modality
- User engagement

### Logging
- All predictions logged with timestamp
- Error logs for debugging
- Performance metrics logged

## Future Enhancements

1. **Real-time Video Processing**
   - Frame-by-frame emotion detection
   - Temporal smoothing

2. **Multi-language Support**
   - Text analysis in multiple languages
   - Speech recognition for multiple languages

3. **Emotion Intensity**
   - Detect emotion intensity (mild, moderate, strong)

4. **Context Awareness**
   - Consider context in emotion prediction
   - Temporal emotion patterns

5. **Model Improvements**
   - Ensemble methods
   - Attention mechanisms
   - Transfer learning from larger datasets

## References

### Datasets
- FER2013: https://www.kaggle.com/datasets/msambare/fer2013
- RAVDESS: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
- GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions

### Models
- BERT: https://huggingface.co/bert-base-uncased
- VGG16: https://keras.io/api/applications/vgg/
- ResNet50: https://keras.io/api/applications/resnet/

### Libraries
- TensorFlow: https://www.tensorflow.org/
- PyTorch: https://pytorch.org/
- Transformers: https://huggingface.co/transformers/
- Streamlit: https://streamlit.io/
