"""
Text Emotion Recognition Model — Upgraded.
Key improvements:
  - RoBERTa-base (outperforms BERT on sentiment/emotion tasks)
  - Focal Loss support for class imbalance
  - max_length=256 for longer student feedback
  - Backward-compatible with existing BERT setup
Implements FR11, FR12.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    BertTokenizer, BertForSequenceClassification
)
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise']


# ─── Focal Loss ────────────────────────────────────────────────────────────── #

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Reduces the relative loss for well-classified examples,
    focusing training on hard negatives.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2)
        alpha: Class balancing weight (None = uniform)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: torch.Tensor = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ─── RoBERTa Classifier ────────────────────────────────────────────────────── #

class RoBERTaEmotionClassifier:
    """
    RoBERTa-based emotion classifier.
    Outperforms BERT on sentiment/emotion classification benchmarks.
    """

    def __init__(self, model_name: str = 'roberta-base', num_classes: int = 6):
        self.model_name = model_name
        self.num_classes = num_classes
        self.tokenizer = None
        self.model = None
        logger.info(f"Initialized RoBERTa classifier: {model_name}")

    def build_model(self):
        """Load RoBERTa tokenizer and model."""
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        logger.info(f"Loaded RoBERTa with {self.num_classes} classes")
        return self.model

    def predict(self, text: str, device: str = 'cpu',
                temperature: float = 1.2) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from text with temperature-calibrated confidence.

        Args:
            text: Input text string
            device: 'cpu' | 'cuda' | 'mps'
            temperature: Softmax temperature for calibration

        Returns:
            (emotion_label, confidence, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call build_model() or load_model().")

        inputs = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=256, return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.model.eval()
        self.model.to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Temperature scaling
            logits = outputs.logits / temperature
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        emotion_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[emotion_idx])
        return EMOTIONS[emotion_idx], confidence, probabilities

    def save_model(self, filepath: str):
        """Save model + tokenizer to directory."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"RoBERTa model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model + tokenizer from directory."""
        self.tokenizer = RobertaTokenizer.from_pretrained(filepath)
        self.model = RobertaForSequenceClassification.from_pretrained(filepath)
        logger.info(f"RoBERTa model loaded from {filepath}")


# ─── BERT Classifier (backward compatible) ─────────────────────────────────── #

class BERTEmotionClassifier:
    """BERT-based emotion classifier (kept for backward compatibility)."""

    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 6):
        self.model_name = model_name
        self.num_classes = num_classes
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        logger.info(f"Initialized BERT classifier: {model_name}")

    def build_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_classes
        )
        return self.model

    def predict(self, text: str, temperature: float = 1.2) -> Tuple[str, float, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not loaded")
        inputs = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=256, return_tensors='pt'
        )
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits / temperature
            probabilities = torch.softmax(logits, dim=1)[0].numpy()
        emotion_idx = int(np.argmax(probabilities))
        return EMOTIONS[emotion_idx], float(probabilities[emotion_idx]), probabilities

    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)

    def load_model(self, filepath: str):
        self.model = BertForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = BertTokenizer.from_pretrained(filepath)


# ─── LSTM Classifier ───────────────────────────────────────────────────────── #

class LSTMEmotionClassifier(nn.Module):
    """Bidirectional LSTM text classifier (lightweight alternative)."""

    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 hidden_dim: int = 256, num_classes: int = 6,
                 num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Concatenate last forward + backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden_cat))


# ─── High-level API ────────────────────────────────────────────────────────── #

class TextEmotionAnalyzer:
    """
    High-level text emotion analyzer.
    Default: RoBERTa (best performance). Falls back to BERT if needed.
    """

    def __init__(self, model_type: str = 'roberta'):
        """
        Args:
            model_type: 'roberta' (default) | 'bert'
        """
        self.model_type = model_type
        self.model = None

        # Check if trained RoBERTa model exists
        roberta_path = 'saved_models/text_bert_model'   # reused name for RoBERTa save
        legacy_path = 'saved_models/text_emotion_model.pt'

        import os, json
        if model_type == 'roberta':
            if os.path.exists(roberta_path):
                # Auto-detect architecture from config.json
                config_path = os.path.join(roberta_path, 'config.json')
                arch = 'roberta'  # default assumption
                if os.path.exists(config_path):
                    try:
                        with open(config_path) as f:
                            cfg = json.load(f)
                        arch = cfg.get('model_type', 'roberta')
                    except Exception:
                        pass

                if arch == 'bert':
                    logger.info(f"Detected BERT architecture in {roberta_path}")
                    self.model = BERTEmotionClassifier()
                    try:
                        self.model.load_model(roberta_path)
                        logger.info("Loaded trained BERT model successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load BERT model: {e}")
                        self.model = None
                else:
                    logger.info(f"Loading trained RoBERTa model from {roberta_path}")
                    self.model = RoBERTaEmotionClassifier()
                    try:
                        self.model.load_model(roberta_path)
                    except Exception as e:
                        logger.warning(f"Failed to load saved RoBERTa model: {e}")
                        self.model = None

            if self.model is None:
                logger.warning("No trained model found. Falling back to fresh RoBERTa-base (untrained)")
                self.model = RoBERTaEmotionClassifier()
                self.model.build_model()

        elif model_type == 'bert':
            self.model = BERTEmotionClassifier()
            self.model.build_model()

    def analyze_emotion(self, text: str) -> dict:
        """
        Analyze emotion in text.

        Args:
            text: Input text

        Returns:
            Dict with emotion, confidence, and per-emotion scores
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        emotion, confidence, probabilities = self.model.predict(text)

        emotion_scores = {
            EMOTIONS[i]: float(probabilities[i])
            for i in range(len(EMOTIONS))
        }

        return {
            'text': text,
            'emotion': emotion,
            'confidence': float(confidence),
            'emotion_scores': emotion_scores
        }

    def batch_analyze(self, texts: List[str]) -> List[dict]:
        """Analyze emotions in a list of texts."""
        results = [self.analyze_emotion(t) for t in texts]
        logger.info(f"Analyzed {len(results)} texts")
        return results
