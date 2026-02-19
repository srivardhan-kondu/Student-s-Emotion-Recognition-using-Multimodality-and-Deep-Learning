"""
Training script for text emotion recognition — Upgraded to RoBERTa + Focal Loss.
Key improvements:
  - roberta-base (replaces bert-base-uncased)
  - FocalLoss (γ=2) for GoEmotions class imbalance
  - max_length=256 for longer student feedback
  - Warmup scheduler (10% of total steps)
  - Best-epoch model checkpoint
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from text_analysis.emotion_model import FocalLoss
from config import TEXT_CONFIG, MODEL_SAVE_PATHS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise']
MODEL_NAME = 'roberta-base'
MAX_LENGTH = 256


# ─── Dataset ───────────────────────────────────────────────────────────────── #

class EmotionDataset(Dataset):
    """PyTorch dataset for RoBERTa emotion classification."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ─── Data loading ──────────────────────────────────────────────────────────── #

def load_goemotions_multilabel(data_dir: str, max_samples: int = 15000):
    """
    Load GoEmotions dataset (multi-label binary format).
    Maps to 6 emotions: happy, sad, angry, neutral, fear, surprise.
    """
    emotion_mapping = {
        'joy': 0,       # happy
        'sadness': 1,   # sad
        'anger': 2,     # angry
        'neutral': 3,   # neutral
        'fear': 4,      # fear
        'surprise': 5   # surprise
    }

    dfs = []
    for i in range(1, 4):
        p = os.path.join(data_dir, f"goemotions_{i}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))

    if not dfs:
        combined = os.path.join(data_dir, "goemotions.csv")
        if os.path.exists(combined):
            dfs = [pd.read_csv(combined)]

    if not dfs:
        logger.error("No GoEmotions data found!")
        return [], []

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(df)}")

    texts, labels = [], []

    for _, row in df.iterrows():
        max_score, best_emotion = 0, None
        for col, lid in emotion_mapping.items():
            if col in df.columns:
                score = row.get(col, 0)
                if score > max_score:
                    max_score = score
                    best_emotion = lid

        if best_emotion is not None and max_score > 0:
            texts.append(str(row['text']))
            labels.append(best_emotion)

    # Balanced stratified sampling
    if max_samples and len(texts) > max_samples:
        per_class = max_samples // len(emotion_mapping)
        balanced_texts, balanced_labels = [], []
        counts = {i: 0 for i in range(6)}
        for t, l in zip(texts, labels):
            if counts[l] < per_class:
                balanced_texts.append(t)
                balanced_labels.append(l)
                counts[l] += 1
        texts, labels = balanced_texts, balanced_labels

    logger.info(f"Training samples: {len(texts)}")
    logger.info(f"Class distribution: {Counter(labels)}")
    return texts, labels


# ─── Training loop ─────────────────────────────────────────────────────────── #

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    """Single training epoch with focal loss."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label_ids = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, label_ids)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == label_ids).sum().item()
        total += len(label_ids)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} — Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device):
    """Single evaluation epoch."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_ids = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, label_ids)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == label_ids).sum().item()
            total += len(label_ids)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_ids.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


# ─── Main ──────────────────────────────────────────────────────────────────── #

def main():
    logger.info("=" * 60)
    logger.info("  TEXT EMOTION RECOGNITION - TRAINING (RoBERTa + Focal Loss)")
    logger.info("=" * 60)

    dataset_dir = "data/text/goemotions"
    model_save_dir = os.path.join(os.path.dirname(str(MODEL_SAVE_PATHS['text'])), 'text_bert_model')
    model_pt_path = str(MODEL_SAVE_PATHS['text'])

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading GoEmotions dataset...")
    texts, labels = load_goemotions_multilabel(dataset_dir, max_samples=15000)

    if not texts:
        logger.error("No data loaded!")
        return

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    logger.info(f"Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # Tokenizer & Datasets
    logger.info(f"Loading {MODEL_NAME} tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_ds = EmotionDataset(X_train, y_train, tokenizer)
    val_ds = EmotionDataset(X_val, y_val, tokenizer)
    test_ds = EmotionDataset(X_test, y_test, tokenizer)

    batch_size = TEXT_CONFIG.get('batch_size', 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    # Model
    logger.info(f"Loading {MODEL_NAME} model...")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
    model.to(device)

    # Focal Loss (gamma=2 — focus on hard examples)
    criterion = FocalLoss(gamma=2.0)

    # Optimizer + Warmup scheduler
    epochs = min(TEXT_CONFIG.get('epochs', 5), 5)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)

        logger.info(
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  ✅ New best val accuracy: {val_acc:.4f}")

    # Restore best epoch
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Test evaluation
    logger.info("\nFinal evaluation on test set...")
    _, test_acc, all_preds, all_labels_test = eval_epoch(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info("\n" + classification_report(all_labels_test, all_preds,
                                              target_names=EMOTIONS, zero_division=0))

    # Save
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_pt_path), exist_ok=True)
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    torch.save(model.state_dict(), model_pt_path)
    logger.info(f"Model saved to {model_save_dir}")

    logger.info("=" * 60)
    logger.info("  TEXT TRAINING COMPLETED!")
    logger.info(f"  Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
