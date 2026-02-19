"""
Utility functions for the Multimodal Emotion Recognition System.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: List[str], save_path: str = None):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    loss = history.history['loss'] if hasattr(history, 'history') else history['loss']
    val_loss = history.history['val_loss'] if hasattr(history, 'history') and 'val_loss' in history.history else history.get('val_loss')
    
    axes[0].plot(loss, label='Training Loss')
    if val_loss:
        axes[0].plot(val_loss, label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    acc = history.history['accuracy'] if hasattr(history, 'history') else history['accuracy']
    val_acc = history.history['val_accuracy'] if hasattr(history, 'history') and 'val_accuracy' in history.history else history.get('val_accuracy')
    
    axes[1].plot(acc, label='Training Accuracy')
    if val_acc:
        axes[1].plot(val_acc, label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Training history plot saved to {save_path}")
    plt.close()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     class_names: List[str]) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary containing metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_metrics': {
            class_names[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i]
            }
            for i in range(len(class_names))
        }
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("\n" + "-"*50)
    print("PER-CLASS METRICS")
    print("-"*50)
    
    for emotion, scores in metrics['per_class_metrics'].items():
        print(f"\n{emotion.upper()}:")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1-Score:  {scores['f1_score']:.4f}")
    print("="*50 + "\n")


def create_directory(path: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Directory created/verified: {path}")


def save_model_info(model_path: str, metrics: Dict[str, Any], 
                   config: Dict[str, Any]):
    """
    Save model information and metrics to a text file.
    
    Args:
        model_path: Path where model is saved
        metrics: Model evaluation metrics
        config: Model configuration
    """
    info_path = model_path.replace('.h5', '_info.txt').replace('.pt', '_info.txt')
    
    with open(info_path, 'w') as f:
        f.write("MODEL INFORMATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Path: {model_path}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        
        f.write("\nPER-CLASS METRICS\n")
        f.write("-"*50 + "\n")
        for emotion, scores in metrics['per_class_metrics'].items():
            f.write(f"\n{emotion.upper()}:\n")
            f.write(f"  Precision: {scores['precision']:.4f}\n")
            f.write(f"  Recall:    {scores['recall']:.4f}\n")
            f.write(f"  F1-Score:  {scores['f1_score']:.4f}\n")
    
    logger.info(f"Model info saved to {info_path}")
