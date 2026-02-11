import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import os
from typing import List, Dict, Any

def setup_plotting_style():
    """Sets a clean, professional style for plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

def plot_training_history(history: Dict[str, List[float]], save_path: str = 'plots/training_history.png'):
    """
    Plots training and validation loss and accuracy with improved aesthetics.
    """
    setup_plotting_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc', linewidth=2, markersize=4)
    ax2.set_title('Training and Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: str = 'plots/confusion_matrix.png'):
    """
    Plots a high-quality confusion matrix.
    """
    setup_plotting_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    cm = confusion_matrix(y_true, y_pred)
    # Normalize CM
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontweight='bold', pad=20)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_performance_report(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> str:
    """
    Returns a comprehensive text-based classification report.
    """
    report = classification_report(y_true, y_pred, target_names=classes)
    
    # Add summary metrics
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    summary = f"\nAdditional Metrics:\n"
    summary += f"F1-Score (Micro): {f1_micro:.4f}\n"
    summary += f"F1-Score (Macro): {f1_macro:.4f}\n"
    
    return report + summary
