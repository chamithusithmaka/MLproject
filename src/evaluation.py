"""model evaluation helpers"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


def evaluate_model(y_true, y_pred, y_prob=None, model_name='Model'):
    """calculate and print metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric.upper():12}: {value:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Default', 'Default']))

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return cm


def plot_roc_curve(y_true, y_prob, model_name='Model'):
    """plot roc curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return auc


def compare_models(results_dict):
    """compare model metrics"""
    import pandas as pd

    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.round(4)

    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(comparison_df.to_string())

    return comparison_df
