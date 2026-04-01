"""model evaluation helpers"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
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
    if plt is None or sns is None:
        raise ModuleNotFoundError("matplotlib and seaborn are required for plotting confusion matrix.")

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
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting ROC curve.")

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
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.round(4)

    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(comparison_df.to_string())

    return comparison_df

def _to_float(value, default=np.nan):
    """Safe conversion to float for mixed csv values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def load_logistic_metrics(logistic_csv_path):
    """Load logistic-regression metrics and map to common names."""
    df = pd.read_csv(logistic_csv_path)
    if df.empty:
        raise ValueError("Logistic regression metrics file is empty.")

    row = df.iloc[0]
    metrics = {
        "model": "Logistic Regression (Optimal)",
        "accuracy": _to_float(row.get("Test_Accuracy_Optimal")),
        "precision": _to_float(row.get("Test_Precision_Optimal")),
        "recall": _to_float(row.get("Test_Recall_Optimal")),
        "f1_score": _to_float(row.get("Test_F1_Optimal")),
        "auc_roc": _to_float(row.get("Test_ROC_AUC")),
    }
    return pd.DataFrame([metrics])

def load_rf_metrics(rf_metrics_path):
    """Load random-forest metrics in either baseline+tuned or tuned-only format."""
    df = pd.read_csv(rf_metrics_path)
    required = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"RF metrics missing columns: {missing}")

    if "model" not in df.columns:
        df["model"] = "Random Forest"

    return df[["model", *required]].copy()

def load_svm_metrics(svm_metrics_path):
    """Load svm metrics where model names are stored as index."""
    df = pd.read_csv(svm_metrics_path, index_col=0)
    df = df.reset_index().rename(columns={"index": "model"})

    required = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SVM metrics missing columns: {missing}")

    return df[["model", *required]].copy()

def load_decision_tree_metrics(dt_metrics_path):
    """Load decision tree metrics from decision_tree_results.csv."""
    df = pd.read_csv(dt_metrics_path)
    if df.empty:
        raise ValueError("Decision tree metrics file is empty.")

    row = df.iloc[0]
    metrics = {
        "model": "Decision Tree",
        "accuracy": _to_float(row.get("Test Accuracy")),
        "precision": _to_float(row.get("Test Precision")),
        "recall": _to_float(row.get("Test Recall")),
        "f1_score": _to_float(row.get("Test F1")),
        "auc_roc": _to_float(row.get("Test ROC-AUC")),
    }
    return pd.DataFrame([metrics])

def compare_saved_model_metrics(results_dir="results", export=True):
    """
    Build one comparison table from existing model result csv files.

    Expected files under results_dir:
      - logistic_regression_results.csv
      - rf_model_comparison.csv (preferred) or rf_metrics.csv
      - svm_metrics_comparison.csv
      - decision_tree_results.csv
    """
    results_dir = Path(results_dir)

    logistic_path = results_dir / "logistic_regression_results.csv"
    rf_model_comparison_path = results_dir / "rf_model_comparison.csv"
    rf_metrics_path = results_dir / "rf_metrics.csv"
    svm_path = results_dir / "svm_metrics_comparison.csv"
    dt_path = results_dir / "decision_tree_results.csv"

    frames = []

    if logistic_path.exists():
        frames.append(load_logistic_metrics(logistic_path))
    else:
        print(f"Warning: missing {logistic_path}")

    if rf_model_comparison_path.exists():
        frames.append(load_rf_metrics(rf_model_comparison_path))
    elif rf_metrics_path.exists():
        frames.append(load_rf_metrics(rf_metrics_path))
    else:
        print("Warning: missing random-forest metrics file.")

    if svm_path.exists():
        frames.append(load_svm_metrics(svm_path))
    else:
        print(f"Warning: missing {svm_path}")

    if dt_path.exists():
        frames.append(load_decision_tree_metrics(dt_path))
    else:
        print(f"Warning: missing {dt_path}")

    if not frames:
        raise FileNotFoundError("No metrics files found for model comparison.")

    combined = pd.concat(frames, ignore_index=True)
    numeric_cols = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    ranked = combined.sort_values(by="f1_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 68)
    print("All Model Comparison (ranked by F1)")
    print("=" * 68)
    print(ranked.to_string(index=False))

    if export:
        all_path = results_dir / "model_comparison_all_models.csv"
        ranked_path = results_dir / "model_comparison_ranked_by_f1.csv"
        combined.to_csv(all_path, index=False)
        ranked.to_csv(ranked_path, index=False)
        print(f"\nSaved: {all_path}")
        print(f"Saved: {ranked_path}")

    return ranked
