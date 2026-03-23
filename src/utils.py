"""utility helpers"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_processed_data(base_path='../dataset/processed'):
    """load processed datasets"""
    return {
        'full': pd.read_csv(f'{base_path}/cleaned_preprocessed_full.csv'),
        'train': pd.read_csv(f'{base_path}/train_preprocessed_smote.csv'),
        'test': pd.read_csv(f'{base_path}/test_preprocessed.csv')
    }


def get_features_and_target(df, target_col='Default'):
    """split features and target"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


def plot_class_distribution(y, title='Class Distribution'):
    """plot class distribution"""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Default')
    plt.ylabel('Count')

    # add count labels
    for i, count in enumerate(pd.Series(y).value_counts().sort_index()):
        ax.text(i, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=15, title='Feature Importance'):
    """plot feature importance"""
    # build and sort dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return importance_df.sort_values('importance', ascending=False)


def print_dataset_info(data, name='Dataset'):
    """print dataset info"""
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    if 'Default' in data.columns:
        print(f"\nTarget distribution:")
        print(data['Default'].value_counts())
