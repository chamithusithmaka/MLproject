"""data preprocessing helpers"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(filepath, sample_size=None, random_state=42):
    """load dataset, optional sampling"""
    data = pd.read_csv(filepath)
    if sample_size:
        data = data.sample(n=sample_size, random_state=random_state)
    return data


def fill_missing_values(data):
    """fill missing values"""
    df = data.copy()

    # fill categorical with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # fill numeric with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    return df


def encode_features(data):
    """encode features"""
    df = data.copy()

    # drop id column
    if 'LoanID' in df.columns:
        df.drop('LoanID', axis=1, inplace=True)

    # yes/no to 1/0
    binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # ordinal encode education
    education_order = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    if 'Education' in df.columns:
        df['Education'] = df['Education'].map(education_order)

    # one-hot remaining categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def split_and_scale(data, target_col='Default', test_size=0.2, random_state=42):
    """split train/test and scale"""
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    feature_names = X.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, feature_names


def apply_smote(x_train, y_train, random_state=42, sampling_strategy='auto'):
    """apply smote"""
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    return x_resampled, y_resampled


def preprocess_pipeline(filepath, sample_size=None, apply_smote_flag=True):
    """full preprocessing pipeline"""
    # load data
    data = load_data(filepath, sample_size)

    # fill missing values
    data = fill_missing_values(data)

    # encode features
    data_encoded = encode_features(data)

    # split and scale
    x_train, x_test, y_train, y_test, scaler, feature_names = split_and_scale(data_encoded)

    # apply smote (optional)
    if apply_smote_flag:
        x_train, y_train = apply_smote(x_train, y_train)

    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'data_encoded': data_encoded
    }
