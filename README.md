# MLproject - Loan Default Prediction

A comprehensive Machine Learning classification project designed to predict whether borrowers will default on their loans. This project implements and compares multiple supervised learning algorithms to identify high-risk borrowers and support loan approval decisions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Technologies](#technologies)
- [Key Findings](#key-findings)

## Overview

MLproject is a financial risk assessment application built for banks and lending institutions. It leverages machine learning to predict loan defaults by analyzing borrower characteristics and loan attributes. The project demonstrates proper ML development practices including data preprocessing, model training, hyperparameter tuning, evaluation, and comprehensive comparison.

**Target Application**: Loan default risk classification (binary classification)

## Dataset

- **Source**: Coursera's Loan Default Prediction Challenge
- **Size**: 255,347 samples with 18 features
- **File**: `dataset/Loan_default.csv` (24.8 MB)
- **Target Variable**: Binary classification (Default/Non-Default)

### Data Preprocessing Pipeline

The project includes a comprehensive preprocessing pipeline:

1. **Missing Value Imputation**
   - Mode imputation for categorical features
   - Median imputation for numeric features

2. **Feature Encoding**
   - Binary encoding (Yes/No → 1/0)
   - Ordinal encoding (ordered categories)
   - One-hot encoding (nominal categories)

3. **Data Normalization**
   - StandardScaler normalization for numeric features

4. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique) applied to training set
   - Stratified train-test split (80-20)

5. **Processed Datasets**
   - `dataset/processed/cleaned_preprocessed_full.csv` - Complete preprocessed data
   - `dataset/processed/train_preprocessed_smote.csv` - Training set with SMOTE resampling
   - `dataset/processed/test_preprocessed.csv` - Test set

## Project Structure

```
MLproject/
├── src/                                    # Core Python modules
│   ├── preprocessing.py                    # Data loading, cleaning, and preprocessing
│   ├── evaluation.py                       # Evaluation metrics and model comparison
│   ├── utils.py                            # Helper utilities for data and visualization
│   └── model_comparison.py                 # Model comparison orchestration
│
├── notebooks/                              # Jupyter notebooks (one per model)
│   ├── data_preprocessing.ipynb            # EDA and data preprocessing pipeline
│   ├── logistic_regression_model.ipynb     # Logistic Regression training and tuning
│   ├── random_forest.ipynb                 # Random Forest training and tuning
│   ├── svm_model.ipynb                     # Support Vector Machine training
│   └── decision_tree_model.ipynb           # Decision Tree training and analysis
│
├── dataset/                                # Data directory
│   ├── Loan_default.csv                   # Raw dataset
│   └── processed/                          # Preprocessed datasets
│       ├── cleaned_preprocessed_full.csv
│       ├── train_preprocessed_smote.csv
│       └── test_preprocessed.csv
│
├── models/                                 # Trained model artifacts
│   └── random_forest_tuned.pkl            # Serialized best model
│
├── results/                                # Evaluation results and visualizations
│   ├── model_comparison_ranked_by_f1.csv  # Final model comparison
│   ├── logistic_regression_results.csv    # Logistic Regression metrics
│   ├── rf_metrics.csv                      # Random Forest metrics
│   ├── svm_metrics_comparison.csv         # SVM metrics
│   ├── decision_tree_results.csv          # Decision Tree metrics
│   ├── *.png                               # Evaluation plots and visualizations
│   └── FINAL_SUMMARY.txt                   # Comprehensive summary report
│
├── requirements.txt                        # Python dependencies
├── test_environment.py                     # Environment validation script
├── README.md                               # This file
└── .gitignore                              # Git configuration
```

## Features

### Data Analysis & Preprocessing
- Exploratory Data Analysis (EDA) with comprehensive visualizations
- Missing value handling with statistical imputation
- Feature scaling and normalization
- Categorical feature encoding (binary, ordinal, one-hot)
- Class imbalance mitigation using SMOTE

### Model Evaluation
- Multiple evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix generation and visualization
- ROC curve plotting and analysis
- Threshold optimization for classification
- Cross-validation support

### Model Comparison
- Automated comparison across all trained models
- Performance ranking and visualization
- Best model identification based on multiple metrics
- Comprehensive comparison reports

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/chamithusithmaka/MLproject.git
   cd MLproject
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify environment setup**
   ```bash
   python test_environment.py
   ```

## Usage

### Option 1: Run Individual Models (Jupyter Notebooks)

```bash
# Start Jupyter
jupyter notebook

# Then open and run the notebooks in order:
# 1. notebooks/data_preprocessing.ipynb    (Data EDA and preprocessing)
# 2. notebooks/logistic_regression_model.ipynb
# 3. notebooks/random_forest.ipynb
# 4. notebooks/svm_model.ipynb
# 5. notebooks/decision_tree_model.ipynb
```

### Option 2: Compare All Models

```bash
python src/model_comparison.py
```

This will generate a comprehensive comparison report and visualizations.

### Typical Workflow

1. **Data Preprocessing**: Run `data_preprocessing.ipynb` to load raw data, perform EDA, handle missing values, encode features, and apply SMOTE resampling.

2. **Model Training**: Execute individual model notebooks to train and evaluate each algorithm with hyperparameter tuning.

3. **Model Comparison**: Use `model_comparison.py` to aggregate results from all models and generate comparison metrics and visualizations.

## Models Implemented

### 1. Logistic Regression
- **Type**: Linear Classifier
- **Hyperparameter Tuning**: GridSearchCV
- **Optimal Parameters**: C=0.01, penalty=l1, solver=liblinear
- **Threshold Optimization**: Dynamic threshold adjustment (0.56)
- **Best Performance**:
  - F1-Score: 0.3774
  - AUC-ROC: 0.7460
  - Accuracy: 81.85%
- **Status**: ⭐ **Best Performing Model**

### 2. Random Forest
- **Type**: Ensemble (Tree-based)
- **Hyperparameter Tuning**: GridSearchCV
- **Threshold Optimization**: Probability-based threshold adjustment
- **Best Performance**:
  - F1-Score: 0.3210
  - AUC-ROC: 0.7093
- **Features**: Feature importance analysis, threshold analysis

### 3. Support Vector Machine (SVM)
- **Type**: Kernel-based Classifier
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Performance**:
  - F1-Score: 0.1634
  - AUC-ROC: 0.6066
- **Note**: Baseline model with room for kernel and parameter optimization

### 4. Decision Tree
- **Type**: Tree-based Classifier
- **Features**: Depth control, feature importance analysis
- **Performance**:
  - F1-Score: 0.2738
  - AUC-ROC: 0.6437
- **Insights**: Provides interpretable decision rules for loan approval

## Results

### Model Performance Rankings

| Rank | Model | F1-Score | AUC-ROC | Accuracy | Recommendation |
|------|-------|----------|---------|----------|-----------------|
| 1 | Logistic Regression (Optimized) | 0.3774 | 0.7460 | 81.85% | ⭐ **Primary Model** |
| 2 | Random Forest (Tuned) | 0.3210 | 0.7093 | - | Alternative |
| 3 | Decision Tree | 0.2738 | 0.6437 | - | Interpretability |
| 4 | SVM Baseline | 0.1634 | 0.6066 | - | Further tuning needed |

### Output Artifacts

All results are stored in the `results/` directory:

- **CSV Reports**: Detailed metrics for each model
  - `logistic_regression_results.csv`
  - `rf_metrics.csv`
  - `svm_metrics_comparison.csv`
  - `decision_tree_results.csv`
  - `model_comparison_ranked_by_f1.csv`

- **Visualizations**: PNG files including
  - Confusion matrices (per model)
  - ROC curves (per model)
  - Feature importance plots
  - Threshold analysis charts
  - Model comparison plots
  - Learning curves

- **Summary**: `FINAL_SUMMARY.txt` with comprehensive findings and recommendations

## Technologies

### Core Machine Learning & Data Science
- **scikit-learn** (1.8.0) - ML algorithms, metrics, preprocessing
- **pandas** (3.0.1) - Data manipulation and analysis
- **numpy** (2.4.2) - Numerical computations
- **imbalanced-learn** - SMOTE for class imbalance handling
- **scipy** (1.17.0) - Scientific computing

### Visualization
- **matplotlib** (3.10.8) - Plotting and visualizations
- **seaborn** (0.13.2) - Statistical data visualization

### Development Environment
- **Jupyter** (1.1.1) / **JupyterLab** (4.5.4) - Interactive notebooks
- **IPython** (9.10.0) - Enhanced Python shell

### Project Management
- **Git** - Version control
- **Python 3.8+** - Programming language

## Key Findings

### Top Predictive Features
Based on feature importance analysis:
1. Age
2. Interest Rate
3. Months Employed
4. Loan Amount
5. Income

### Class Imbalance Insights
- Original dataset was heavily imbalanced (mostly non-defaults)
- SMOTE resampling applied to training set to improve minority class prediction
- Threshold optimization improved F1-score on imbalanced test data

### Model Selection Rationale
**Logistic Regression** selected as the primary model because:
- ✅ Highest F1-Score (0.3774) on imbalanced test set
- ✅ Highest AUC-ROC (0.7460) - good discrimination ability
- ✅ Interpretable coefficients for business stakeholders
- ✅ Fast inference time for production deployment
- ✅ Well-calibrated probability estimates

### Recommendations for Deployment
1. Use threshold of 0.56 for optimal F1-Score in loan approval system
2. Implement probability calibration for confidence-based decision support
3. Monitor model performance on new data and retrain periodically
4. Use feature importance for risk factor analysis
5. Consider ensemble approach combining Logistic Regression and Random Forest for robustness

## Project Workflow & Git History

The project has been developed iteratively with the following approach:

- **Modular Design**: Separate Jupyter notebooks for each model
- **Code Organization**: Reusable Python modules in `src/` directory
- **Version Control**: Git branches for individual model development
- **Incremental Integration**: Pull requests for model comparisons and final evaluation

Recent commits:
- Add model comparison functionality and evaluation metrics loading
- Merge decision-tree branch with updated feature importance
- Merge support-vector-machine implementation
- Merge logistic-regression with hyperparameter tuning
- Merge random-forest with threshold optimization

## Contributing

This is an academic project for SLIIT (Sri Lanka Institute of Information Technology).

## License

Academic use only

## Contact & Repository

- **Repository**: [Github: chamithusithmaka/MLproject](https://github.com/chamithusithmaka/MLproject)
- **Course**: IT4060 - Machine Learning (SLIIT Year 4 Semester 1)
- **Assignment**: ML Model Development and Comparison

---

**Last Updated**: March 2026
**Current Branch**: random-forest
**Status**: Active Development
