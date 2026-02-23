# Loan Default Prediction Dataset

## Overview

The Loan Default Prediction dataset is designed to help predict which individuals are at risk of defaulting on their loan payments. This dataset is particularly relevant for financial institutions, banks, and lending companies aiming to reduce default rates and target interventions effectively. By applying machine learning models, you can identify high-risk borrowers and support decision-making for loan approvals or risk mitigation strategies.

## Source

The dataset is taken from **Coursera's Loan Default Prediction Challenge** and provides a real-world scenario for practicing supervised learning on a financial dataset.

## Dataset Details

- **Number of rows:** 255,347
- **Number of columns:** 18

## Features

The dataset includes the following attributes:

| Feature Name | Description |
|--------------|-------------|
| `LoanID` | Unique identifier for each loan |
| `Age` | Age of the borrower |
| `Income` | Annual income of the borrower |
| `LoanAmount` | Total amount of the loan |
| `CreditScore` | Credit score of the borrower |
| `MonthsEmployed` | Number of months the borrower has been employed |
| `NumCreditLines` | Number of open credit lines |
| `InterestRate` | Interest rate applied to the loan |
| `LoanTerm` | Loan term in months |
| `DTIRatio` | Debt-to-income ratio |
| `Education` | Education level (High School, Bachelor's, Master's, PhD) |
| `EmploymentType` | Type of employment (e.g., Full-time, Part-time, Self-employed, Unemployed) |
| `MaritalStatus` | Marital status of the borrower (e.g., Single, Married, Divorced) |
| `HasMortgage` | Whether the borrower has a mortgage (Yes/No) |
| `HasDependents` | Whether the borrower has dependents (Yes/No) |
| `LoanPurpose` | Purpose of the loan (e.g., Home, Auto, Education, Business, Other) |
| `HasCoSigner` | Whether the loan has a co-signer (Yes/No) |
| `Default` | **Target variable:** whether the loan defaulted (1 = default, 0 = paid) |

> **Note:** `Default` is the column you will predict using machine learning algorithms.

## Objective

The main goal of this dataset is to apply supervised machine learning algorithms to predict whether a borrower will default on their loan. Models can include:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- And others

## Applications

- **Credit risk assessment** for banks and lenders
- **Early detection** of high-risk borrowers
- **Financial decision-making** and loan approval automation

## File Information

- **Filename:** `Loan_default.csv`
- **Format:** CSV (Comma-Separated Values)
- **Encoding:** UTF-8
