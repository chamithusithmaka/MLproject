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
| `loan_id` | Unique identifier for each loan |
| `customer_id` | Unique identifier for each customer |
| `loan_amount` | Total amount of the loan |
| `term` | Loan term (e.g., 36 or 60 months) |
| `interest_rate` | Interest rate applied to the loan |
| `grade` | Loan grade assigned by lender |
| `sub_grade` | More detailed sub-grade of the loan |
| `employment_length` | Length of employment in years |
| `home_ownership` | Home ownership status (e.g., RENT, OWN, MORTGAGE) |
| `annual_income` | Annual income of the borrower |
| `verification_status` | Whether the income is verified |
| `purpose` | Purpose of the loan (e.g., debt consolidation, credit card) |
| `dti` | Debt-to-income ratio |
| `delinquency_2yrs` | Number of 30+ days past-due incidences of delinquency in the past 2 years |
| `fico_range_low` | Lower bound of FICO credit score |
| `fico_range_high` | Upper bound of FICO credit score |
| `open_acc` | Number of open credit accounts |
| `total_acc` | Total number of credit accounts |
| `loan_status` | **Target variable:** whether the loan defaulted (1 = default, 0 = paid) |

> **Note:** `loan_status` is the column you will predict using machine learning algorithms.

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
