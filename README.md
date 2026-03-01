# Loan Approval Prediction: MLP vs AdaBoost
## Overview

This project investigates the effectiveness of a Neural Network model (Multilayer Perceptron – MLP) compared to a Machine Learning ensemble model (AdaBoost) for automated loan approval prediction.

The objective is to evaluate how a neural approach differs from a strong traditional machine learning baseline under identical preprocessing and evaluation conditions.

The study uses a large-scale loan approval dataset containing 148,670 loan applications and follows a rigorous machine learning workflow including preprocessing, feature selection, model training, and evaluation.

---
## Research Objective

The main goal of this project is to:

Train and evaluate a Multilayer Perceptron (MLP) neural network.

Implement AdaBoost as a strong ML baseline.

Compare both models using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Analyze differences between neural and ensemble learning approaches.

---
## Dataset

Source: Public Kaggle Loan Approval Dataset

Size: 148,670 applications

Task: Binary Classification

0 → Approved

1 → Denied

Features Include:

Loan amount

Income

Credit score

Age group

Loan type

Loan purpose

Region

Business/commercial indicator

Pre-approval status

And additional financial attributes

## Data Preprocessing

The following steps were applied:

### 1.Data Cleaning

Mean imputation for numerical features

Median-based encoding for categorical features

Removal of highly missing features

Removal of redundant/highly correlated features

### 2.Encoding

Ordinal encoding for age groups

Encoding of categorical variables

Boolean conversion to numerical

### 3.Feature Scaling

Normalization for neural network stability

### 4.Train/Validation/Test Split

80% / 10% / 10%

Stratified sampling

Fixed random seed for reproducibility

---

## Models  

                    1. Multilayer Perceptron (MLP)

Architecture:

Input layer

Hidden Layer 1: 64 neurons (ReLU)

Dropout (0.2)

Hidden Layer 2: 32 neurons (ReLU)

Dropout (0.2)

Output Layer: 1 neuron (Sigmoid)

Training:

Optimizer: Adam (lr = 0.001)

Loss: Binary Crossentropy

Epochs: 10

Batch size: 64

Total parameters: 4,673

                      2. AdaBoost Classifier

Configuration:

Base estimator: Decision Tree stump (max_depth=1)

100 estimators

Learning rate = 1.0

Fixed random seed

#### Results
MLP Performance
| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.32% |
| Precision | 94.06% |
| Recall    | 99.45% |
| F1-score  | 96.68% |
| ROC-AUC   | 0.9977 |

AdaBoost Performance
| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.99% |
| Precision | 96.21% |
| Recall    | 99.77% |
| F1-score  | 97.99% |
| ROC-AUC   | 0.9985 |

## Performance Comparison

Both models achieved outstanding performance.

AdaBoost slightly outperformed MLP across all evaluation metrics; however, the difference is minimal (~0.005), confirming that neural network approaches are highly competitive with strong ensemble methods.

## Key Findings

MLP effectively captures nonlinear financial relationships.

AdaBoost benefits from adaptive ensemble learning.

Feature selection improved methodological validity.

Both models are suitable for automated loan approval systems.

## Limitations

Single dataset used.

Moderate class imbalance (75% approved / 25% denied).

No explainability methods implemented (future improvement).

## Future Work

Advanced neural architectures

Explainable AI (SHAP, LIME)

Multi-dataset validation

Multi-class decision framework (Approve / Review / Reject)

## Installation
git clone https://github.com/yourusername/loan-approval-mlp-vs-adaboost.git
cd loan-approval-mlp-vs-adaboost
pip install -r requirements.txt  

## Run the Project
python src/mlp_model.py
python src/adaboost_model.py


## Author:
Fatjona Murrani
