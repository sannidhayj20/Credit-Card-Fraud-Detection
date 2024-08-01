# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used for this project is highly imbalanced, with a very small percentage of transactions being fraudulent. To address this, various data preprocessing and modeling techniques are employed to improve the detection accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

## Project Overview

The primary goal of this project is to build a machine learning model that can accurately identify fraudulent credit card transactions. The XGBoost algorithm is used for this purpose due to its robustness and performance.

## Data Preprocessing

1. **Loading Data:**
   ```python
   import pandas as pd
   data = pd.read_csv('/content/drive/MyDrive/creditcard.csv')
   df = data
