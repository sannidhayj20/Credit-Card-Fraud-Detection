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
  ```

2. **Mounting Google Drive:**
  ```python
   from google.colab import drive
   drive.mount('/content/drive')
  ```

## Exploratory Data Analysis

- **Initial Data Exploration:**
 ```python
  print(data.head())
  print(data.describe())
 ```

- **Distribution of Transaction Amounts:**
 ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  fig, axs = plt.subplots(1, 2, figsize=(20, 6))
  sns.histplot(legal_df['Amount'], bins=50, kde=True, ax=axs[0])
  axs[0].set_title('Legal Transaction Amount Distribution')
  sns.histplot(illegal_df['Amount'], bins=50, kde=True, ax=axs[1])
  axs[1].set_title('Illegal Transaction Amount Distribution')
  plt.show()
 ```
![image](https://github.com/user-attachments/assets/ed4eb856-0b30-46f1-9c89-8daf55d84ef0)


- **Correlation Analysis:**
 ```python
  correlation_matrix = df.corr()
  correlation_with_class = correlation_matrix['Class'].drop('Class')
  plt.figure(figsize=(10, 6))
  correlation_with_class.plot(kind='bar', color='skyblue')
  plt.title('Correlation with Class')
  plt.xlabel('Features')
  plt.ylabel('Correlation Coefficient')
  plt.xticks(rotation=45)
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.show()
```
![image](https://github.com/user-attachments/assets/71962e6a-4b96-4d1d-8cb8-97a4e580cf43)

## Feature Engineering

- **Filtering Most Correlated Features:**
 ```python
  most_correlated_features = correlation_with_class[correlation_with_class > 0].index.tolist()
 ```

- **Heatmap of Top Correlated Features:**
 ```python
  correlation_matrix_filtered = correlation_matrix.loc[positive_correlation_features, positive_correlation_features]
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix_filtered, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Correlation Heatmap of Features with Most Correlation to Class')
  plt.show()
``` 
![image](https://github.com/user-attachments/assets/29deba26-15b1-49af-867f-24af32209073)

## Model Training

- **Training XGBoost Model:**
 ```python
  from xgboost import XGBClassifier
  model = XGBClassifier()
  model.fit(X_res, y_res)
 ```

- **Cross-Validation:**
 ```python
  from sklearn.model_selection import cross_val_score
  model_cv = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=300, use_label_encoder=False, eval_metric='logloss')
  scores = cross_val_score(model_cv, X_res, y_res, cv=10, scoring='accuracy')
  print("Cross-validation scores:", scores)
  print("Average cross-validation score:", scores.mean())
 ```

## Model Evaluation

- **Validation and Testing:**
 ```python
  from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

  # Validation
  y_val_pred = model_final.predict(X_val)
  y_val_proba = model_final.predict_proba(X_val)[:, 1]
  print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
  print("Validation ROC-AUC:", roc_auc_score(y_val, y_val_proba))

  # Testing
  y_test_pred = final_model.predict(X_test)
  y_test_proba = final_model.predict_proba(X_test)[:, 1]
  print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
  print("Test ROC-AUC:", roc_auc_score(y_test, y_test_proba))
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
  print("Classification Report:\n", classification_report(y_test, y_test_pred))
 ```
![image](https://github.com/user-attachments/assets/2f58fbd8-6082-4019-9187-07011530a32e)


## Conclusion

This project successfully demonstrates the use of XGBoost for credit card fraud detection. The model is evaluated using accuracy, ROC-AUC, and other classification metrics, showing promising results in identifying fraudulent transactions.

## How to Run

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/sannidhayj20/Credit-Card-Fraud-Detection.git
   ```

2. Install the required packages:
   bash
   pip install -r requirements.txt
   

3. Open the Jupyter notebook and run all cells:
   bash
   jupyter notebook CreditCardFraudDetectionLatest.ipynb
   

## License

Feel free to modify the GitHub repository link and other details as needed!
