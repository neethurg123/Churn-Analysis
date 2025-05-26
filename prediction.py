# -*- coding: utf-8 -*-
"""Prediction.ipynb

**Import Libraries**
"""

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

"""**Load Cleaned and Merged Data for predictions**"""

df=pd.read_csv('data_for_predictions.csv')
df.head(5)

df.drop(['Unnamed: 0','id'],axis=1,inplace=True)

"""**Split Data for Train & Test**"""

(X_train,X_test,y_train,y_test)=train_test_split(df.drop('churn',axis=1),df['churn'],test_size=0.3,random_state=42)
X_train.shape,X_test.shape

"""**Model selection and training**"""

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=2,
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)
# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred=best_model.predict(X_test)

y_pred[:5]

y_test[:5]

best_model.score(X_test,y_test)

classification_report=sklearn.metrics.classification_report(y_test,y_pred)
print(classification_report)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

y_test.value_counts()

"""**Insights**

**1. Accuracy is high, but misleading:**

The overall accuracy is ~90.5%,
This suggests a high-performing model at first glance. However, accuracy can be misleading in cases of class imbalance

**2. Class Imbalance is evident:**

Class 0 (negative class) dominates the dataset with 3942 instances, while class 1 (positive class) only has 440 instances.This imbalance can cause the model to bias its predictions toward the majority class (class 0), leading to suboptimal performance for class 1.

**3. The model struggles with predicting class 1:**

False Negatives (FN): There are 416 instances where the model wrongly predicted class 0 instead of class 1. This means that class 1 instances are being missed in favor of predicting class 0.

False Positives (FP): Only 11 instances are wrongly predicted as class 1 when they are actually class 0. This shows that class 1 is rarely predicted incorrectly.

True Positives (TP): Only 24 instances of class 1 were correctly identified. This highlights a major issue in detecting the positive class, which could be critical depending on the application (e.g., fraud detection, rare disease diagnosis).

Precision and Recall for class 1 (positive class):

Precision (how many predicted class 1 are actually class 1):
The model has a decent precision for class 1 (about 68.6%), meaning that when it predicts class 1, it’s often correct, but it still has some false positives.

Recall (how many actual class 1 are detected):

Recall is very low for class 1 (5.45%), meaning the model is missing most of the actual class 1 instances. This is a major concern because it indicates that class 1 is being severely under-predicted.

Precision-Recall trade-off:

There’s a significant precision-recall trade-off. While precision is relatively okay for class 1 (68.6%), the recall is very poor (5.45%). This means that the model has a high tendency to predict the majority class (class 0), and when it does predict class 1, it's often correct, but it’s missing a lot of actual class 1 instances.

 ** Possible Solutions to Improve the Model:**

**Handle Class Imbalance:**

Oversample the minority class (class 1) or undersample the majority class (class 0) to make the dataset more balanced.

Alternatively, use SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for class 1.

**Use Class Weights:**

Assign higher class weights to the minority class (class 1) to penalize the model more for incorrect predictions on this class. This can encourage the model to focus on correctly classifying class 1.

**Use Different Evaluation Metrics:**

Accuracy alone is misleading. Focus on other metrics such as F1-score, Precision, and Recall:

F1-score for class 1: A balance between precision and recall, especially useful when you care equally about false positives and false negatives.

**Adjust the Decision Threshold:**

Consider adjusting the decision threshold to predict class 1 more often. The current threshold of 0.5 might not be ideal, especially when class 1 is underrepresented.
"""
