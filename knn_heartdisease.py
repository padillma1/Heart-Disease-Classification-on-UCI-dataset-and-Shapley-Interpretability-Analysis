# -*- coding: utf-8 -*-
"""KNN_HeartDisease.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1twojpk0Pq6fQKRoT0VzE0tLslg5EMH9K
"""

pip install shap

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import shap

#dataset = pd.read_csv("/content/heart.csv")
dataset1 = pd.read_csv("processed.cleveland.data", header=None)  # Assuming no header in the dataset
dataset2 = pd.read_csv("processed.hungarian.data", header=None)
dataset3 = pd.read_csv("processed.switzerland.data", header=None)
dataset4 = pd.read_csv("processed.va.data", header=None)

# Print the number of instances (examples) and features in each dataset (center)
num_instances1, num_features1 = dataset1.shape
num_instances2, num_features2 = dataset2.shape
num_instances3, num_features3 = dataset3.shape
num_instances4, num_features4 = dataset4.shape
print("Number of instances dataset 1:", num_instances1)
print("Number of features dataset 1:", num_features1-1)

print("Number of instances dataset 2:", num_instances2)
print("Number of features dataset 2:", num_features2-1)

print("Number of instances dataset 3:", num_instances3)
print("Number of features dataset 3:", num_features3-1)

print("Number of instances dataset 4:", num_instances4)
print("Number of features dataset 4:", num_features4-1)

# Replace '?' with NaN and then drop rows with NaN values (FLamby method)
dataset1 = dataset1.replace("?", np.NaN).drop([10, 11, 12], axis=1).dropna(axis=0)
dataset2 = dataset2.replace("?", np.NaN).drop([10, 11, 12], axis=1).dropna(axis=0)
dataset3 = dataset3.replace("?", np.NaN).drop([10, 11, 12], axis=1).dropna(axis=0)
dataset4 = dataset4.replace("?", np.NaN).drop([10, 11, 12], axis=1).dropna(axis=0)


# Only replace '?' with NaN
#dataset1 = dataset1.replace("?", np.NaN).dropna(axis=0)
#dataset2 = dataset2.replace("?", np.NaN).dropna(axis=0)
#dataset3 = dataset3.replace("?", np.NaN).dropna(axis=0)
#dataset4 = dataset4.replace("?", np.NaN).dropna(axis=0)

dataset1 = dataset1.apply(pd.to_numeric)
dataset2 = dataset2.apply(pd.to_numeric)
dataset3 = dataset3.apply(pd.to_numeric)
dataset4 = dataset4.apply(pd.to_numeric)

num_instances1, num_features1 = dataset1.shape
num_instances2, num_features2 = dataset2.shape
num_instances3, num_features3 = dataset3.shape
num_instances4, num_features4 = dataset4.shape
print("Number of instances dataset 1:", num_instances1)
print("Number of features dataset 1:", num_features1-1)

print("Number of instances dataset 2:", num_instances2)
print("Number of features dataset 2:", num_features2-1)

print("Number of instances dataset 3:", num_instances3)
print("Number of features dataset 3:", num_features3-1)

print("Number of instances dataset 4:", num_instances4)
print("Number of features dataset 4:", num_features4-1)

# Concatenate the datasets vertically (row-wise)
combined_dataset = pd.concat([dataset1, dataset2, dataset3, dataset4], ignore_index=True)

# Add feature labels for the remaining 10 features and the label column
feature_labels = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
label = 'num'  # The label column

# Assign feature labels
combined_dataset.columns = feature_labels + [label]

# Print the number of instances (examples) and features in the combined data
num_instances, num_features = combined_dataset.shape
print("Number of instances:", num_instances)
print("Number of features:", num_features-1)
print(combined_dataset.columns)

# Add the column name 'label' to the top of the combined label column
#combined_dataset.columns = ['label'] + list(range(1, combined_dataset.shape[1]))

X = combined_dataset.iloc[:,:-1]
y = combined_dataset.iloc[:,-1].values
label_column = combined_dataset.iloc[:, -1]  # Extract the last column
# Print the range of values in the last column
print("Range of values in the last column:")
print(label_column.describe())
print(combined_dataset.columns)

# Convert labels > 0 to 1
label_column.where(y == 0, 1, inplace=True)
label_column = combined_dataset.iloc[:, -1]  # Extract the last column

# Print the range of values in the last column
print("Range of values in the last column:")
print(label_column.describe())

np.random.seed(42)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=42)

# Initializing and fitting the model
#knn = KNN(k=10)
#knn.fit(X_train, y_train)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Fitting the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Making predictions
predictions = knn_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

import shap

# Explain the model's predictions using SHAP values
explainer = shap.Explainer(knn_classifier.predict, X_train)
shap_values = explainer(X_test)

# Visualize the SHAP values for a single prediction
#shap.summary_plot(shap_values, X_test)
shap.plots.bar(shap_values)