import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.Accessing the data

df = pd.read_csv('./data.csv')
# Prints the first 5 rows with the name of columns.
print(df.head())
# Describe the whole columns value as range, mean, no. of rows
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.drop(columns=['Unnamed: 32'], inplace=True))

# Model Data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

X = df.drop(columns=['diagnosis'])
y = df.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

#Malignanat is 0 and benign is 1
le = LabelEncoder()
y = le.fit_transform(y)

# With logistics regression
lr = LogisticRegression()  # creating object with default parameters
lr.fit(X_train, y_train)  # fit will assign weights to all parameters and will return a model

y_pred = lr.predict(X_test)
print("Confusion matrix for LR is :")
print(confusion_matrix(y_test, y_pred))

# With Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Confusion matrix of RandomForestClassifier is :")
print(confusion_matrix(y_test, y_pred))

# NaiveBaye's
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("Confusion matrix of Gaussian is :")
print(confusion_matrix(y_test, y_pred))

# K nearest neighbour
neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print("Confusion matrix of KNeighborsClassifier is :")
print(confusion_matrix(y_test, y_pred))

# Support Vector Machine
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Confusion matrix of svm is :")
print(confusion_matrix(y_test, y_pred))
