import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt

df = pd.read_csv('survey_results_public.csv',
                         usecols=[ 'YearsCode', 'Respondent','Age', 'Hobbyist','Student', 'YearsCodePro'],
                         index_col='Respondent')

#remove na
df.dropna(inplace=True)

print(df)
#replace values"Yes" to 1 ; "No" to 0
df.replace(to_replace={'Yes': '1',
                       'No': '0'}, inplace=True)

#replace values "Yes, part-time" to '1' and 'Yes, full-time' to 1
df.replace(to_replace={'Yes, part-time': '1',
                       'Yes, full-time': '1'}, inplace=True)
print(df)

#replace values 'Less than 1 year' to '0' and More than 50 years' to 51
df.replace(to_replace={'Less than 1 year': '0',
                               'More than 50 years': '51'},inplace=True)

df = df.astype('int64')

print(df)

# logistic regression
clf = LogisticRegression()
X_train = df[['Hobbyist','Age', 'YearsCode', 'YearsCodePro']]
y_train = df.Student
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)

# accuracy
accuracy = accuracy_score(y_train, y_train_pred)
print("Accurency (training data):", accuracy)
# sensitivity
sensitivity = recall_score(y_train, y_train_pred)
print("Sensitivity (training data):", sensitivity)
#calculate specificity
conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn/(tn+fp)
print("Specificity (training data):",specificity)


#dividing the set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                    y_train, test_size=0.33,
                                                    random_state=0)
# logistic regression classifier on all 
clf_divided = LogisticRegression()
clf_divided.fit(X_train, y_train)
y_pred = clf_divided.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accurency (test data):", accuracy)
# sensitivity
sensitivity = recall_score(y_test, y_pred)
print("Sensitivity (test data):", sensitivity)

#calculate  specificity
matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn/(tn+fp)
print("Specificity (test data):",specificity)
