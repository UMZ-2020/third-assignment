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

df = pd.read_csv('survey_results_public.csv', usecols=['Hobbyist',
                                                       'Respondent',
                                                       'YearsCode',
                                                       'Age1stCode',
                                                       'YearsCodePro',
                                                       'WorkWeekHrs',
                                                       'Age'],
                 index_col='Respondent')

df.shape
df.dropna(inplace=True)

column_values = df[['Hobbyist']].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)
df.replace(to_replace={'Yes': '1',
                       'No': '0'}, inplace=True)
df.replace(to_replace={'Less than 1 year': '0',
                       'More than 50 years': '51'}, inplace=True)
df.replace(to_replace={'Younger than 5 years': '4', 'Older than 85': '86'},
           inplace=True)

df.dtypes
df = df.astype('int64', copy=False)

df.describe()
df.head(20)

# Occupancy
occupancy_percentage = sum(df.Hobbyist) / len(df)
print("Occupancy percentage is: " + str(occupancy_percentage))
print("Zero rule model accurancy on training set is: " + str
      (1 - occupancy_percentage))

# logistic regression classifier on all but Respondent independent variables
clf_all = LogisticRegression()
X_train_all = df[['YearsCode', 'Age1stCode', 'YearsCodePro',
                  'WorkWeekHrs', 'Age']]
y_train = df.Hobbyist
clf_all.fit(X_train_all, y_train)

y_train_pred_all = clf_all.predict(X_train_all)

# accuracy
clf_all_accuracy = accuracy_score(y_train, y_train_pred_all)
print("Training set accuracy for logisitic regression model " +
      "on all but Respondent variable:\n" + str(clf_all_accuracy))

# sensitivity, also called recall
recall_score(y_train, y_train_pred_all)

# specificity
conf_matrix = confusion_matrix(y_train, y_train_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp
specificity = tn/(tn+fp)
print("Training set specificity for logisitic regression model " + "on
      all but Respondent variable: \n" + str(specificity))

# division into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_all,
                                                    y_train, test_size=0.33,
                                                    random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# logistic regression classifier on all but Respondent independent variables
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# accuracy
clf_accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy for logisitic regression model " +
      "on all but Respondent variable:\n" + str(clf_all_accuracy))

# sensitivity, also called recall
recall_score(y_test, y_pred)

# specificity
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp
specificity = tn/(tn+fp)
print("Test set specificity for logisitic regression model " + "on
      all but Respondent variable: \n" + str(specificity))

plot_confusion_matrix(clf, X_test, y_test, normalize='true')
plt.show()

