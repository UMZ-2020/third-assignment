import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score

import seaborn as sns
import matplotlib.pyplot as plt


data_file = os.path.join('data', 'train.tsv')
test_data_file = os.path.join('data', 'test.tsv')
results_file = os.path.join('data', 'results.tsv')
output_file = os.path.join('data', 'out.tsv')

df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']
df = pd.read_csv(data_file, sep='\t', names=df_names)
df = df.dropna()
df.describe()
df.head(20)

occupancy_percentage = float(sum(df.Occupancy)) / float(len(df))
print("Occupancy percentage is: " + str(occupancy_percentage))
print("Zero rule model accurancy on training set is: " + str
      (1 - occupancy_percentage))

# logistic regression classifier on one independent variable
clf = LogisticRegression()
X_train = df[['Temperature']]
y_train = df.Occupancy

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

# accuracy
clf_accuracy = sum(y_train == y_train_pred) / len(df)
print("Training set accuracy for logisitic regression model " + "on
      Temperature variable: \n" + str(clf_accuracy))

# sensitivity, also called recall
recall_score(y_train, y_train_pred)

# specificity
conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp
specificity = tn/(tn+fp)
print("Training set specificity for logisitic regression model " + "on
      Temperature variable: \n" + str(specificity))

# test dataset for one variable
df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_column_name = ['Temperature']
X_test = pd.read_csv(test_data_file, sep='\t', names=df_column_names,
                     usecols=X_column_name)

df_results = pd.read_csv(results_file, sep='\t', names=['y'])
df_results['y'] = df_results['y'].astype('category')
y_true = df_results['y']
y_test_pred = clf.predict(X_test)

# accuracy
clf_test_accuracy = accuracy_score(y_true, y_test_pred)
print('Accuracy on test dataset: ' + str(clf_test_accuracy))

# sensitivity, also called recall
recall_score(y_true, y_test_pred)

# specificity
conf_matrix = confusion_matrix(y_true, y_test_pred)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp
specificity_test = tn/(tn+fp)
print("Test set specificity for logisitic regression model " + "on
      Temperature variable: \n" + str(specificity_test))

plot_confusion_matrix(clf, X_test, y_true)
plt.show()

plot_confusion_matrix(clf, X_test, y_true, normalize='true')
plt.show()

# logistic regression classifier on all but date independent variables
clf_all = LogisticRegression()
X_train_all = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
clf_all.fit(X_train_all, y_train)

y_train_pred_all = clf_all.predict(X_train_all)

# accuracy
clf_all_accuracy = accuracy_score(y_train, y_train_pred_all)
print("Training set accuracy for logisitic regression model " +
      "on all but date variable:\n" + str(clf_all_accuracy))

# sensitivity, also called recall
recall_score(y_train, y_train_pred_all)

# specificity
conf_matrix = confusion_matrix(y_train, y_train_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp
specificity = tn/(tn+fp)
print("Training set specificity for logisitic regression model " + "on
      all but date variable: \n" + str(specificity))

# test dataset for all but date independent variables
df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_column_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']

X_test_all = pd.read_csv(test_data_file, sep='\t', names=df_column_names,
                         usecols=X_column_names)

df_results = pd.read_csv(results_file, sep='\t', names=['y'])
df_results['y'] = df_results['y'].astype('category')

y_true_all = df_results['y']
y_test_pred_all = clf_all.predict(X_test_all)

# accuracy
clf_test_accuracy = accuracy_score(y_true_all, y_test_pred_all)
print('Accuracy on test dataset (full model): ' + str(clf_test_accuracy))

# sensitivity, also called recall
recall_score(y_true_all, y_test_pred_all)

# specificity
conf_matrix = confusion_matrix(y_true_all, y_test_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp
specificity_test = tn/(tn+fp)
print("Specificity on test dataset (full model): " + str(specificity_test))

plot_confusion_matrix(clf_all, X_test_all, y_true_all)
plt.show()

plot_confusion_matrix(clf_all, X_test_all, y_true_all, normalize='true')
plt.show()

df = pd.DataFrame(y_test_pred)
df.to_csv(output_file, index=False, header=False)

