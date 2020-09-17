import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

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

# logistic regression classifier on one independent variable - Light
clf = LogisticRegression()
X_train = df[['Light']]
y_train = df.Occupancy

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

# accuracy
clf_accuracy = sum(y_train == y_train_pred) / len(df)

#sensitivity
recall_score(y_train, y_train_pred)

#calculate specificity
conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp
specificity = tn/(tn+fp)

#measure f for train data
f1_score(y_train, y_train_pred)
# In the room should be a person
fbeta_score(y_train, y_train_pred, beta=0.5)
# In the room shouldn't be a person
fbeta_score(y_train, y_train_pred, beta=2)

# logistic regression classifier on all but date independent variables
clf_all = LogisticRegression()
X_train_all = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
clf_all.fit(X_train_all, y_train)

y_train_pred_all = clf_all.predict(X_train_all)

#calculate accuracy
clf_all_accuracy = accuracy_score(y_train, y_train_pred_all)

#calculate sensitivity
recall_score(y_train, y_train_pred_all)

#calculate specificity
conf_matrix = confusion_matrix(y_train, y_train_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp 
specificity = tn/(tn+fp)

#measure f for train data
f1_score(y_train, y_train_pred_all)
# In the room should be a person
fbeta_score(y_train, y_train_pred_all, beta=0.5)
# In the room shouldn't be a person
fbeta_score(y_train, y_train_pred_all, beta=2)


df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_column_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']

X_test_all = pd.read_csv(test_data_file, sep='\t', names=df_column_names, usecols=X_column_names)
df_results = pd.read_csv(results_file, sep='\t', names=['y'])
df_results['y'] = df_results['y'].astype('category')

y_true = df_results['y']
y_test_pred = clf_all.predict(X_test_all)
#calculate accuracy
clf_test_accuracy = accuracy_score(y_true, y_test_pred)
#calculate sensitivity
recall_score(y_true, y_test_pred)
#calculate specificity
conf_matrix = confusion_matrix(y_true, y_test_pred)
tn, fp, fn, tp = conf_matrix.ravel()
tn, fp, fn, tp 
specificity = tn/(tn+fp)

#measure f for test data
f1_score(y_true, y_test_pred)
# In the room should be a person
fbeta_score(y_true, y_test_pred, beta=0.5)
# In the room shouldn't be a person
fbeta_score(y_true, y_test_pred, beta=2)


pd.DataFrame(y_test_pred).to_csv(output_file, index=False, header=False)


