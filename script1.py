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

#Read training data file 
df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']

df = pd.read_csv(data_file, sep='\t', names=df_names)
df = df.dropna()

# logistic regression  on one variable - Light
clf = LogisticRegression()
X_train = df[['Light']]
y_train = df.Occupancy
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

# accuracy for logistic regression model on Light
accuracy = accuracy_score(y_train, y_train_pred)

#sensitivity for logistic regression model on Light
sensitivity = recall_score(y_train, y_train_pred)

conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()
#calculate specificity for logistic regression model on Light
specificity = tn/(tn+fp)

#measure f for train data (one variable)
f1_train = f1_score(y_train, y_train_pred)
# In the room should be a person
f_beta_train_a = fbeta_score(y_train, y_train_pred, beta=0.5)
# In the room shouldn't be a person
f_beta_train_b = fbeta_score(y_train, y_train_pred, beta=2)


# logistic regression  on all 
clf_all = LogisticRegression()
X_train_all = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
clf_all.fit(X_train_all, y_train)
y_train_pred_all = clf_all.predict(X_train_all)

#calculate accuracy for logistic regression model on all variables
all_accuracy = accuracy_score(y_train, y_train_pred_all)

#calculate sensitivity for logistic regression model on all variables
al_sensitivity = recall_score(y_train, y_train_pred_all)

#calculate specificity for logistic regression model on all variables
conf_matrix = confusion_matrix(y_train, y_train_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
all_specificity = tn/(tn+fp)

#measure f for train data
f1_train_all = f1_score(y_train, y_train_pred_all)
# In the room should be a person
f_beta_train_a_all = fbeta_score(y_train, y_train_pred_all, beta=0.5)
# In the room shouldn't be a person
f_beta_train_b_all = fbeta_score(y_train, y_train_pred_all, beta=2)



# Read testing data file
df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_column_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
X_test_all = pd.read_csv(test_data_file, sep='\t', names=df_column_names, usecols=X_column_names)
X_test_all = X_test_all.dropna()

# Read result
df_results = pd.read_csv(results_file, sep='\t', names=['y'])
df_results['y'] = df_results['y'].astype('category')

#Logistic regression  on all testing variables 
y_true = df_results['y']
y_test_pred_all = clf_all.predict(X_test_all)

#test accuracy  on all variables
all_test_accuracy = accuracy_score(y_true, y_test_pred_all)
#test sensitivity on all variables
all_test_sensitivity = recall_score(y_true, y_test_pred_all)
#test specificity  on all variables
conf_matrix = confusion_matrix(y_true, y_test_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()
all_test_specificity = tn/(tn+fp)

#measure f for test data (all variables)
f1_test_all = f1_score(y_true, y_test_pred_all)
# In the room should be a person
f_bata_test_a_all = fbeta_score(y_true, y_test_pred_all, beta=0.5)
# In the room shouldn't be a person
f_beta_test_b_all = fbeta_score(y_true, y_test_pred_all, beta=2)

#Logistic regression  on Light variable (testing data)
x_test = X_test_all [['Light']]
y_test_pred = clf.predict(x_test)

# accuracy for logistic regression model on Light
test_accuracy = accuracy_score(y_true, y_test_pred)

# sensitivity for logistic regression model on Light
test_sensitivity = recall_score(y_true, y_test_pred)

#specificity for logistic regression model on Light

conf_matrix = confusion_matrix(y_true, y_test_pred)
tn, fp, fn, tp = conf_matrix.ravel()
test_specificity = tn / (tn + fp)

#measure f for test data (one variable)
f1_test = f1_score(y_true, y_test_pred)
# In the room should be a person
f_beta_test_a = fbeta_score(y_true, y_test_pred, beta=0.5)
# In the room shouldn't be a person
f_beta_test_b = fbeta_score(y_true, y_test_pred, beta=2)

# Save results 
out_file = os.path.join('data', 'out.tsv')
df = pd.DataFrame(y_test_pred, y_test_pred_all)
df.to_csv(out_file, index=False, header=False)


