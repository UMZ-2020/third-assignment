# importing libraries and setting options
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# setting paths to files
path = "J:/Studia/Uczenie_maszynowe/third-assignment-master/data"
data_file = os.path.join(path, "train.tsv")
test_data_file = os.path.join(path, "test.tsv")
results_file = os.path.join(path, "results.tsv")
output_file = os.path.join(path, 'out.tsv')

# naming columns and opening file with training set, droping NA values
df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']
df = pd.read_csv(data_file, sep='\t', names=df_names)
df = df.dropna()
print(df.describe())

# counting occupancy percentage and zero rule model accuracy
occupancy_percentage = sum(df.Occupancy) / len(df)
print("Occupancy percentage is: " + str(occupancy_percentage))
print("Zero rule model accuracy on training set is: "
      + str(1 - occupancy_percentage))

# creating logistic regression object, fitting the model with single column Humidity
clf = LogisticRegression()
X_train = df[['Humidity']]
y_train = df.Occupancy
clf.fit(X_train, y_train)

# predicting the value for training set on column 'Humidity' and counting accuracy
y_train_pred = clf.predict(X_train)
clf_accuracy = accuracy_score(y_train, y_train_pred)
print("Training set accuracy for logisitic regression model "
      + "on Humidity variable: " + str(clf_accuracy))

# creating the confusion matrix
conf_matrix1 = confusion_matrix(y_train, y_train_pred)
tn1, fp1, fn1, tp1 = conf_matrix1.ravel()

# counting the sensivity and specifity
sensitivity1 = tp1 / (tp1 + fn1)
specificity1 = tn1 / (fp1 + tn1)
print("Training set sensivity for logisitic regression model "
      + "on Humidity variable: " + str(sensitivity1))
print("Training set specificity for logisitic regression model "
      + "on Humidity variable: " + str(specificity1))

# creating logistic regression object, fitting the model with all variables
clf_all = LogisticRegression()
X_train_all = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
clf_all.fit(X_train_all, y_train)

# predicting the value for training set and counting accuracy
y_train_pred_all = clf_all.predict(X_train_all)
clf_all_accuracy = accuracy_score(y_train, y_train_pred_all)
print("Training set accuracy for logisitic regression model " +
      "on all variables: " + str(clf_all_accuracy))

# creating the confusion matrix
conf_matrix2 = confusion_matrix(y_train, y_train_pred_all)
tn2, fp2, fn2, tp2 = conf_matrix2.ravel()

# counting the sensivity and specifity
sensitivity2 = tp2 / (tp2 + fn2)
specificity2 = tn2 / (fp2 + tn2)
print("Training set sensivity for logisitic regression model "
      + "on all variables: " + str(sensitivity2))
print("Training set specificity for logisitic regression model "
      + "on all variables: " + str(specificity2))

# naming columns
df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_column_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
# opening file with testing set and results
X_test = pd.read_csv(test_data_file, sep='\t', names=df_column_names, usecols=X_column_names)
df_results = pd.read_csv(results_file, sep='\t', names=['y'])

# changing type of results column to 'category'
df_results['y'] = df_results['y'].astype('category')
# assigning to variable
y_true = df_results['y']

# predicting the value for testing set on reshaped column 'Humidity' and counting accuracy
y_test_pred = clf.predict(X_test['Humidity'].values.reshape((-1, 1)))
clf_test_accuracy = accuracy_score(y_true, y_test_pred)
print('Accuracy on Humidity column for test dataset: ' + str(clf_test_accuracy))

# creating the confusion matrix
conf_matrix3 = confusion_matrix(y_true, y_test_pred)
tn3, fp3, fn3, tp3 = conf_matrix3.ravel()

# counting the sensivity and specifity
sensitivity3 = tp3 / (tp3 + fn3)
specificity3 = tn3 / (fp3 + tn3)
print("Testing set sensivity for logisitic regression model "
      + "on Humidity variable: " + str(sensitivity3))
print("Testing set specificity for logisitic regression model "
      + "on Humidity variable: " + str(specificity3))

# predicting the value for testing set and counting accuracy
y_test_pred_all = clf_all.predict(X_test)
clf_test_all_accuracy = accuracy_score(y_true, y_test_pred_all)
print('Accuracy on test dataset (full model): ' + str(clf_test_all_accuracy))

# creating the confusion matrix
conf_matrix4 = confusion_matrix(y_true, y_test_pred_all)
tn4, fp4, fn4, tp4 = conf_matrix4.ravel()

# counting the sensivity and specifity
sensitivity4 = tp4 / (tp4 + fn4)
specificity4 = tn4 / (fp4 + tn4)
print("Testing set sensivity for logisitic regression model "
      + "on all variables: " + str(sensitivity4))
print("Testing set specificity for logisitic regression model "
      + "on all variables: " + str(specificity4))

# creating a DataFrame from predicted value of testing set and exporting to csv file
df_export = pd.DataFrame(y_test_pred)
df_export.to_csv(output_file, index=False, header=False)
