import os
import pandas as pd

from sklearn.linear_model import LogisticRegression

train_file = os.path.join('data', 'train.tsv')
test_file = os.path.join('data', 'test.tsv')
results_file = os.path.join('data', 'results.tsv')
output_file = os.path.join('data', 'out.tsv')

df_train_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
                  'Light', 'CO2', 'HumidityRatio']
df_train = pd.read_csv(train_file, sep='\t', names=df_train_names)
df_train = df_train.dropna()

df_test_names = ['Date', 'Temperature', 'Humidity',
                 'Light', 'CO2', 'HumidityRatio']
df_test = pd.read_csv(test_file, sep='\t', names=df_test_names)
df_test = df_test.dropna()

x_train = df_train[['Light']]
X_train = df_train[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
y_train = df_train.Occupancy

x_test = df_test[['Light']]
X_test = df_test[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
y_test = pd.read_csv(results_file, sep='\t', names=['Occupancy']).Occupancy

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

log_reg_all = LogisticRegression()
log_reg_all.fit(X_train, y_train)

y_train_pred = log_reg.predict(x_train)
y_test_pred = log_reg.predict(x_test)

train_accuracy = sum(y_train == y_train_pred) / len(y_train_pred)
train_sensitivity = sum(y_train & y_train_pred) / sum(y_train_pred)
train_specificity = sum(-~-(y_train | y_train_pred)) / (len(y_train_pred) - sum(y_train_pred))

test_accuracy = sum(y_test == y_test_pred) / len(y_test_pred)
test_sensitivity = sum(y_test & y_test_pred) / sum(y_test_pred)
test_specificity = sum(-~-(y_test | y_test_pred)) / (len(y_test_pred) - sum(y_test_pred))

y_train_all_pred = log_reg_all.predict(X_train)
y_test_all_pred = log_reg_all.predict(X_test)

train_all_accuracy = sum(y_train == y_train_all_pred) / len(y_train_all_pred)
train_all_sensitivity = sum(y_train & y_train_all_pred) / sum(y_train_all_pred)
train_all_specificity = sum(-~-(y_train | y_train_pred)) / (len(y_train_all_pred) - sum(y_train_all_pred))

test_all_accuracy = sum(y_test == y_test_all_pred) / len(y_test_all_pred)
test_all_sensitivity = sum(y_test & y_test_all_pred) / sum(y_test_all_pred)
test_all_specificity = sum(-~-(y_test | y_test_all_pred)) / (len(y_test_all_pred) - sum(y_test_all_pred))

pd.DataFrame(y_test_pred).to_csv(output_file, index=False, header=False)
