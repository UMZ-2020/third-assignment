import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df_results = pd.read_csv('survey_results_public.csv',
                         usecols=['Respondent', 'Student', 'Age', 'Hobbyist', 'YearsCode', 'Dependents'],
                         index_col='Respondent')

df_results.dropna(inplace=True)
df_results.replace(to_replace={'No': '0',
                               'Yes': '1',
                               'Yes, part-time': '1',
                               'Yes, full-time': '1'},
                   inplace=True)
df_results.replace(to_replace={'Younger than 5 years': '4',
                               'Older than 85': '86'},
                   inplace=True)
df_results.replace(to_replace={'Less than 1 year': '0',
                               'More than 50 years': '51'},
                   inplace=True)

df_results[['Student', 'Age', 'Hobbyist', 'YearsCode', 'Dependents']] = df_results[
    ['Student', 'Age', 'Hobbyist', 'YearsCode', 'Dependents']].astype(int)

X_train = df_results[['Age', 'Hobbyist', 'YearsCode', 'Dependents']]

Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1

df_results = df_results[~((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).any(axis=1)]

X_train, X_test, y_train, y_test = train_test_split(
    df_results[['Age', 'Hobbyist', 'YearsCode', 'Dependents']], df_results.Student, test_size=0.2, random_state=777)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_test_pred = log_reg.predict(X_test)

zer_rule_accuracy = 1 - (sum(y_test) / len(y_test))
train_accuracy = sum(y_test == y_test_pred) / len(y_test_pred)
train_sensitivity = sum(y_test & y_test_pred) / sum(y_test_pred)
train_specificity = sum(-~-(y_test | y_test_pred)) / (len(y_test_pred) - sum(y_test_pred))
f_train = f1_score(y_test, y_test_pred)
