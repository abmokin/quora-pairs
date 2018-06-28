import os
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import ttest_ind
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import difflib
from quora_sem_utils import *
import time
import winsound

# Start timer
start_time = time.time()

# Load training set
# data = pd.read_csv(get_current_path() + '/data/train.csv', header=0, nrows=0000)   # for nrows
data = pd.read_csv(get_current_path() + '/data/train.csv', header=0)             # for all rows
data = data.dropna()

# Rename columns in the dataframe
data.rename(columns={'question1': 'q1', 'question2': 'q2', 'is_duplicate': 'y'}, inplace=True)

# Filter both questions -- filter \"
data['q1'] = data.q1.apply(signs_filter)
data['q2'] = data.q2.apply(signs_filter)
# # Filter some words -- "Are, Does, Will, Why, ..."
# data['q1'] = data.q1.apply(common_filter_simple)
# data['q2'] = data.q2.apply(common_filter_simple)
data_input = data.copy()

# Create features
data['Na'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['NN', 'NNS', 'NNP'], is_norm=False, is_stem=True, penalty=0), axis=1)
data['Nr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['NN', 'NNS', 'NNP'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Wa'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['WDT', 'WP', 'WRB'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Wr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['WDT', 'WP', 'WRB'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Da'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['DT', 'CC', 'EX', 'IN'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Dr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['DT', 'CC', 'EX', 'IN'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ma'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['MD'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Mr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['MD'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ca'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['CD'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Cr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['CD'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Pa'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['PDT'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Pr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['PDT'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Va'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Vr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ra'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['RB', 'RBR', 'RBS', 'RP'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Rr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['RB', 'RBR', 'RBS', 'RP'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ja'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['JJ', 'JJR', 'JJS'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Jr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['JJ', 'JJR', 'JJS'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ca'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['NUM'], tagset='universal', is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Cr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['NUM'], tagset='universal', is_norm=True, is_stem=True, penalty=0), axis=1)
# data['nouns'] =  data.apply(lambda row: compare_pos_universal(row['q1'], row['q2'], ['NOUN']), axis=1)
# data['nouns_num'] =  data.apply(lambda row: compare_pos_universal(row['q1'], row['q2'], ['NOUN', 'NUM']), axis=1)
# data['verbs'] =  data.apply(lambda row: compare_pos_universal(row['q1'], row['q2'], ['VERB']), axis=1)
# data['VE'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['VERB'], tagset='universal', is_norm=True, is_stem=True, penalty=0), axis=1)
# data['AD'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['ADJ'], tagset='universal', is_norm=True, is_stem=True, penalty=0), axis=1)
# data['JJ'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['JJR', 'JJS'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['JJ'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['JJR', 'JJS'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ga'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'GPE', is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Gr'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'GPE', is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Pa'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'PERSON', is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Pr'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'PERSON', is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Oa'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'ORGANIZATION', is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Or'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'ORGANIZATION', is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ca'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['CD'], is_norm=False, is_stem=True, penalty=0), axis=1)
# data['Cr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['CD'], is_norm=True, is_stem=True, penalty=0), axis=1)
# data['Ya'] =  data.apply(lambda row: compare_years(row['q1'], row['q2'], is_norm=False, penalty=0), axis=1)
# data['Y'] =  data.apply(lambda row: compare_years(row['q1'], row['q2'], is_norm=True, penalty=3), axis=1)

# Drop unnecessary columns
data.drop(data.columns[range(5)], axis=1, inplace=True)

# Check the independence between the features
sns.heatmap(data.corr(), cmap='coolwarm')
plt.show()

# Split the data into training and test sets -- for now, without dev set
X = data.iloc[:,1:]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set results and creating confusion matrix
y_pred = classifier.predict(X_test)

# print([(val, idx) for idx, val in y_test.iteritems()])
# print(y_pred)

# Errors analysis
pd.set_option('display.max_colwidth', 68)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', None)

data_check = X_test.copy()
data_check['y'] = y_test
data_check['y_pred'] = y_pred
data_check[['q1', 'q2']] = data_input[['q1', 'q2']]
print(data_check[['q1', 'q2'] + list(data.columns.values[1:]) + ['y_pred', 'y']][data_check.y != data_check.y_pred])
# print(data_check[['q1', 'q2'] + list(data.columns.values[1:]) + ['y_pred', 'y']])

# print(X_test[X_test.index == y_test.index][y_test.values != y_pred])
# print(X_test)

# print(y_test.index[1])
# print(y_test.values)
# print(y_test.values[1])

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Accuracy
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

# Compute precision, recall, F-measure and support
print(classification_report(y_test, y_pred))

# Display the time of program execution
print('--- {:.2f} minutes ---'.format((time.time() - start_time)/60))

# # Create sound alarm when code finishes
# winsound.Beep(frequency=440, duration=500)
# winsound.Beep(frequency=440, duration=500)
# winsound.Beep(frequency=440, duration=500)
