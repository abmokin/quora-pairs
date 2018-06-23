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
from quora_log_utils import *
import time
import winsound

# Start timer
start_time = time.time()

# Load training set
data = pd.read_csv(get_current_path() + '/data/train.csv', header=0, nrows=10000)   # for nrows
# data = pd.read_csv(get_current_path() + '/data/train.csv', header=0)             # for all rows
data = data.dropna()

# Rename columns in the dataframe
data.rename(columns={'question1': 'q1', 'question2': 'q2', 'is_duplicate': 'y'}, inplace=True)
# data.columns = ['id', 'qid1', 'qid2', 'q1', 'q2', 'y']
# print(data.shape)
# print(list(data.columns))

# # Barplot for the dependent variable
# sns.countplot(x='y',data=data, palette='hls')
# plt.show()

# Filter both questions
# Filter "?!.,()":;[]{}""
data['q1'] = data.q1.apply(signs_filter)
data['q2'] = data.q2.apply(signs_filter)
# # Filter the most frequently used words -- "a, of, the, ..."
# data['q1'] = data.q1.apply(common_filter_simple)
# data['q2'] = data.q2.apply(common_filter_simple)
# # Check result after filtering
# print(data[0:10][['y', 'q1', 'q2']][data.y == 1])

# Create features
# data['delta_length'] = data.apply(lambda row: abs(len(row['q1']) - len(row['q2'])), axis=1)
# data['delta_words'] = data.apply(lambda row: abs(len(row['q1'].split()) - len(row['q2'].split())), axis=1)
data['rough_ratio'] =  data.apply(lambda row: difflib_ratio(row['q1'], row['q2'], is_lower=True), axis=1)
data['numbers'] =  data.apply(lambda row: compare_numbers(row['q1'], row['q2']), axis=1)
# data['big_letters'] =  data.apply(lambda row: compare_big_letters(row['q1'], row['q2']), axis=1)
data['terms'] = data.apply(lambda row: find_terms(row['q1'], row['q2']), axis=1)
# data['acr'] = data.apply(lambda row: find_acronyms(row['q1'], row['q2']), axis=1)
# data['symbols'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "all"), axis=1)
# data['hash'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "#"), axis=1)
# data['percent'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "%"), axis=1)
# data['plus'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "+"), axis=1)
# data['caret'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "^"), axis=1)
# data['amp'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "&"), axis=1)
# data['semicolon'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], ";"), axis=1)
# data['star'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "*"), axis=1)
# data['dollar'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "$"), axis=1)
# data['at_sign'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "@"), axis=1)
# data['backlash'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "\\"), axis=1)
# data['underscore'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "_"), axis=1)
# data['tilde'] =  data.apply(lambda row: symbol_in_both(row['q1'], row['q2'], "~"), axis=1)
data['terms_and_numbers'] =  data.terms * data.numbers
# data['amp_semicolon'] =  data.amp * data.semicolon
# data['numbers_caret'] =  data.numbers * data.caret
# data['numbers_star'] =  data.numbers * data.star
# data['numbers_symbol'] =  data.numbers * data.symbol

# Create dummy variables
# data = pd.get_dummies(data, columns =[''])

# Check data
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', None)
print(data[0:500][['y', 'q1', 'q2', 'numbers']][data.numbers > 0])

# Drop unnecessary columns
data.drop(data.columns[range(5)], axis=1, inplace=True)

# # Check the independence between the features
# sns.heatmap(data.corr(), cmap='coolwarm')
# plt.show()

# Split the data into training and test sets -- for now, without dev set
X = data.iloc[:,1:]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Check out training data is sufficient
# print(X_train.shape)

# Fit logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set results and creating confusion matrix
y_pred = classifier.predict(X_test)
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
