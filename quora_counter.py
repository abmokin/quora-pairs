import os
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from scipy.stats import ttest_ind
import seaborn as sns
import difflib
from quora_utils import *
import time
import winsound
from termcolor import colored
from xgboost import XGBRegressor, XGBClassifier
import pickle
import itertools
from collections import Counter


def main():
    # Start timer
    start_time = time.time()

    # Load training set
    data = pd.read_csv(get_current_path() + '/data/train.csv', header=0, nrows=10000)   # for nrows
    # data = pd.read_csv(get_current_path()  + '/data/train.csv', header=0)             # for all rows
    data = data.dropna()

    # Rename columns in the dataframe
    data.columns = ['id', 'qid1', 'qid2', 'q1', 'q2', 'y']
    # data.rename(columns={'question1': 'q1', 'question2': 'q2', 'is_duplicate': 'y'}, inplace=True)

    # Convert all question into strings (for reliability)
    data['q1'] = data.q1.apply(str)
    data['q2'] = data.q2.apply(str)

    # Filter both questions -- filter \"
    # data['q1'] = data.q1.apply(signs_filter)
    # data['q2'] = data.q2.apply(signs_filter)
    data['q1'] = data.q1.apply(all_signs_filter)
    data['q2'] = data.q2.apply(all_signs_filter)
    # # Filter some words -- "Are, Does, Will, Why, ..."
    data['q1'] = data.q1.apply(common_filter_simple)
    data['q2'] = data.q2.apply(common_filter_simple)

    # # Count most frequent words
    # data['q12'] = data.q1 + " " + data.q2
    # print(data.head())
    # pd.set_option('display.max_rows', 100)
    # print(pd.Series(' '.join(data['q12']).lower().split()).value_counts()[:100])

    # Count specific words
    data['words'] = data.apply(lambda row: find_specific_words(row['q1'], row['q2'], row['y']), axis=1)
    feature_words_list = data['words'].tolist()
    feature_words = list(itertools.chain(*feature_words_list))
    print(feature_words[:500])
    print(len(feature_words))

    # Save all feature words
    pickle.dump(feature_words, open(get_current_path() + '/feature_words/feature_words_NN_JJ_CD_10000__1.dat', 'wb'))

    # Create list according to frequency of the words
    # feature_words = pickle.load(open(get_current_path() + '/feature_words/feature_words_NN_JJ_CD_1000.dat', 'rb'))
    feature_counter = Counter(feature_words)
    for word_frequency in range(1,2):
        feature_words_freq = [x[0] for x in feature_counter.items() if x[1] >= word_frequency]
        print('{}: {} -- {}'.format(word_frequency, len(feature_words_freq), feature_words_freq[:20]))

    # Display the time of program execution
    print('--- {:.2f} minutes ---'.format((time.time() - start_time)/60))

 if __name__ == "__main__":
    main()