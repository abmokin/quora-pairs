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

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('universal_tagset')

# Decomposition constants
FILE_NUM = 7
N_ROWS = 100000
SKIP_ROWS = (FILE_NUM - 1) * N_ROWS 


def main():
    # Start timer
    start_time = time.time()

    # Load training set
    data = pd.read_csv(get_current_path() + '/data/test.csv', header=0, skiprows=SKIP_ROWS, nrows=N_ROWS)   # for nrows
    # data = pd.read_csv(get_current_path()  + '/data/test.csv', header=0)             # for all rows
    # data = data.dropna()

    # Rename columns in the dataframe
    # data.columns = ['id', 'qid1', 'qid2', 'q1', 'q2']
    data.columns = ['test_id', 'q1', 'q2']
    # data.rename(columns={'question1': 'q1', 'question2': 'q2', 'is_duplicate': 'y'}, inplace=True)

    # Convert all question into strings (for reliability)
    data['q1'] = data.q1.apply(str)
    data['q2'] = data.q2.apply(str)    

    # Filter both questions -- filter \"
    data['q1'] = data.q1.apply(signs_filter)
    data['q2'] = data.q2.apply(signs_filter)
    # # Filter some words -- "Are, Does, Will, Why, ..."
    # data['q1'] = data.q1.apply(common_filter_simple)
    # data['q2'] = data.q2.apply(common_filter_simple)

    # data_input = data.copy()

    # Create features
    data['NNa'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['NN', 'NNS', 'NNP'], is_norm=False), axis=1)
    data['NNr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['NN', 'NNS', 'NNP'], is_norm=True), axis=1)

    data['BGa'] =  data.apply(lambda row: compare_bigrams(row['q1'], row['q2'], is_norm=False), axis=1)
    data['BGr'] =  data.apply(lambda row: compare_bigrams(row['q1'], row['q2'], is_norm=True), axis=1)

    # data['CDa'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['CD'], is_norm=False), axis=1)
    data['CDr'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['CD'], is_norm=True), axis=1)

    # data['JRS'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['JJR', 'JJS'], is_norm=True), axis=1)
    data['best'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['best'], is_all=True), axis=1)

    data['first'] =  data.apply(lambda row: compare_first(row['q1'], row['q2']), axis=1)
    data['last'] =  data.apply(lambda row: compare_last(row['q1'], row['q2']), axis=1)

    data['year'] =  data.apply(lambda row: compare_years(row['q1'], row['q2'], is_all=True), axis=1)
    data['negs'] =  data.apply(lambda row: compare_negations(row['q1'], row['q2']), axis=1)

    data['GPE'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'GPE'), axis=1)
    data['PER'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'PERSON'), axis=1)
    data['ORG'] =  data.apply(lambda row: ne_counter(row['q1'], row['q2'], 'ORGANIZATION'), axis=1)

    data['verb'] =  data.apply(lambda row: compare_verbs(row['q1'], row['q2'], is_norm=True), axis=1)

    data['india'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['india'], is_all=True), axis=1)
    data['modi'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['indian', 'modi'], is_all=False), axis=1)
    data['movie'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['movie'], is_all=True), axis=1)

    data['quora'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['quora'], is_all=True), axis=1)
    data['facebook'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['facebook'], is_all=True), axis=1)
    data['google'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['google'], is_all=True), axis=1)
    data['trump'] =  data.apply(lambda row: compare_specific_words(row['q1'], row['q2'], ['donald', 'trump'], is_all=False), axis=1)

    data['100500'] =  data.apply(lambda row: compare_specific_numbers(row['q1'], row['q2'], ['500', '1000'], is_all=True), axis=1)

    # data['WN1'] =  data.apply(lambda row: words_number(row['q1']), axis=1)
    # data['WN2'] =  data.apply(lambda row: words_number(row['q2']), axis=1)
    # data['JJ'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['JJ'], is_norm=True), axis=1)
    # data['verb'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['VERB'], tagset='universal', is_norm=True), axis=1)
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
    # data['AD'] =  data.apply(lambda row: compare_pos(row['q1'], row['q2'], ['ADJ'], tagset='universal', is_norm=True, is_stem=True, penalty=0), axis=1)


    # Drop unnecessary columns
    # data.drop(data.columns[range(5)], axis=1, inplace=True)

    # # Check the independence between the features
    # sns.heatmap(data.corr(), cmap='coolwarm')
    # plt.show()

    # Split the data into training and test sets -- for now, without dev set
    X_test = data.iloc[:,3:]
    # print(X_test)
    # y = data.iloc[:,0]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Use XGBoost
    # xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.1)
    # xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)

    # Save and load model
    # pickle.dump(xgb_model, open(get_current_path() + '/xgb_model.dat', 'wb'))
    xgb_model = pickle.load(open(get_current_path() + '/models/001__model_07_07_2018.dat', 'rb'))

    # Prediction
    data['is_duplicate'] = xgb_model.predict(X_test)
    data.drop(data.columns[range(1,25)], axis=1, inplace=True)
    
    data.to_csv(get_current_path() + '/pred/pred_' + str(FILE_NUM) + '.csv', index=False, header=False)

    # # Errors analysis
    # pd.set_option('display.max_colwidth', 50)
    # pd.set_option('display.max_columns', 50)
    # pd.set_option('display.width', None)

    # data_check = X_test.copy()
    # data_check['y'] = y_test
    # data_check['y_pred'] = y_pred
    # data_check[['q1', 'q2']] = data_input[['q1', 'q2']]
    # print(data_check[['q1', 'q2'] + list(data.columns.values[1:]) + ['y_pred', 'y']][data_check.y != data_check.y_pred])
    # # print(data_check[['q1', 'q2'] + list(data.columns.values[1:]) + ['y_pred', 'y']][data_check.y != data_check.y_pred][data_check.india != 0])

    # # Obtain confusion matrix
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)

    # # Calculate the precision, recall, accuracy and F1 score on the test set
    # tp = conf_matrix[0,0]
    # tn = conf_matrix[1,1]
    # fp = conf_matrix[0,1]
    # fn = conf_matrix[1,0]
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1_score_on_test_set = 2 / (1/precision + 1/recall)
    # accuracy_on_test_set = (tp + tn)/(tp + tn + fp + fn)
    # model_log_loss = log_loss(y_test, y_pred)

    # print('TP + TN: {}'.format(tp + tn))
    # print('Accuracy on test set: {:.2f}'.format(accuracy_on_test_set))
    # print('F1 score on test set: {:.2f}'.format(f1_score_on_test_set))
    # print('Log loss: {:.2f}'.format(model_log_loss))

    # # Compute precision, recall, F-measure and support
    # print(classification_report(y_test, y_pred))

    # Display the time of program execution
    print('--- {:.2f} minutes ---'.format((time.time() - start_time)/60))

    # Create sound alarm when code finishes
    winsound.Beep(frequency=440, duration=500)
    winsound.Beep(frequency=440, duration=500)
    winsound.Beep(frequency=440, duration=500)


if __name__ == "__main__":
    main()