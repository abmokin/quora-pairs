import os
import csv
import numpy as np
import ast
import re
import itertools
from gensim.models import Word2Vec, KeyedVectors


def get_current_path():
    # Get path of the current directory
    return os.path.dirname(os.path.abspath(__file__))

def load_data(filename, capacity=-2):
    # Load the data from csv-file
    # Data source: https://www.kaggle.com/c/quora-question-pairs/data
    with open(get_current_path() + "/data/" + filename, encoding = "utf8")  as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list_1_questions = []
        list_2_questions = []
        list_classes = []
        counter = -2
        for row in readCSV:
            counter += 1
            if counter == -1:
                continue
            list_1_questions.append(row[3][:-1].lower().split())
            list_2_questions.append(row[4][:-1].lower().split())
            list_classes.append(row[5])
            if capacity == counter+1:
                break
    return list_1_questions, list_2_questions, list_classes

def merge_lists(list_1, list_2, is_sub=True):
    # Merge (pairwise) two lists into one, e.g., for [1,2,3] and [4,5,6] result: [1,4,2,5,3,6]
    # is_sub -- for filtering the words (letters, numbers, dashes and underscores)
    result_list_raw = list(zip(list_1, list_2))
    result_list = []
    for item in result_list_raw:
        flattened_list  = list(itertools.chain(*item))
        parsed_list = []
        for word_raw in flattened_list:            
            if is_sub:
                word = re.sub("[^a-zA-Z0-9_-]+", "", word_raw)
            parsed_list.append(word)
        result_list.append(parsed_list)
    return result_list

def get_w2v(list_questions, filename):
    # Get Google's Word2Vec embeddings for both questions to file
    # Data source: http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    with open(get_current_path() + "/" + filename + ".csv", mode="a", encoding="utf-8", newline='') as csvfile:
        with open(get_current_path() + "/" + filename  + "_unknown_words.txt", mode="a", encoding="utf-8") as un_file:
            writeCSV = csv.writer(csvfile, delimiter=',')
            for sentence in list_questions:
                for word in sentence:
                    if word in model.wv.vocab:
                        un_file.write('\n')
                        writeCSV.writerow(model.wv[word])
                    else:
                        un_file.write(word+'\n')
                        writeCSV.writerow([word])                    
                writeCSV.writerow(['***'])
                un_file.write('***\n')
            un_file.close()
        csvfile.close()

if __name__ == "__main__":
    # Load train.csv as lists; "capacity" -- number of pair of questions
    train_1st, train_2nd, train_class = load_data("train.csv", capacity=2)

    # Merge lists of the 1st and 2nd questions
    train_set_X = merge_lists(train_1st, train_2nd)

    # Load Google's Word2Vec model, pre-trained on GoogleNews
    model = KeyedVectors.load_word2vec_format(get_current_path() + '/data/GoogleNews-vectors-negative300.bin', binary=True)

    # Get Google's Word2Vec embeddings for both questions in the "train_set_X" and save them to file
    get_w2v(train_set_X, "train_set_X")
