import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk import ne_chunk, pos_tag, word_tokenize
# from itertools import chain

# # Make feature values bigger, for convenience and faster gradient descent
# SCALE_FACTOR = 100

def get_current_path():
    # Get path of the current directory
    return os.path.dirname(os.path.abspath(__file__))

def signs_filter(sentence):
    # Filtering signs, e.g., "!", ".", etc.
    # return re.sub("[\?\!\.\,\(\)\"\:\;\[\]\{\}]+", "", sentence)
    sen = re.sub("[\"]+", "", sentence)
    # sen = sen[0].lower() + sen[1:] if len(sen) > 1 else sen  # low case first letter
    return sen

def common_filter_simple(sentence):
    # Filtering common words
    sentence = sentence.split()
    words_list = ["Can", "Can't", "Could", "Couldn't", "Did", "Do", "Does", "Didn't", "Don't", "Doesn't",
                  "How", "What", "When", "Who", "Why", "Will", "Would", "Where", "Whom", "Whose", "Which",
                  "Shall", "Should", "Are", "Aren't", "Whether", "Wouldn't", "ask", "If", "While", "Is", "Isn't",
                  "Have", "Has", "Had", "There", "Haven't", "Hasn't", "Hadn't", "Was", "Were", "After",
                  "Before", "As soon as", "By", "one"]
    for word in words_list:
        sentence[:] = (value for value in sentence if value != word)
    return " ".join(sentence)

def select_pos(sentence, pos_list, tagset, is_stem):
    # Select all non-GPE nouns from the sentence
    sen = word_tokenize(sentence)
    sen_tagged = pos_tag(sen, tagset=tagset)
    sen_pos = []
    for pos in pos_list:
        sen_pos = sen_pos + [word.lower() for (word, tag) in sen_tagged if tag == pos]
    # Stem GPE-nouns
    if is_stem:
        # stemmer = PorterStemmer()
        stemmer = SnowballStemmer("english")
        sen_pos = [stemmer.stem(word) for word in sen_pos]
    return sen_pos

def count_counter(sen1, sen2, sentence1, sentence2, pos_list, is_stem, penalty):
    counter = 0

    # Tokenize entire sentences
    sen1_all = word_tokenize(sentence1.lower())
    sen2_all = word_tokenize(sentence2.lower())

    # Stem entire sentences
    if is_stem:
        stemmer = SnowballStemmer("english")
        sen1_all = [stemmer.stem(word) for word in sen1_all]
        sen2_all = [stemmer.stem(word) for word in sen2_all]

    # Count common non-GPE words from 1st related to 2nd sentence, with the penalty for different words
    for word in sen1:
        counter = counter + 1 if word in sen2_all else counter - penalty
    # Count common non-GPE words from 2nd related to 1st sentence, with the penalty for different words
    for word in sen2:
        counter = counter + 1 if word in sen1_all else counter - penalty
    return counter

def compare_pos(sentence1, sentence2, pos_list=['NN', 'NNS'], tagset=None, is_norm=False, is_stem=False, penalty=1):
    # Select and compare all non-GPE nouns in both sentences; for target='universal'  -->  pos_list=['NOUN', 'NUM']
    sen1 = select_pos(sentence1, pos_list, tagset, is_stem)
    sen2 = select_pos(sentence2, pos_list, tagset, is_stem)

    # Count common non-GPE tags with the penalty for mismatch
    counter = count_counter(sen1, sen2, sentence1, sentence2, pos_list, is_stem, penalty)

    # Normalize if is_stem=True
    if is_norm and len(sen1 + sen2) > 0:
        counter /= len(sen1 + sen2)
    return counter

def count_years(sen1, sen2, sentence1, sentence2, penalty):
    # Count years in the sentences
    counter = 0

    # Tokenize entire sentences
    sen1_all = word_tokenize(sentence1.lower())
    sen2_all = word_tokenize(sentence2.lower())

    # Count years from 1st related to 2nd sentence, with the penalty for different years
    for year in sen1:
        counter = counter + 1 if year in sen2_all else counter - penalty
    # Count years from 2nd related to 1st sentence, with the penalty for different years
    for year in sen2:
        counter = counter + 1 if year in sen1_all else counter - penalty
    return counter

def select_years(sentence):
    # Select all years from the sentence
    sen_tagged = pos_tag(word_tokenize(sentence))
    sen_pos = [word for (word, tag) in sen_tagged if tag == 'CD']
    sen_years = []
    for word in sen_pos:
        num_word = re.sub("\D", "", word)
        if len(num_word) == 4:            
            if int(num_word) > 1900 and int(num_word) < 2100:
                sen_years.append(num_word)
    return sen_years

def compare_years(sentence1, sentence2, is_norm=False, penalty=1):
    # Select and compare all non-GPE nouns in both sentences; for target='universal'  -->  pos_list=['NOUN', 'NUM']
    sen1 = select_years(sentence1)
    sen2 = select_years(sentence2)

    # Count common non-GPE tags with the penalty for mismatch
    counter = count_years(sen1, sen2, sentence1, sentence2, penalty)

    # Normalize if is_stem=True
    if is_norm and len(sen1 + sen2) > 0:
        counter /= len(sen1 + sen2)
    return counter


def find_labeled_words(sen_chunked, ne_label, is_stem):
    # Find all GPE ne-labeled words in the sentence
    sen_word_list = []
    for chunk in sen_chunked:
        if hasattr(chunk, 'label'):
            if chunk.label() == ne_label:
            # if chunk[0][1] == 'NNP' and chunk.label() == ne_label:
                sen_word_list.append([ch[0] for ch in chunk])
    if is_stem:
        # stemmer = PorterStemmer()
        stemmer = SnowballStemmer("english")
        sen_word_list_stemmed = []
        for word_list in sen_word_list:
            word_list_stemmed = [stemmer.stem(word) for word in word_list]
            sen_word_list_stemmed.append(word_list_stemmed)
        sen_word_list = sen_word_list_stemmed
    return sen_word_list

def count_ne_counter(sen1, sen2, sentence1, sentence2, is_stem, penalty):
    counter = 0

    # Tokenize entire sentences
    sen1_one_list = word_tokenize(sentence1.lower())
    sen2_one_list = word_tokenize(sentence2.lower())

    # Stem entire sentences
    if is_stem:
        stemmer = SnowballStemmer("english")
        sen1_one_list = [stemmer.stem(word) for word in sen1_one_list]
        sen2_one_list = [stemmer.stem(word) for word in sen2_one_list]

    # Count common GPE-labeled words from 1st related to 2nd sentence, with the penalty for different words
    for sen1_words_list in sen1:
        is_in_sen2 = False
        for sen1_word in sen1_words_list:
            if sen1_word.lower() in sen2_one_list:
                counter += 1
                is_in_sen2 = True
                break
        if not is_in_sen2:
            counter -= penalty

    # Count common GPE-labeled words from 2nd related to 1st sentence, with the penalty for different words
    for sen2_words_list in sen2:
        is_in_sen1 = False
        for sen2_word in sen2_words_list:
            if sen2_word.lower() in sen1_one_list:
                counter += 1
                is_in_sen1 = True
                break
        if not is_in_sen1:
            counter -= penalty
    return counter

def ne_counter(sentence1, sentence2, ne_label='GPE', is_norm=False, is_stem=False, penalty=1):
    # Count GPE-labeled tags, with negative-valued penalty for different GPEs in the sentences
    sen1_chunked = ne_chunk(pos_tag(word_tokenize(sentence1)))
    sen2_chunked = ne_chunk(pos_tag(word_tokenize(sentence2)))
    # print(sen1_chunked, sen2_chunked)

    # Get list of GPE-labeled words
    sen1 = find_labeled_words(sen1_chunked, ne_label, is_stem)
    sen2 = find_labeled_words(sen2_chunked, ne_label, is_stem)
    # print(sen1, sen2)

    # Count common GPE-labeled words with the penalty for mismatch
    counter = count_ne_counter(sen1, sen2, sentence1, sentence2, is_stem, penalty)

    # Normalize if is_stem=True
    if is_norm and len(sen1 + sen2) > 0:
        counter /= len(sen1 + sen2)
    return counter
