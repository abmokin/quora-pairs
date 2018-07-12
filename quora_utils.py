import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, pos_tag, word_tokenize, bigrams, trigrams
# from itertools import chain


# # Make feature values bigger, for convenience and faster gradient descent
# SCALE_FACTOR = 100
DEFAULT_PENALTY = 0
DEFAULT_PENALTY_SPECIFIC = 0

def get_current_path():
    # Get path of the current directory
    return os.path.dirname(os.path.abspath(__file__))

def signs_filter(sentence, replace_this="[\"]+", by_this=""):
    # Filtering signs, e.g., "!", ".", etc.
    # return re.sub("[\?\!\.\,\(\)\"\:\;\[\]\{\}]+", "", sentence)
    sentence = re.sub("[\"]+", "", sentence)
    sentence = re.sub("[/]+", " ", sentence)
    # sen = sen[0].lower() + sen[1:] if len(sen) > 1 else sen  # low case first letter
    return sentence

def all_signs_filter(sentence):
    # Filtering signs, e.g., "!", ".", etc.
    sentence = re.sub("[/]+", " ", sentence)    
    sentence = re.sub("[\?\!\.\,\(\)\"\:\;\[\]\{\}’]+", "", sentence)
    return sentence


def filter_for_bigrams(sentence):
    # Filtering common words
    sentence = sentence.split()
    words_list = ["Can", "Can't", "Could", "Couldn't", "Did", "Do", "Does", "Didn't", "Don't", "Doesn't",
                  "How", "What", "When", "Who", "Why", "Will", "Would", "Where", "Whom", "Whose", "Which",
                  "Shall", "Should", "Are", "Aren't", "Whether", "Wouldn't", "If", "While", "Is", "Isn't",
                  "Have", "Has", "Had", "There", "Haven't", "Hasn't", "Hadn't", "Was", "Were", "After",
                  "Before", "As soon as", "By", "one", "a", "an", "the", "of", "to", "I'll", "I'm", "I",
                  "at", "in", "on", "me", "my"]
    for word in words_list:
        sentence[:] = (value for value in sentence if value != word)
    return " ".join(sentence)


def common_filter_simple(sentence):
    # Filtering common words
    # sentence = sentence[:-1]  # delete last symbol ('?', '.', etc.)
    sentence = sentence.split()
    # words_list = ["a", "about", "all", "and", "are", "as", "at", "back", "be", "because", "been",
    #               "but", "can", "come", "could", "did", "do", "for",  "can't", "didn't", "don't"
    #               "from", "get", "go", "going", "good", "got", "had", "have", "he", "her", "here",
    #               "he's", "hey", "him", "his", "how", "I", "if", "I'll", "I'm", "in", "is", "it",
    #               "it's", "just", "know", "like", "look", "me", "mean", "my", "now", "not", "no"
    #               "of", "oh", "OK", "okay", "on", "one", "or", "out", "really", "right", "say",
    #               "see", "she", "so", "some", "something", "tell", "that", "that's", "the", "then",
    #               "there", "they", "think", "this", "time", "to", "up", "want", "was", "we", "well",
    #               "were", "what", "when", "who", "why", "will", "with", "would", "yeah", "yes",
    #               "you", "your", "you're",  # https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists
    #               "where", "why", "whose", "which", "shall", "should", "does", "has",  "aren't",
    #               "an", "whether", "by", "after", "am", "it", "its", "them", "haven't", "hasn't",
    #               "their", "than",  "yours", "these", "those", "being", "while", "wouldn't", "hadn't",
    #               "of", "from"]
    words_list = ["can", "come", "could", "did", "do", "can't", "didn't", "don't", "had", "have", "how", "I'll", "I'm",
                  "so", "that", "that's", "there", "was", "were", "what", "when", "who", "why", "will", "would", 
                  "where", "why", "whose", "which", "shall", "should", "does", "has",  "aren't", "whether", 
                  "haven't", "hasn't", "these", "those", "being", "while", "wouldn't", "hadn't", "what's"]                  
    for word in words_list:
        sentence[:] = (value for value in sentence if value.lower() != word.lower())
    return " ".join(sentence)


# --- POS MAIN ---

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

def count_counter(sen1, sen2, sentence1, sentence2, pos_list, is_norm, is_stem, penalty):
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

    # # Count common words
    # sen1 = list(set(sen1))
    # sen2 = list(set(sen2))
    # counter = 2* len(list(set(sen1) & set(sen2)))

    # Normalize if is_norm=True
    if is_norm and len(sen1 + sen2) > 0:
        counter /= len(sen1 + sen2)

    return counter

def compare_pos(sentence1, sentence2, pos_list=['NN', 'NNS'], tagset=None, is_norm=False, is_stem=True, penalty=DEFAULT_PENALTY):
    # # Filter sentences
    # sentence1 = signs_filter(sentence1, "[/]+", " ")
    # sentence2 = signs_filter(sentence2, "[/]+", " ")

    # Select and compare all non-GPE nouns in both sentences; for target='universal'  -->  pos_list=['NOUN', 'NUM']
    sen1 = select_pos(sentence1, pos_list, tagset, is_stem)
    sen2 = select_pos(sentence2, pos_list, tagset, is_stem)

    # # Remove duplicates
    # sen1 = list(set(sen1))
    # sen2 = list(set(sen2))

    # Count common non-GPE tags with the penalty for mismatch
    counter = count_counter(sen1, sen2, sentence1, sentence2, pos_list, is_norm, is_stem, penalty)

    return counter


# --- YEARS ---

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

def compare_years(sentence1, sentence2, is_norm=False, is_all=False, penalty=0, result_base=1):
    # Select and compare all non-GPE nouns in both sentences; for target='universal'  -->  pos_list=['NOUN', 'NUM']
    sen1 = select_years(sentence1)
    sen2 = select_years(sentence2)

    # # Count common years tags with the penalty for mismatch
    # counter = count_years(sen1, sen2, sentence1, sentence2, penalty)

    # # Normalize if is_norm=True
    # if is_norm and len(sen1 + sen2) > 0:
    #     counter /= len(sen1 + sen2)

    # return 1 if len(sen_inter) == len(sen1_uniq) and len(sen1_uniq) == len(sen2_uniq) and len(sen1_uniq) > 0 else 0

    return sen_checker(sen1, sen2, is_all, penalty, result_base)


# --- NE COUNTER ---

def find_labeled_words(sen_chunked, ne_label, is_stem):
    # Find all GPE ne-labeled words in the sentence
    sen_word_list = []
    for chunk in sen_chunked:
        if hasattr(chunk, 'label'):
            # if chunk.label() == ne_label:
            if chunk[0][1] == 'NNP' and chunk.label() == ne_label:
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

def ne_counter(sentence1, sentence2, ne_label='GPE', is_norm=False, is_stem=True, penalty=DEFAULT_PENALTY):
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

    # Normalize if is_norm=True
    if is_norm and len(sen1 + sen2) > 0:
        counter /= len(sen1 + sen2)


    result = 0

    if counter > 0:
        result = 1
    elif counter == 0 and (len(sen1) > 0 or len(sen2) > 0):
        result = -1

    return result


# --- BIGRAMS ---

def create_bigrams(sentence):
    sen = word_tokenize(sentence)
    stemmer = SnowballStemmer("english")
    sen = [stemmer.stem(word) for word in sen]
    return list(bigrams(sen))

def compare_bigrams(sentence1, sentence2, is_norm=False):
    # Select and compare all bigrams in both sentences

    # Filter first words, i.e "How, What, Who, ..."
    sentence1 = filter_for_bigrams(sentence1)
    sentence2 = filter_for_bigrams(sentence2)

    # Filter signs
    sentence1 = re.sub("[\?\!\.\,\(\)\"\'\$\%\@\#\&\^\:\;\[\]\{\}]+", "", sentence1)
    sentence2 = re.sub("[\?\!\.\,\(\)\"\'\$\%\@\#\&\^\:\;\[\]\{\}]+", "", sentence2)

    # Prepair bigrams
    sen1 = create_bigrams(sentence1)
    sen2 = create_bigrams(sentence2)
    # sen1_reversed = [bigram[::-1] for bigram in sen1]

    # Count common bigrams
    counter = len(list(set(sen1) & set(sen2)))
    # counter += len(list(set(sen1_reversed) & set(sen2)))

    # Normalize if is_norm=True
    if is_norm and len(sen1 + sen2) > 0:
        counter /= len(sen1 + sen2)
    return counter


# --- LAST WORD ---

def select_last_word(sentence):
    # Select all non-GPE nouns from the sentence
    sen = word_tokenize(sentence)
    sen_last = []
    word_prev = ""
    for word in sen:
        if word == "?" and word_prev != "":
            sen_last.append(word_prev)
        word_prev = word
    # Stem GPE-nouns
    stemmer = SnowballStemmer("english")
    sen_last = [stemmer.stem(word) for word in sen_last]
    return sen_last

def compare_last(sentence1, sentence2):
    # Select and compare last words in both sentences

    # Filter signs except "?"
    sentence1 = re.sub("[\!\.\,\(\)\"\'\$\%\@\#\&\^\:\;\[\]\{\}]+", "", sentence1)
    sentence2 = re.sub("[\!\.\,\(\)\"\'\$\%\@\#\&\^\:\;\[\]\{\}]+", "", sentence2)

    # Select last words
    sen1 = select_last_word(sentence1)
    sen2 = select_last_word(sentence2)

    return len(list(set(sen1) & set(sen2)))


# --- FIRST WORD ---

def select_first_word(sentence):
    # Select all non-GPE nouns from the sentence
    sen = word_tokenize(sentence)
    stemmer = SnowballStemmer("english")
    sen = [stemmer.stem(word) for word in sen]
    return sen[0] if len(sen) > 0 else ""

def compare_first(sentence1, sentence2):
    # Select and compare first words in both sentences

    # Select first words
    sen1 = select_first_word(sentence1)
    sen2 = select_first_word(sentence2)

    return int(sen1 == sen2 and len(sen1) > 1 and len(sen2) > 1)


# --- NEGATIONS ---

def select_negations(sentence):
    # Select all negations from the sentence
    sen = word_tokenize(sentence)
    sen_neg = []
    neg_list = ["n't", "not", "no", "nothing", "noone", "nobody", "never"]
    for word in sen:
        if word in neg_list:
            sen_neg.append(word)
    return sen_neg

def compare_negations(sentence1, sentence2):
    # Select and compare negations in both sentences
    result = 0

    # Select negations
    sen1 = select_negations(sentence1)
    sen2 = select_negations(sentence2)

    if len(sen1) > 0 and len(sen2) == 0 or len(sen2) > 0 and len(sen1) == 0:
        result = 1

    return result


# --- VERBS ---

def select_verbs(sentence):
    # Select all verbs from the sentence
    sen = word_tokenize(sentence)
    sen_tagged = pos_tag(sen, tagset='universal')
    sen_verbs = [word.lower() for (word, tag) in sen_tagged if tag == 'VERB']

    # # Lemmatize verbs
    # wordnet_lemmatizer = WordNetLemmatizer()
    # sen_verbs = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in sen_verbs]

    # Stem entire sentences
    stemmer = SnowballStemmer("english")
    sen_verbs = [stemmer.stem(word) for word in sen_verbs]

    return sen_verbs

def count_counter_verbs(sen1, sen2, sentence1, sentence2, is_norm):
    counter = 0

    # Tokenize entire sentences
    sen1_all = word_tokenize(sentence1.lower())
    sen2_all = word_tokenize(sentence2.lower())

    # # Lemmatize sentences
    # wordnet_lemmatizer = WordNetLemmatizer()
    # sen1_all = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in sen1_all]
    # sen2_all = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in sen2_all]

    # Stem entire sentences
    stemmer = SnowballStemmer("english")
    sen1_all = [stemmer.stem(word) for word in sen1_all]
    sen2_all = [stemmer.stem(word) for word in sen2_all]

    # Count common non-GPE words from 1st related to 2nd sentence, with the penalty for different words
    for word in sen1:
        counter = counter + 1 if word in sen2_all else counter
    # Count common non-GPE words from 2nd related to 1st sentence, with the penalty for different words
    for word in sen2:
        counter = counter + 1 if word in sen1_all else counter

    # Normalize if is_norm=True
    if is_norm and len(sen1 + sen2) > 0:
        counter /= len(sen1 + sen2)

    return counter

def compare_verbs(sentence1, sentence2, is_norm=False):
    # Select and compare all verbs in both sentences
    sen1 = select_verbs(sentence1)
    sen2 = select_verbs(sentence2)

    # Count common verbs
    # sen1 = list(set(sen1))
    # if 'be' in sen1:
    #     sen1.remove('be')
    # sen2 = list(set(sen2))
    # if 'be' in sen2:
    #     sen2.remove('be')
    # counter = 2* len(list(set(sen1) & set(sen2)))
    counter = count_counter_verbs(sen1, sen2, sentence1, sentence2, is_norm)

    # # Normalize if is_norm=True
    # if is_norm and len(sen1 + sen2) > 0:
    #     counter /= len(sen1 + sen2)
    return counter


# --- WORDS NUMBER ---

def words_number(sentence):
    # Count word number
    return len(word_tokenize(sentence)) / 50


# --- SPECIFIC WORDS ---

def select_words(sentence, words_list):
    # Select all soecific words from the sentence
    sen = word_tokenize(sentence)
    sen = [word.lower() for word in sen]
    sen_words = [word for word in words_list if word.lower() in sen]
    return sen_words

def sen_checker(sen1, sen2, is_all, penalty, result_base):
    # Remove duplicates and compare sen lists
    sen1_uniq = list(set(sen1))
    sen2_uniq = list(set(sen2))

    sen_inter = list(set(sen1_uniq) & set(sen2_uniq))

    result = 0

    if is_all:
        if len(sen_inter) == len(sen1_uniq) and len(sen1_uniq) == len(sen2_uniq) and len(sen1_uniq) > 0:
            result = result_base
        elif len(sen1_uniq) > 0 or len(sen2_uniq) > 0:
            result = penalty
    else:
        if len(sen_inter) > 0:
            result = result_base
        elif len(sen_inter) == 0 and (len(sen1_uniq) > 0 or len(sen2_uniq) > 0):
            result = penalty
    return result

def compare_specific_words(sentence1, sentence2, words_list, is_all = True, penalty=DEFAULT_PENALTY_SPECIFIC, result_base=1):
    # Select and compare all specific words in both sentences
    sen1 = select_words(sentence1, words_list)
    sen2 = select_words(sentence2, words_list)

    return sen_checker(sen1, sen2, is_all, penalty, result_base)


# --- SPECIFIC RNUMBERS ---

def select_numbers(sentence, numbers_list):
    # Select all numbers from the sentence
    sen_tagged = pos_tag(word_tokenize(sentence))
    sen_pos = [word for (word, tag) in sen_tagged if tag == 'CD']
    sen_numbers = []
    for num in sen_pos:
        num = re.sub("\D", "", num)
        if num in numbers_list:
            sen_numbers.append(num)
    return sen_numbers

def compare_specific_numbers(sentence1, sentence2, numbers_list, is_all=False, penalty=DEFAULT_PENALTY_SPECIFIC, result_base=1):
    # Select and compare all specific numbers in both sentences
    sen1 = select_numbers(sentence1, numbers_list)
    sen2 = select_numbers(sentence2, numbers_list)

    return sen_checker(sen1, sen2, is_all, penalty, result_base)


def find_specific_words(sentence1, sentence2, label):
    # Select and compare all specific numbers in both sentences
    pos_list = ['NOUN','ADJ','NUM']
    tagset = 'universal'
    is_stem = True
    sen1 = select_pos(sentence1, pos_list, tagset, is_stem)
    sen2 = select_pos(sentence2, pos_list, tagset, is_stem)

    if label:
        result = list(set(sen1) & set(sen2))
        # result = []
    else:
        # result = list(set(sen1).symmetric_difference(set(sen2)))
        result = []

    return result
