import os
import re
import difflib

def get_current_path():
    # Get path of the current directory
    return os.path.dirname(os.path.abspath(__file__))

def difflib_ratio(sentence1, sentence2, is_lower=False):
    # Check rough ratio of sentences
    if is_lower:
        result = difflib.SequenceMatcher(a=sentence1.lower(), b=sentence2.lower()).ratio()
    else:
        result = difflib.SequenceMatcher(a=sentence1, b=sentence2).ratio()
    return result

def compare_numbers(sentence1, sentence2):
    # Select and compare all the numbers in the sentences combined together
    sen1 = re.sub("[^0-9\s]+", "", sentence1).split()
    sen2 = re.sub("[^0-9\s]+", "", sentence2).split()
    counter = 0
    if (len(sen1) > 0) and (len(sen1) > 0):
        for num in sen1:
            if num in sen2:
                counter += 1
        counter = counter * 2 / (len(sen1) + len(sen2))
    return counter

def compare_big_letters(sentence1, sentence2):
    # Compare all capital letters in the sentences (except the first letter)
    sen1 = re.sub("[^A-Z]+", "", sentence1)[1:]
    sen2 = re.sub("[^A-Z]+", "", sentence2)[1:]
    result = 0
    if (len(sen1) > 0) and (len(sen2) > 0):
        result = difflib.SequenceMatcher(a=sen1, b=sen2).ratio()
    return result

def signs_filter(sentence):
    # Filtering signs, e.g., "!", ".", etc.
    return re.sub("[\?\!\.\,\(\)\"\:\;\[\]\{\}]+", "", sentence)

def common_filter_simple(sentence):
    # Filtering common words
    sentence = sentence[:-1]  # delete last symbol ('?', '.', etc.)
    sentence = sentence.split()
    words_list = ["a", "about", "all", "and", "are", "as", "at", "back", "be", "because", "been",
                  "but", "can", "come", "could", "did", "do", "for",   # "can't", "didn't", "don't"
                  "from", "get", "go", "going", "good", "got", "had", "have", "he", "her", "here",
                  "he's", "hey", "him", "his", "how", "I", "if", "I'll", "I'm", "in", "is", "it",
                  "it's", "just", "know", "like", "look", "me", "mean", "my", "now",  # "not", "no"
                  "of", "oh", "OK", "okay", "on", "one", "or", "out", "really", "right", "say",
                  "see", "she", "so", "some", "something", "tell", "that", "that's", "the", "then",
                  "there", "they", "think", "this", "time", "to", "up", "want", "was", "we", "well",
                  "were", "what", "when", "who", "why", "will", "with", "would", "yeah", "yes",
                  "you", "your", "you're",  # https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists
                  "where", "why", "whose", "which", "shall", "should", "does", "has",  #"aren't",
                  "an", "whether", "by", "after", "am", "it", "its", "them",   # "haven't", "hasn't",
                  "their", "than",  "yours", "these", "those", "being", "while",  # "wouldn't", "hadn't"
                  "ask", "question", "questions"]
    for word in words_list:
        sentence[:] = (value for value in sentence if value.lower() != word.lower())
    return " ".join(sentence)

def is_word_in_sentence(word, sen):
    # Check whether the word is in the list, with .lower()
    result = 0
    for word_sen in sen:
        result = max(result, difflib.SequenceMatcher(a=word.lower(), b=word_sen.lower()).ratio())
    return result >= 0.8   # to catch words with misspellings

def terms_counter(sen1, sen2, similar_terms):
    # Special counter for find_terms function
    counter = 0
    for word in sen1:
        if word[0].isupper() and is_word_in_sentence(word, sen2) and (len(word) > 1) and (word != "I'm") and (word != "I'll"):
            similar_terms.append(word)
            counter += 1
    return counter, similar_terms

def other_terms_counter(sen1, sen2, similar_terms):
    # Count all terms in both sentences
    counter = 0
    for word in sen1:
        if word[0].isupper() and (len(word) > 1) and (word not in similar_terms) and (word != "I'm") and (word != "I'll"):
            counter += 1
    for word in sen2:
        if word[0].isupper() and (len(word) > 1) and (word not in similar_terms) and (word != "I'm") and (word != "I'll"):
            counter += 1
    return counter

def find_terms(sentence1, sentence2):
    # Find terms that begin with the uppercase letter, including acronyms
    result = 0
    sen1 = sentence1.split()
    sen2 = sentence2.split()
    # Make the first letter of the sentences lowercase
    if len(sen1) > 0:
        if len(sen1[0]) > 0:
            sen1[0] = sen1[0].lower()
    if len(sen2) > 0:
        if len(sen2[0]) > 0:
            sen2[0] = sen2[0].lower()
    # Normalized counter
    similar_terms = []
    counter1, similar_terms = terms_counter(sen1, sen2, similar_terms)
    counter2, similar_terms = terms_counter(sen2, sen1, similar_terms)
    counter = 2 * max(counter1, counter2)
    counter_others = other_terms_counter(sen1, sen2, similar_terms)
    if (counter + counter_others):
        result = counter/(counter + counter_others)
    return result

def is_acronym_in_sen(word, sen):
    # Check whether the acronym is in the list, with lower method
    result = 0
    for word_sen in sen:
        result = max(result, difflib.SequenceMatcher(a=word, b=word_sen).ratio())
    return result == 1.0

def acronyms_counter(sen1, sen2):
    # Special counter for the find_acronyms function
    counter = 0
    for word in sen1:
        if word.isupper() and is_acronym_in_sen(word, sen2) and (len(word) > 1):
            counter += 1
    return counter

def all_acronyms_counter(sen1, sen2):
    # Count all acronyms in both sentences
    counter = 0
    for word in sen1:
        if word.isupper() and (len(word) > 1):
            counter += 1
    for word in sen2:
        if word.isupper() and (len(word) > 1):
            counter += 1
    return counter

def find_acronyms(sentence1, sentence2):
    # Find acronyms in both sentences
    result = 0
    sen1 = sentence1.split()
    sen2 = sentence2.split()
    # Count acronyms
    counter1 = acronyms_counter(sen1, sen2)
    counter2 = acronyms_counter(sen2, sen1)
    counter_all = all_acronyms_counter(sen1, sen2)
    if counter_all:
        result = (counter1 + counter2)/counter_all
    return result

def symbol_in_both(sentence1, sentence2, symbol):
    # Is there some symbol in both sentences
    result = 0
    if symbol == "all":
        symbols = list("#$%&@*+^_|~")
        for sym in symbols:
            if (sentence1.find(sym)!=-1) and (sentence2.find(sym)!=-1):
                result += 1
        if (sentence1.find("\\")!=-1) and (sentence2.find("\\")!=-1):
            result += 1
        if (sentence1.find("++")!=-1) and (sentence2.find("++")!=-1):
            result += 1
        if ((sentence1.find("^")!=-1) and (sentence2.find("**")!=-1)) or ((sentence1.find("**")!=-1) and (sentence2.find("^")!=-1)):
            result += 1
        if ((sentence1.find("!")!=-1) and (sentence2.find("factorial")!=-1)) or ((sentence1.find("factorial")!=-1) and (sentence2.find("!")!=-1)):
            result += 1
        if ((sentence1.find("%")!=-1) and (sentence2.find("percent")!=-1)) or ((sentence1.find("percent")!=-1) and (sentence2.find("%")!=-1)):
            result += 1
        if ((sentence1.find("+")!=-1) and (sentence2.find("plus")!=-1)) or ((sentence1.find("plus")!=-1) and (sentence2.find("+")!=-1)):
            result += 1
    else:
        result = int((sentence1.find(symbol)!=-1) and (sentence2.find(symbol)!=-1))
    return result
