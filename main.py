# !/usr/bin/env python3

import json
import string
import re
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize

from constants import NEWSPAPER_ORIENTATION

# Filters the documents that do not adhere to the given minima
# Returns the new list of filtered documents
def filter_documents(documents, labels, minimum_words=100):
    data = list(zip(documents, labels))
    print("Filtering documents with less than {0} words".format(minimum_words))
    filtered_documents = [(d,l) for (d,l) in data if len(d.split(" ")) >= minimum_words]
    print(f"Done!; removed {len(documents) - len(filtered_documents)} documents\n")
    return list(zip(*filtered_documents))

# Balances the dataset 50/50 by removing examples from the more present labels
# Returns the new list of documents and labels
def balance_dataset(documents,labels):
    print("Balancing dataset 50/50")
    data = list(zip(documents,labels))
    np.random.shuffle(data)
    lefts = [(d,l) for d,l in data if l == "Left-Center"]
    rights = [(d,l) for d,l in data if l == "Right-Center"]
    number_docs = min(len(lefts), len(rights))
    data = lefts[:number_docs] + rights[:number_docs]
    np.random.shuffle(data)
    no_lefts = len([(d,l) for d,l in data if l == "Left-Center"])
    no_rights = len([(d,l) for d,l in data if l == "Right-Center"])
    print(f"Done! Balance of data: \t Left-Center={no_lefts}  :  Right-Center={no_rights}\n")
    return list(zip(*data))

# Preprocessing: Removes endlines, non_alpha characters and makes all characters lowercase
# Returns the new list of preprocessed documents
def preprocess_text(text):
    stoplist = stopwords.words('english')
    punctuations = string.punctuation + "’¶•@°©®™"
    txt = text.lower()
    txt = re.sub(r"[^a-zA-ZÀ-ÿ]", " ", txt)
    translator = str.maketrans(punctuations, " " * len(punctuations))
    s = txt.translate(translator)
    no_digits = ''.join([i for i in s if not i.isdigit()])
    cleaner = " ".join(no_digits.split())
    word_tokens = word_tokenize(cleaner)
    filtered_sentence = [w for w in word_tokens if not w in stoplist]
    return filtered_sentence

# Read in COP file data, for the default all files are read, otherwise a list of indices should be provided
def read_data(cop_selection=None, surpress_print=False):
    if not cop_selection:
        cop_selection = list(range(1, 24 + 1))
    cop_data = dict()
    for COP in cop_selection:
        file_path = "data/COP" + str(COP) + ".filt3.sub.json"
        if not surpress_print:
            print('Reading in articles from {0}...'.format(file_path))
        cop = json.load(open(file_path, 'r'))
        cop_edition = cop.pop('cop_edition', None)
        if COP == 6:
            cop_6a = json.load(open("data/COP6a.filt3.sub.json", 'r'))
            cop['collection_end'] = cop_6a['collection_end']
            cop['articles'] = cop['articles'] + cop_6a['articles']
        cop_data[int(cop_edition)] = cop

    print('Done!')
    return cop_data

# Retrieve training data (X = article_body, Y = political_orientation), for the default all
# files are read, otherwise a list of indices should be provided
def get_train_data(cop_selection=None):
    cop_data = read_data(cop_selection)
    trainX, trainY = list(), list()

    for cop in cop_data.keys():
        for article in cop_data[cop]['articles']:
            article_body = article['body']
            political_orientation = NEWSPAPER_ORIENTATION[article['newspaper']]
            trainX.append(article_body)
            trainY.append(political_orientation)

    return trainX, trainY

# Splits the data into a train and test set according to the given split
# Returns (X_train, Y_train, X_test, Y_test)
def split_data(X, Y, split=0.8):
    # Split off development set from training data
    bound = math.floor(split * len(X))
    X_train, Y_train = (X[:bound], Y[:bound])   # Default = 80%
    X_dev, Y_dev = (X[bound:], Y[bound:])       # Default = 100% - train% = 20%

    return X_train, X_dev, Y_train, Y_dev
