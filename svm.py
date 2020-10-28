########################################################
########################################################
######                                            ######
######      LEARNING FROM DATA, FINAL PROJECT     ######
######               (SVM CLASSIFIER)             ######
######                                            ######
########################################################
########################################################

# Importing libraries

import time
import re
import string
import sys
import json
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from nltk.corpus import stopwords
from nltk import word_tokenize

from constants import NEWSPAPER_ORIENTATION

stoplist = stopwords.words('english')
punctuations = string.punctuation + "’¶•@°©®™"


def preprocess_text(text):
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
        file_path = "COP" + str(COP) + ".filt3.sub.json"
        # if not surpress_print:
        print('Reading in articles from {0}...'.format(file_path))
        cop = json.load(open(file_path, 'r'))
        cop_edition = cop.pop('cop_edition', None)
        # Merging 6a data with rest of the data
        if COP == 6:
            cop_6a = json.load(open("COP6a.filt3.sub.json", 'r'))
            cop['collection_end'] = cop_6a['collection_end']
            cop['articles'] = cop['articles'] + cop_6a['articles']
        cop_data[int(cop_edition)] = cop

    print('Done!')
    return cop_data

# Retrieve training data (X = article_body, Y = political_orientation), for the default all files are read, otherwise a list of indices should be provided


def get_train_data(cop_selection=None):
    cop_data = read_data(cop_selection)
    trainX, trainY = list(), list()
    idx = 1

    for cop in cop_data.keys():
        for article in cop_data[cop]['articles']:
            article_body = article['headline'] + " " + article['body']
            political_orientation = NEWSPAPER_ORIENTATION[article['newspaper']]
            trainX.append(preprocess_text(article_body))
            trainY.append(political_orientation)
            print(idx, end="\r")
            idx += 1

    print("Processing data. This may take some time.")
    return trainX, trainY


X, Y = get_train_data()

# CHECK DISTRIBUTION OF LABELS:

print(Y.count('Left-Center'))
print(Y.count('Right-Center'))

split_point = int(0.80 * len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]


def identity(x):
    return x


vec = TfidfVectorizer(preprocessor=identity,
                      tokenizer=identity)

svc = SVC(kernel='rbf', C=10, gamma=1)

classifier = make_pipeline(vec, svc)

# Fitting our classifier onto our data
t0 = time.time()
classifier.fit(Xtrain, Ytrain)
train_time = time.time() - t0
print("training time: ", train_time)

# Testing new data
t0 = time.time()
Yguess = classifier.predict(Xtest)
test_time = time.time() - t0
print("testing time: ", test_time, '\n')

print(accuracy_score(Ytest, Yguess))
print(classification_report(Ytest, Yguess, zero_division=0))
print(confusion_matrix(Ytest, Yguess))

# Defining 5 - fold cross - validation
print("Performing 5-fold cross validation..")
scores = cross_val_score(classifier, Xtrain, Ytrain, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

########################################################
########################################################
####                                                ####
####                END OF SCRIPT                   ####
####                                                ####
########################################################
########################################################
