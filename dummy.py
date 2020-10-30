########################################################
########################################################
######                                            ######
######      LEARNING FROM DATA, FINAL PROJECT     ######
######             (DUMMY CLASSIFIER)             ######
######                                            ######
########################################################
########################################################


# IMPORTING LIBRARIES

from main import *

import time
import re
import string
import sys
import json
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords

# from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from constants import NEWSPAPER_ORIENTATION

# Read in COP file data, for the default all files are read, otherwise a list of indices should be provided

"""
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


def read_data(cop_selection=None, surpress_print=False):
    if not cop_selection:
        cop_selection = list(range(1, 25))
        print(cop_selection)
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
            print("Merging COP6a with COP6...")
        cop_data[int(cop_edition)] = cop
    print('Done!')
    return cop_data

# Retrieve training data (X = article_body, Y = political_orientation), for the default all files are read, otherwise a list of indices should be provided


def get_train_data(cop_selection=None):
    cop_data = read_data(cop_selection)
    trainX, trainY = list(), list()
    index = 1

    for cop in cop_data.keys():
        for article in cop_data[cop]['articles']:
            article_body = article['headline'] + " " + article['body']
            political_orientation = NEWSPAPER_ORIENTATION[article['newspaper']]
            trainX.append(preprocess_text(article_body))
            trainY.append(political_orientation)
            print(index, end="\r")
            index += 1

    return trainX, trainY
"""

X, Y = get_train_data()

"""
print("Preprocessing happening")
X = [preprocess_text(doc) for doc in X]
"""

split_point = int(0.80 * len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]


def identity(x):
    return x


vec = TfidfVectorizer(preprocessor=identity,
                      tokenizer=identity)

Xtrain_vec = vec.fit_transform(Xtrain)
Xtest_vec = vec.transform(Xtest)
Ytrain_vec = [1 if i.startswith('L') else 0 for i in Ytrain]
Ytest_vec = [1 if i.startswith('L') else 0 for i in Ytest]

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(Xtrain, Ytrain)
Yguess = dummy_clf.predict(Xtest)
Yguess = [np.argmax(a) for a in Yguess]

# Fitting our classifier onto our data
t0 = time.time()
dummy_clf.fit(Xtrain, Ytrain)
train_time = time.time() - t0
print("training time: ", train_time)

# Testing new data
t0 = time.time()
Yguess = dummy_clf.predict(Xtest)
test_time = time.time() - t0
print("testing time: ", test_time, '\n')

# Printing metrics
print("Accuracy:\t", accuracy_score(Ytest, Yguess))
#print("F-scores:\t", f1_score(Ytest, Yguess))
print("Precision:\t", precision_score(Ytest, Yguess))
print("Recall:\t", recall_score(Ytest, Yguess))
print(classification_report(Ytest, Yguess))
print(confusion_matrix(Ytest, Yguess))


# Defining 5 - fold cross - validation
print("Performing 5-fold cross validation..")
scores = cross_val_score(dummy_clf, Xtrain, Ytrain, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

########################################################
####                END OF SCRIPT                   ####
########################################################
