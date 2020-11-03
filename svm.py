# !/usr/bin/env python3
"""
svm.py
This script performs the Support Vector Machines clasification 
on the binary task of political orientation of articles regarding COP meetings.
"""

# Importing libraries

from main import *

import time
import sys
import json
import re
import string
import argparse

import numpy as np

from sklearn.metrics import *
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import NEWSPAPER_ORIENTATION


def identity(x):
    return x


def main(train_data, test_data):

    Xtrain, Ytrain = train_data
    Xtest, Ytest = test_data

    print("Preprocessing happening")
    index = 1

    for doc in Xtrain:
        preprocess_text(doc)
        print(index, end="\r")
        index += 1

    for doc in Xtest:
        preprocess_text(doc)
        print(index, end="\r")
        index += 1

    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)

    svc = SVC(kernel='rbf', C=100, gamma=0.01)

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

    # Printing metrics
    print("Accuracy:\t", accuracy_score(Ytest, Yguess))
    print("F-scores:\t", f1_score(Ytest, Yguess, pos_label='Left-Center'))
    print("Precision:\t", precision_score(
        Ytest, Yguess, pos_label='Left-Center'))
    print("Recall:\t", recall_score(Ytest, Yguess, pos_label='Left-Center'))
    print(classification_report(Ytest, Yguess))
    print(confusion_matrix(Ytest, Yguess))

    # Defining 5 - fold cross - validation
    print("Performing 5-fold cross validation..")
    cv_classifier = make_pipeline(vec, svc)
    scores = cross_val_score(cv_classifier, Xtrain, Ytrain, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# If the script is run directly from the command line this is executed:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        train_data = get_train_data()
        test_data = get_test_data()
        print(
            f'TOTAL TRAINING INSTANCES = {len(train_data[0])},{len(train_data[1])}')
        print(
            f'TOTAL TESTING INSTANCES = {len(test_data[0])},{len(test_data[1])}\n')
        main(train_data, test_data)
    else:
        train_data = get_train_data(DOCS)
        print(
            f'TOTAL DATA INSTANCES = {len(train_data[0])},{len(train_data[1])}')
        validate_mlp(train_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     use_cross_validation=USE_CROSSVALIDATION)
