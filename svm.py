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
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import *
from sklearn.model_selection import cross_val_score

from constants import NEWSPAPER_ORIENTATION

def identity(x):
    return x

def main():
    X, Y = get_train_data(cop_selection=None)

    print("Preprocessing happening")
    X = [preprocess_text(doc) for doc in X]

    split_point = int(0.80 * len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]


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
    print("Precision:\t", precision_score(Ytest, Yguess, pos_label='Left-Center'))
    print("Recall:\t", recall_score(Ytest, Yguess, pos_label='Left-Center'))
    print(classification_report(Ytest, Yguess))
    print(confusion_matrix(Ytest, Yguess))

    # Defining 5 - fold cross - validation
    print("Performing 5-fold cross validation..")
    cv_classifier = make_pipeline(vec, svc)
    scores = cross_val_score(cv_classifier, X, Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == "__main__":
    main()
