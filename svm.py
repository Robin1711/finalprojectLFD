# !/usr/bin/env python3
"""
svm.py
This script performs the Support Vector Machines clasification 
on the binary task of political orientation of articles regarding COP meetings.
"""

# Importing libraries
from main import *

import time
import argparse

from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer


def identity(x):
    return x

def prepare_data(articles):
    # return [preprocess_text(article) for article in articles]
    prepared = list()
    for index, article in enumerate(articles):
        print(index, end="\r")
        preprocessed_article = preprocess_text(article)
        prepared.append(preprocessed_article)
    return prepared

def svm(Xtrain, Ytrain, Xtest, Ytest, use_cross_validating=False):
    print("Preprocessing..")
    Xtrain = prepare_data(Xtrain)
    Xtest = prepare_data(Xtest)
    print("Done!")

    print("Building model..")
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    svc = SVC(kernel='rbf', C=100, gamma=0.01)
    classifier = make_pipeline(vec, svc)
    print("Done!")

    # Fitting our classifier onto our training data
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

    if (use_cross_validating):
        # Defining 5 - fold cross - validation
        print("Performing 5-fold cross validation..")
        cv_classifier = make_pipeline(vec, svc)
        scores = cross_val_score(cv_classifier, Xtrain, Ytrain, cv=5)
        print("Done! Results:")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# If the script is run directly from the command line this is executed:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        Xtrain, Ytrain = get_train_data()
        Xtest, Ytest = get_test_data()
        print(
            f'TOTAL TRAINING INSTANCES = {len(Xtrain)},{len(Ytrain)}')
        print(
            f'TOTAL TESTING INSTANCES = {len(Xtest)},{len(Ytest)}\n')
        svm(Xtrain, Ytrain, Xtest, Ytest)
    else:
        X, Y = get_train_data([1,2,3,4])
        split_point = int(0.80 * len(X))
        Xtrain = X[:split_point]
        Ytrain = Y[:split_point]
        Xtest = X[split_point:]
        Ytest = Y[split_point:]
        print(
            f'TOTAL TRAINING INSTANCES = {len(Xtrain)},{len(Ytrain)}')
        print(
            f'TOTAL TESTING INSTANCES = {len(Xtest)},{len(Ytest)}\n')
        svm(Xtrain, Ytrain, Xtest, Ytest, use_cross_validating=True)
