# !/usr/bin/env python3
from main import *

import argparse
import numpy as np
import math, nltk, json, time
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils, generic_utils
from sklearn.preprocessing import label_binarize

EPOCHS = 1
BATCH_SIZE = 10

SHOW_CONFUSION_MATRIX = False
USE_EMBEDDINGS = False
EMBEDDINGS_PATH = "embeddings/embeddings_5.json"

np.random.seed(1995)  # DON'T CHANGE; for reproducibility and comparability

# Build confusion matrix with matplotlib
def create_confusion_matrix(true, pred, classes):
    # Build matrix
    cm = confusion_matrix(true, pred, labels=classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Make plot
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.xlabel('Predicted label')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.show()

# Read in word embeddings
def read_embeddings(embeddings_file):
    print('Reading in embeddings from {0}...'.format(embeddings_file))
    embeddings = json.load(open(embeddings_file, 'r'))
    embeddings = {word: np.array(embeddings[word]) for word in embeddings}
    print('Done!')
    return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(X, embeddings):
    print("Vectorizing data..")
    vectorized_X = list()
    # # Take number characters and number words
    # characters = [len(body) for body in X]
    # words = [len(((body.replace("\n"," ")).replace("."," ")).split(" ")) for body in X]
    # vectorized_X = np.array([[c/max(characters) for c in characters], [w/max(words) for w in words]])
    # vectorized_X = np.transpose(vectorized_X)

    # # Use the tfidf vectorizer
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    vectorized_X = tfidfvectorizer.fit_transform(X)
    vectorized_X = vectorized_X.toarray()

    print("Done!")
    return vectorized_X

# Make from text body's numerical data (feature vectors)
def preprocess_data(X):
    print("Preprocessing data..")
    X_preprocessed = list()
    # for document in X:
    #     print(type(document), len(document))
    X_preprocessed = X
    print("Done!")
    return np.array(X_preprocessed)
    # TOP_WORDS = 2
    # arr = np.empty((0, TOP_WORDS))
    # for body in X:
    #     words = [w.lower() for w in nltk.word_tokenize(body) if w.isalpha() and len(w) > 4]
    #     counted_words = Counter(words)
    #     counted_words = sorted([(counted_words[key], key) for key in counted_words.keys()])
    #     vectorized_words = [vectorizer(word, embeddings) for amount,word in counted_words[TOP_WORDS:]]
    #     # arr = np.append(arr, np.array([[1, 2, 3]]), axis=0)
    #     # arr = np.append(arr, np.array([[4, 5, 6]]), axis=0)
    #     print(vectorized_words[0].shape)
    #     break

# Prepare data; Labels are binarized; X (the article) is made into a feature vector
# Divide the set in a train, development, and test set.
def prepare_data(X,Y):
    print("Preparing data...")
    # Binarize labels and get classes
    classes = sorted(list(set(Y)))
    # Y_binary = np.array(label_binarize(Y, classes))
    Y_binary = np.array([[1,0] if label == "Left-Center" else [0, 1] for label in Y])

    low = math.floor(0.7 * len(X))
    up = math.floor(0.9 * len(X))
    X_train, Y_train = (X[:low], Y_binary[:low])    # 70%
    X_dev, Y_dev = (X[low:up], Y_binary[low:up])    # 20%
    X_test, Y_test = (X[up:], Y_binary[up:])        # 10%
    Y_test = [0 if cs[0] == 1 else 1 for cs in Y_test]

    print("Done!")
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test, classes

# Build the model and train the model with the given train and development set
def build_model(X_train, X_dev, Y_train, Y_dev):
    print("\nBuilding model...\n")

    nb_documents = X_train.shape[0]
    print(nb_documents, 'training documents')
    nb_features = X_train.shape[1]
    print(nb_features, 'features')
    nb_classes = Y_train.shape[1]
    print(nb_classes, 'classes')

    # Build the model
    model = Sequential()
    # Single 200-neuron hidden layer with sigmoid activation
    model.add(Dense(input_dim=nb_features, output_dim=200, activation='relu'))
    # Output layer with softmax activation
    model.add(Dense(output_dim=nb_classes, activation='softmax'))
    # Specify optimizer, loss and validation metric
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    # Train the model
    history = model.fit(X_train, Y_train, nb_epoch=EPOCHS, batch_size=BATCH_SIZE
                        , validation_data=(X_dev, Y_dev), shuffle=True, verbose=1)
    print("Done!")
    return model

# From the given data prepare data, build model and produce accuracy
def mlp(data):
    X, Y = data

    # preprocess data
    X = preprocess_data(X)

    # vectorize data
    embeddings = read_embeddings(EMBEDDINGS_PATH) if USE_EMBEDDINGS else dict()
    X = vectorizer(X, embeddings)

    # split data into train, dev, test
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test, classes = prepare_data(X,Y)

    # build model
    model = build_model(X_train, X_dev, Y_train, Y_dev)

    # Predict labels for test set
    outputs = model.predict(X_test, batch_size=BATCH_SIZE)
    pred_classes = np.argmax(outputs, axis=1)
    print(pred_classes)
    print("Accuracy:", accuracy_score(Y_test, pred_classes))

    # Make confusion matrix on development data
    if SHOW_CONFUSION_MATRIX:
        Y_dev_names = [classes[x] for x in np.argmax(Y_dev, axis=1)]
        pred_dev = model.predict(X_dev, batch_size=BATCH_SIZE)
        pred_class_names = [classes[x] for x in np.argmax(pred_dev, axis=1)]
        create_confusion_matrix(Y_dev_names, pred_class_names, classes)

if __name__ == '__main__':
    # data = get_train_data([20,21,22,23,24])
    data = get_train_data([20])
    data = list(zip(data[0], data[1]))
    np.random.shuffle(data)
    samples = 200
    data = ([x for x,y in data][:samples], [y for x,y in data][:samples])
    time.sleep(1)
    mlp(data)
