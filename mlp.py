# !/usr/bin/env python3
from main import *

import argparse
import numpy as np
import math

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils, generic_utils
from sklearn.preprocessing import label_binarize

EPOCHS = 10
BATCH_SIZE = 10
CONFUSION_MATRIX = False

np.random.seed(1995)  # DON'T CHANGE; for reproducibility and comparability


# Make from text body's numerical data (number characters, number words)
def preprocess_data(X,Y):
    classes = sorted(list(set(Y)))
    characters = [len(body) for body in X]
    words = [len(((body.replace("\n"," ")).replace("."," ")).split(" ")) for body in X]
    X_numpy = np.array([characters, words])
    # print("MEAN =", np.mean(X_numpy[0], axis=0), ":", X_numpy[0][:15])
    # print("STD =", np.std(X_numpy[0], axis=0))
    # print("MEAN =", np.mean(X_numpy[1], axis=0), ":", X_numpy[1][:15])
    # print("STD =", np.std(X_numpy[1], axis=0))
    X_numpy = np.transpose(X_numpy)
    Y_binary = label_binarize(Y, classes)

    return X_numpy, Y_binary, classes


def prepare_data(X,Y):
    print("Preparing data...")
    # Prepare data
    X_nump, Y_binary, classes = preprocess_data(X,Y)
    low = math.floor(0.7 * len(X))
    up = math.floor(0.9 * len(X))
    X_train = X_nump[:low]    # 70%
    Y_train = Y_binary[:low]  # 70%
    X_dev = X_nump[low:up]    # 20%
    Y_dev = Y_binary[low:up]  # 20%
    X_test = X_nump[up:]      # 10%
    Y_test = Y_binary[up:]    # 10%

    print("Done!")
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test, classes


def build_model(X_dev, X_train, Y_dev, Y_train):
    print("\nBuilding model...\n")

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


# From the given data prepare data, build model and produce accuracy
def mlp(data):
    X, Y = data
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test, classes = prepare_data(X,Y)

    model = build_model(X_dev, X_train, Y_dev, Y_train)

    # Predict labels for test set
    outputs = model.predict(X_test, batch_size=BATCH_SIZE)
    pred_classes = np.argmax(outputs, axis=1)
    print("Accuracy:", accuracy_score(Y_test, pred_classes))

    # Make confusion matrix on development data
    if CONFUSION_MATRIX:
        Y_dev_names = [classes[x] for x in np.argmax(Y_dev, axis=1)]
        pred_dev = model.predict(X_dev, batch_size=BATCH_SIZE)
        pred_class_names = [classes[x] for x in np.argmax(pred_dev, axis=1)]
        create_confusion_matrix(Y_dev_names, pred_class_names, classes)


if __name__ == '__main__':
    print("Load data...")

    data = get_train_data([20,21,22,23,24])
    data = list(zip(data[0],data[1]))
    np.random.shuffle(data)
    data = ([x for x,y in data],[y for x,y in data])
    mlp(data)