# !/usr/bin/env python3
from keras.optimizers import SGD

from main import *
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from keras.models import Sequential
from keras.layers.core import Dense, Activation

np.random.seed(2018)  # for reproducibility and comparability, don't change!

# Filters the documents that do not adhere to the given minima
def filter_documents(data, minimum_words=100):
    print("Filtering documents with less than {0} words".format(minimum_words))
    filtered_documents = [(d,l) for d,l in data if len(d.split(" ")) >= minimum_words]
    removed_documents = len(data) - len(filtered_documents)
    print("Done; removed {0} documents".format(removed_documents))
    return filtered_documents

# Changes al endlines to spaces in a document
def convert_document(document):
    return ' '.join([sentence for sentence in document.split("\n") if len(sentence) > 2])

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(document, maximum_features=50):
    # Use the tfidf vectorizer
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=maximum_features)
    # countvectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=maximum_features)
    vectorized_document = tfidfvectorizer.fit_transform([document])
    vectorized_document = vectorized_document.data.tolist()
    # if max(vectorized_document) > 200:
    #     print("WE HAVE GREAT MAX:", max(vectorized_document))
    # vectorized_document = [(v/200) for v in vectorized_document]

    return vectorized_document

# Load noun-noun compound data
def load_data():
    data = get_train_data(list(range(20,23)))
    data = list(zip(data[0], data[1]))
    data = filter_documents(data, minimum_words=100)

    # BEGINNING
    # Some code for modifying the left-center - right-center balance of the data
    lefties = [(d,l) for d,l in data if l == "Left-Center"]
    righties = [(d,l) for d,l in data if l == "Right-Center"]
    number_docs = max(len(lefties), len(righties))
    print("BALANCE OF DATA:\n", "\tLEFT-CENTER={0}".format(len(lefties)), "\tRIGHT-CENTER={0}".format(len(righties)) )
    data = righties[:number_docs] + lefties[:number_docs]
    np.random.shuffle(data)
    # END

    X, Y = list(), list()
    print("Vectorizing data..")
    for document,label in data:
        vectorized_doc = vectorizer(convert_document(document), maximum_features=20)
        X.append(vectorized_doc)
        Y.append(label)
    print("Done!")

    classes = sorted(list(set(Y)))
    X = np.array(X)

    # Convert string labels to one-hot vectors
    # Y = label_binarize(Y, classes)
    print(classes)
    print(Y[1])
    Y = np.array([[1, 0] if label == classes[0] else [0, 1] for label in Y])

    # Split off development set from training data
    low = math.floor(0.8 * len(X))
    up = math.floor(0.95 * len(X))
    X_train, Y_train = (X[:low], Y[:low])  # 70%
    X_dev, Y_dev = (X[low:up], Y[low:up])  # 25%
    X_test, Y_test = (X[up:], Y[up:])  # 5%

    return X_train, X_dev, X_test, Y_train, Y_dev ,Y_test, classes

if __name__ == '__main__':
    BATCH_SIZE = 15
    EPOCHS = 100

    # Load data
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test, classes = load_data()

    print(type(X_train), X_train.shape, X_train[2])
    print(type(Y_train), Y_train.shape, Y_train[:10])

    print(len(X_train), 'training instances')
    print(len(X_dev), 'develoment instances')
    print(len(X_test), 'testing instances')
    print(X_train.shape[1], 'features')
    print(Y_train.shape[1], 'classes')

    # Build the model
    print("Building model...")
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=Y_train.shape[1]))
    model.add(Activation("linear"))
    sgd = SGD(lr=0.001)
    loss_function = 'mean_squared_error'
    model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, verbose=2, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)

    # Predict test set
    Y_guess = model.predict(X_dev)
    Y_dev = np.argmax(Y_dev, axis=1)
    Y_guess = np.argmax(Y_guess, axis=1)

    print("First 30 True labels:\t\t", Y_dev[:30])
    print("First 30 Predicted labels:\t", Y_guess[:30])
    print("No label 1 true:\t\t", len([l for l in Y_dev if l == 1]),"/",len(Y_dev))
    print("No label 1 predicted:\t", len([l for l in Y_guess if l == 1]),"/",len(Y_guess))
    print('Classification accuracy on development: {0}'.format(accuracy_score(Y_dev, Y_guess)))

    print("\nENF OF SCRIPT")
