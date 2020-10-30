# !/usr/bin/env python3
from keras.optimizers import SGD

from main import *
import numpy as np
import math
import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Activation

np.random.seed(2018)  # for reproducibility and comparability, don't change!
BATCH_SIZE = 10
EPOCHS = 50
K_FOLDS = 8
USE_CROSSVALIDATION = False

# Preprocessing: Removes endlines, non_alpha characters and makes all characters lowercase
# Returns the new list of preprocessed documents
def preprocess_data(documents):
    print("Preprocessing data..")
    preprocessed_docs = list()
    for document in documents:
        doc = ' '.join([sentence for sentence in document.split("\n") if len(sentence) > 2])
        doc = ''.join([c for c in doc if c.isalpha() or c == ' '])
        preprocessed_docs.append(doc.lower())

    print("Done!\n")
    return preprocessed_docs

# Creates a feature vector using the document and the given embeddings
# Returns the feature vector for the given document
def use_embeddings(document, embeddings, features=10):
    words = document.split(' ')[:10]
    vectorized_doc = list()
    for word in words:
        try:
            v = embeddings[word.lower()]
        except KeyError:
            v = embeddings['UNK']
        vectorized_doc.append(v)
    # Vectorized_doc is now a list of Concat
    vectorized_doc = [inner for outer in vectorized_doc for inner in outer]
    return vectorized_doc

# Turns every document in the list into a feature vector
# Returns the new list of vectorized documents in dimension: (number of documents, number of features)
def vectorizer(documents, maximum_features=50, mode='embeddings'):
    print(f"Vectorizing data into {maximum_features} dimensions..")
    print(f"Dimensions = {np.array(documents).shape}")
    all_vectorized_docs = list()
    if mode == 'embeddings':
        embeddings = json.load(open('embeddings/embeddings_5.json', 'r'))
        for document in documents:
            vectorized_doc = use_embeddings(document, embeddings)
            all_vectorized_docs.append(vectorized_doc)
    else:  # != embedding':
        if mode == 'tfidf':
            method = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english', max_features=maximum_features)
        else: # == 'count':
            method = CountVectorizer(analyzer='word', stop_words='english', max_features=maximum_features)

        for document in documents:
            vectorized_doc = method.fit_transform([document])
            vectorized_doc = vectorized_doc.data.tolist()
            all_vectorized_docs.append(vectorized_doc)

    print(f"New dimensions = {np.array(all_vectorized_docs).shape}")
    print("Done!\n")
    return all_vectorized_docs

# Normalizes the given list of document vectors
# Returns the new list of normalized vectors
def normalize_vectors(documents):
    print("Normalizing Vectors..")
    normalized = list()
    maximum = max([max(document_vector) for document_vector in documents])
    for document_vector in documents:
        normalized.append([val/maximum for val in document_vector])

    print("Done!\n")
    return normalized

# Prepares data; filters, preprocesses and vectorizes the documents, and binarize the labels
# Returns (prepared documents, binarized labels, list of unique classes)
def prepare_data(documents,labels):
    # Filter short documents
    documents,labels = filter_documents(documents, labels, minimum_words=100)

    # Balance dataset 50/50
    documents, labels = balance_dataset(documents, labels)

    # Preprocess the documents
    documents = preprocess_data(documents)

    # Vectorize documents and binarize labels
    documents = vectorizer(documents, maximum_features=20)
    documents = normalize_vectors(documents)
    documents = np.array(documents)

    # Convert string labels to one-hot vectors
    classes = sorted(list(set(labels)))
    # Y = label_binarize(Y, classes)
    labels = np.array([[1, 0] if label == classes[0] else [0, 1] for label in labels])

    return documents, labels, classes

# Define model, build model, and train model.
# Returns (model, training history)
def run_model(X_train, Y_train, e=EPOCHS, bs=BATCH_SIZE):
    print("Building model...")
    nb_features = X_train.shape[1]
    nb_classes = Y_train.shape[1]
    print(f'{nb_features} features')
    print(f'{nb_classes} classes')

    # Define model
    model = Sequential()
    # Single hidden layer
    model.add(Dense(input_dim=nb_features, units=200))
    # Output layer with softmax activation
    model.add(Dense(units=nb_classes, activation='softmax'))
    # Specify optimizer, loss and validation metric
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    # # The perceptron:
    # model = Sequential()
    # model.add(Dense(input_dim=nb_features, units=nb_classes))
    # model.add(Activation("linear"))
    # sgd = SGD(lr=0.002)
    # loss_function = 'mean_squared_error'
    # model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])

    print("Training model...")
    # Train model
    history = model.fit(X_train, Y_train, verbose=1, epochs=e, batch_size=bs)
    print("Done!\n")

    return model, history

# Performs the training and validating of the accuracy
def mlp(data, epochs=EPOCHS, batch_size=BATCH_SIZE, use_cross_validation=True, kfolds=K_FOLDS):
    X, Y = data
    X, Y, classes = prepare_data(X,Y)
    X_train, X_test, Y_train, Y_test = split_data(X,Y)

    if use_cross_validation:
        ### Perform K-fold Cross Validation
        print('Peforming {0} Cross Validation...'.format(kfolds))
        acc_per_fold = []
        loss_per_fold = []
        # Merge inputs and targets
        inputs = np.concatenate((X_train, X_test), axis=0)
        targets = np.concatenate((Y_train, Y_test), axis=0)

        # Define folds
        fold_number = 1
        kf = KFold(n_splits=kfolds)
        for train_idx, test_idx in kf.split(inputs, targets):
            print('------------------------------------------------------------------------')
            print('Training for fold {0} ...'.format(fold_number))
            model, history = run_model(inputs[train_idx], targets[train_idx], epochs, batch_size)
            scores = model.evaluate(inputs[test_idx], targets[test_idx], verbose=0)

            print(f'Score for fold {fold_number}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_number = fold_number + 1
    else:
        ### Perform 'regular' training/testing by splitting the dataset
        print('Training and testing the model with:')
        print(len(X_train), len(Y_train), 'training instances')
        print(len(X_test), len(Y_test), 'testing instances')
        model, history = run_model(X_train, Y_train, epochs, batch_size)
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

        # Predict test set
        Y_test = np.argmax(Y_test, axis=1)
        Y_predicted = model.predict(X_test)
        Y_predicted = np.argmax(Y_predicted, axis=1)
        print("First 50 True labels:\t\t", Y_test[:50])
        print("First 50 Predicted labels:\t", Y_predicted[:50])
        print("Labels in True labels:\t\t 0: {0},\t 1: {1}\t total: {2}".format(len([l for l in Y_test if l == 0]), len([l for l in Y_test if l == 1]), len(Y_test)))
        print("Labels in Predicted labels:\t 0: {0},\t 1: {1}\t total: {2}".format(len([l for l in Y_predicted if l == 0]), len([l for l in Y_predicted if l == 1]), len(Y_predicted)))
        print('Classification accuracy on development: {0}'.format(accuracy_score(Y_test, Y_predicted)))

        print("\nFULL CLASSIFICATION REPORT")
        print(classification_report(Y_test, Y_predicted))

# If the script is run directly from the command line this is executed:
if __name__ == '__main__':
    data = get_train_data(list(range(20,22)))
    print(f'TOTAL DATA INSTANCES = {len(data[0])},{len(data[1])}')
    mlp(data, epochs=20, batch_size=20, use_cross_validation=False)

