import sys, time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# CLASSIFIER_MODE can be set to 'NB' | 'DT' | 'KNN'
CLASSIFIER_MODE = "NB"
K = 11
# FSCORE_MODE can be set to 'micro' | 'macro' | 'weighted'
FSCORE_MODE = "weighted"
MULTICLASS = True

# Read corpus of sentences into list of lists.
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels

# a dummy function that just returns its input
def identity(x):
    return x

# a function that contains all setings for the classifier
def setup_classifier(mode="KNN", nearest_neighbours=K):
    # let's use the TF-IDF vectorizer
    tfidf = True

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity,
                              tokenizer = identity)
    else:
        vec = CountVectorizer(preprocessor = identity,
                              tokenizer = identity)

    # combine the vectorizer with a classifier base on constant MODE
    if mode == "NB":
        # print("SPECIFIED CLASSIFIER = MULTINOMIAL NAYIVE BAYES")
        return Pipeline([('vec', vec), ('cls', MultinomialNB(fit_prior=False))])
    elif mode == "KNN":
        # print("SPECIFIED CLASSIFIER = K NEAREST NEIGHBORS")
        return Pipeline([('vec', vec), ('cls', KNeighborsClassifier(n_neighbors=nearest_neighbours, weights='distance'))])
    elif mode == "DT":
        # print("SPECIFIED CLASSIFIER = DECISION TREE")
        # CHANGE PARAMETERS:
        ### min_samples_split = 2
        ### min_samples_leaf = 1
        ### max_depth = None
        ### max_leaf_nodes = None
        return Pipeline([('vec', vec), ('cls', DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=4, max_leaf_nodes=250))])
    else:
        exit("No valid classifier mode is provided")

# Preprocess given data (list of list of words), and return the modified data
def preprocess_data(X):
    stemmer = SnowballStemmer("english")
    X_new = list()
    for sentence in X:
        X_new.append([stemmer.stem(word) for word in sentence if word.isalpha()])

    return X_new

# Recieve train and test data, and specified classifier
# Return accuracy, avg precision, avg recall, avg f-score, training time and time prediction test labels
def train_test(Xtrain, Ytrain, Xtest, Ytest, mode=CLASSIFIER_MODE, nearest_neighbours=K):
    # Specify classifier
    classifier = setup_classifier(mode, nearest_neighbours)
    Xtrain = preprocess_data(Xtrain)
    Xtest = preprocess_data(Xtest)

    # Time the training process
    t0 = time.time()
    classifier.fit(Xtrain, Ytrain)
    train_time = time.time() - t0

    # Time the testing process
    t1 = time.time()
    Yguess = classifier.predict(Xtest)
    prediction_time = time.time() - t1

    # print_confusion_matrix(classifier, Ytest, Yguess)

    accuracy = accuracy_score(Ytest, Yguess)
    prf = precision_recall_fscore_support(Ytest, Yguess, average=FSCORE_MODE)
    return accuracy, prf[0], prf[1], prf[2], train_time, prediction_time


def print_results(accuracy, precision, recall, fscore, train_time, prediction_time):
    print("\nPERFORMACNE RESULTS:\n")
    print("Training time \t\t\t\t= {:.3f}s".format(train_time))
    print("Testing/Prediction time \t= {:.3f}s".format(prediction_time))
    print("\nEFFECTIVENESS RESULTS:\n")
    print("The recall, precision and f-score are averaged values over all classes, using FSCORE =", FSCORE_MODE)
    print("ACCURACY: \t {:.3f}".format(accuracy))
    print("PRECISION: \t {:.3f}".format(precision))
    print("RECALL: \t {:.3f}".format(recall))
    print("FSCORE: \t {:.3f}".format(fscore))


def print_confusion_matrix(classifier, Ytest, Yguess):
    print("\nCONFUSION MATRIX:")
    print('\t\t\t', classifier.classes_)
    for idx,r in enumerate(confusion_matrix(Ytest, Yguess)):
        print(classifier.classes_[idx], "\t\t", r)

########################################################################
## Start script
########################################################################


if __name__ == "__main__":
    # Check correct number of arguments
    if len(sys.argv) < 3:
        exit("The first two arguments have to be specified.")

    trainset_path = sys.argv[1]
    testset_path = sys.argv[2]

    # Read corpus(es)   if MULTICLASS is TRUE, the topic tags will be used
    #                   if MULTICLASS if False, the sentiment tags will be used.
    Xtrain, Ytrain = read_corpus(trainset_path, use_sentiment=not(MULTICLASS))
    Xtest, Ytest = read_corpus(testset_path, use_sentiment=not(MULTICLASS))

    # No test set defined, so we will divide the train set; 75% train and 25% test
    if (len(Xtest) == 0 ):
        split_point = int(0.90 * len(Xtrain))
        Xtest = Xtrain[split_point:]
        Ytest = Ytrain[split_point:]
        Xtrain = Xtrain[:split_point]
        Ytrain = Ytrain[:split_point]


    accuracy, precision, recall, fscore, train_time, prediction_time = train_test(Xtrain, Ytrain, Xtest, Ytest, mode=CLASSIFIER_MODE)

    ########################################################################

    print_results(accuracy, precision, recall, fscore, train_time, prediction_time)
