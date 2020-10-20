import json, numpy

from constants import NEWSPAPER_ORIENTATION
from svm import *
from mlp import *

# Read in COP file data, for the default all files are read, otherwise a list of indices should be provided
def read_data(cop_selection=None, surpress_print=False):
    if not cop_selection:
        cop_selection = list(range(1, 24 + 1))
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
        cop_data[int(cop_edition)] = cop

    print('Done!')
    return cop_data

# Retrieve training data (X = article_body, Y = political_orientation), for the default all files are read, otherwise a list of indices should be provided
def get_train_data(cop_selection=None):
    cop_data = read_data(cop_selection)
    trainX, trainY = list(), list()

    for cop in cop_data.keys():
        for article in cop_data[cop]['articles']:
            article_body = article['body']
            political_orientation = NEWSPAPER_ORIENTATION[article['newspaper']]
            trainX.append(article_body)
            trainY.append(political_orientation)

    return trainX, trainY


if __name__ == '__main__':
    # read_data(list(range(5,6)))
    data = get_train_data(list(range(1,4)))
    svm(data)
    mlp(data)
