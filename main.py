import json, numpy
from svm import *
from mlp import *

# Read in COP file data, for the default all files are read
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


if __name__ == '__main__':
    read_data(list(range(5,6)))
    data = {}
    svm(data)
    mlp(data)
