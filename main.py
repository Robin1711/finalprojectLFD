import json, numpy
from svm import *
from mlp import *

# Read in COP file data, for the default all files are read
def read_data(first=1, last=24, include_6a=True):
    articles = dict()
    COP_files = list()
    for COP in range(first, last+1):
        file_path = "data/COP" + str(COP) + ".filt3.sub.json"
        print('Reading in articles from {0}...'.format(file_path))
        cop_article = json.load(open(file_path, 'r'))
        cop_edition = cop_article.pop('cop_edition', None)
        print("KEYS:", cop_article.keys)
        if COP == 6:
            cop_article_6a = json.load(open("data/COP6a.filt3.sub.json", 'r'))
        articles[int(cop_edition)] = cop_article

    print('Done!')
    return articles


if __name__ == '__main__':
    read_data(first=5, last=6)
    print("I read data!")
    data = {}
    svm(data)
    mlp(data)
