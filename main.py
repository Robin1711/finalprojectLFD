import json, numpy

# Read in COP file data, for the default all files are read
def read_data(first=1, last=24, include_6a=True):
    articles = dict()
    COP_files = list()
    for COP in range(first, last+1):
        file_path = "data/COP" + str(COP) + ".filt3.sub.json"
        print('Reading in articles from {0}...'.format(file_path))
        cop_article = json.load(open(file_path, 'r'))
        cop_edition = cop_article.pop('cop_edition', None)
        articles[int(cop_edition)] = cop_article

    if first <= 6 and last >= 6:
        file_path = "data/COP6a.filt3.sub.json"
        print('Reading in articles from {0}...'.format(file_path))
        cop_article = json.load(open(file_path, 'r'))
        cop_edition = cop_article.pop('cop_edition', None)
        articles[cop_edition] = cop_article

    print('Done!')
    return articles


if __name__ == '__main__':
    data_keys = list(range(1,25))
    cop_keys = ['collection_start', 'collection_end', 'articles']
    article_keys = ['path', 'raw_text', 'newspaper', 'date', 'headline', 'body', 'classification']
    classification_keys = ['subject', 'organization', 'industry', 'geographic']

    d = read_data()
    first_article = d[1]['articles'][0]
    for key in first_article.keys():
        print(key, " : ", type(first_article[key]))

    print(first_article['classification'].keys())
