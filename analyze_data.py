from main import *
from constants import *


def basics(cop):
    stats = dict()
    stats['date_from'] = cop['collection_start']
    stats['date_to'] = cop['collection_end']
    return stats


def counts(articles):
    counts = dict()
    counts['total_articles'] = len(articles)
    counts['newspapers'] = dict([(np,0) for np in get_newspaper_list()])
    counts['labels'] = {'Right-Center':0, 'Left-Center':0}
    counts['country'] = {'Australian':0, 'India':0, 'South Africa':0, 'United States':0}

    for article in articles:
        counts['newspapers'][article['newspaper']] += 1
        counts['labels'][get_newspaper_orientation(article['newspaper'])] += 1
        counts['country'][get_newspaper_country(article['newspaper'])] += 1

    return counts


def get_stats_per_cop(cop_selection=None):
    cop_data = read_data(cop_selection)
    statistics = dict()
    for cop in cop_data:
        statistics[cop] = dict()
        statistics[cop]['basics'] = dict()
        statistics[cop]['basics'] = basics(cop_data[cop])
        statistics[cop]['counts'] = dict()
        statistics[cop]['counts'] = counts(cop_data[cop]['articles'])

    return statistics


def get_stats(cop_selection=None):
    cop_data = read_data(cop_selection)
    articles = list()

    if not cop_selection:
        cop_selection = cop_data.keys()
    for cop in cop_selection:
        articles = articles + cop_data[cop]['articles']

    stats = dict()
    stats['counts'] = counts(articles)
    return stats


if __name__ == '__main__':
    print(get_stats_per_cop(list(range(5,6))))
    # uniques = get_unique_classifications()
    # print(len(uniques['subject']))
    # print(len(uniques['organization']))
    # print(len(uniques['subject']))
    # print(len(uniques['subject']))
