from main import *
from constants import *


def basics(cop):
    stats = dict()
    stats['date_from'] = cop['collection_start']
    stats['date_to'] = cop['collection_end']
    return stats


def counts(articles):
    counts = dict()
    counts['#total_articles'] = len(articles)
    counts['newspapers'] = dict([(np,0) for np in get_newspaper_list()])
    counts['labels'] = {'Right-Center':0, 'Left-Center':0}
    counts['country'] = {'Australian':0, 'India':0, 'South Africa':0, 'United States':0}

    for article in articles:
        counts['newspapers'][article['newspaper']] += 1
        counts['labels'][get_newspaper_orientation(article['newspaper'])] += 1
        counts['country'][get_newspaper_country(article['newspaper'])] += 1

    return counts


def get_stats_per_cop(first=1, last=24):
    cops = read_data(first=first, last=last)
    # cops = read_data()
    statistics = dict()
    for cop in cops:
        statistics[cop] = dict()
        statistics[cop]['basics'] = dict()
        statistics[cop]['basics'] = basics(cops[cop])
        statistics[cop]['counts'] = dict()
        statistics[cop]['counts'] = counts(cops[cop]['articles'])

    return statistics


def get_stats(cop_selection=None):
    cops = read_data()
    articles = list()

    if not cop_selection:
        cop_selection = cops.keys()
    for cop in cop_selection:
        articles = articles + cops[cop]['articles']

    stats = dict()
    stats['counts'] = counts(articles)
    return stats


if __name__ == '__main__':
    print(get_stats_per_cop(first=5, last=6))
    # uniques = get_unique_classifications()
    # print(len(uniques['subject']))
    # print(len(uniques['organization']))
    # print(len(uniques['subject']))
    # print(len(uniques['subject']))
