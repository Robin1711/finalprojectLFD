from main import *

NEWSPAPERS = [
    'The Australian' , 'Sydney Morning Herald (Australia)', 'The Age (Melbourne, Australia)'
    , 'The Times of India (TOI)', 'The Hindu'
    , 'The Times (South Africa)', 'Mail & Guardian'
    , 'The Washington Post', 'The New York Times'
    ]

NEWSPAPER_COUNTRY = {
    'The Australian':'Australian' , 'Sydney Morning Herald (Australia)':'Australian', 'The Age (Melbourne, Australia)':'Australian'
    , 'The Times of India (TOI)':'India', 'The Hindu':'India'
    , 'The Times (South Africa)':'South Africa', 'Mail & Guardian':'South Africa'
    , 'The Washington Post':'United States', 'The New York Times':'United States'
}

NEWSPAPER_ORIENTATION = {
    'The Australian':'Right-Center' , 'Sydney Morning Herald (Australia)':'Left-Center', 'The Age (Melbourne, Australia)':'Left-Center'
    , 'The Times of India (TOI)':'Right-Center', 'The Hindu':'Left-Center'
    , 'The Times (South Africa)':'Right-Center', 'Mail & Guardian':'Left-Center'
    , 'The Washington Post':'Left-Center', 'The New York Times':'Left-Center'
}

# given newspaper, returns orientation
def get_newspaper_orientation(newspaper):
    return NEWSPAPER_ORIENTATION[newspaper]


# given newspaper, returns country
def get_newspaper_country(newspaper):
    return NEWSPAPER_COUNTRY[newspaper]


# returns list of all newspapers
def get_newspaper_list():
    return NEWSPAPERS


def get_unique_classifications():
    cop_data = read_data()
    articles = list()
    for cop in cop_data.keys():
        articles = articles + cop_data[cop]['articles']

    uniques = {'subject':set(),'organization':set(),'industry':set(),'geographic':set()}
    for article in articles:
        subs = [s["name"] for s in article['classification']['subject']] if article['classification']['subject'] else None
        if subs: uniques['subject'].update(subs)
        orgs = [o["name"] for o in article['classification']['organization']] if article['classification']['organization'] else None
        if orgs: uniques['organization'].update(orgs)
        inds = [i["name"] for i in article['classification']['industry']] if article['classification']['industry'] else None
        if inds: uniques['industry'].update(inds)
        geos = [g["name"] for g in article['classification']['geographic']] if article['classification']['geographic'] else None
        if geos: uniques['geographic'].update(geos)

    return uniques


if __name__ == '__main__':
    print("nothing to see here.. Only some constants defined")