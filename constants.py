# !/usr/bin/env python3
"""
constants.py

This script does not compute anything on its own, but rather defines several terms and constants the other scripts utilize.
"""

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

if __name__ == '__main__':
    print("nothing to see here.. Only some constants defined")
