import sys

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

from encoding import *

def _calc_probability(most, secode_most) :
    proba = (float(most) /(most + secode_most) * 100)
    return round(proba)


def _calc_ratios(text):

    ratios = {}

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    for lang in stopwords.fileids():
        stopwords_set = set(stopwords.words(lang))
        words_set = set(words)
        common_words = words_set.intersection(stopwords_set)

        ratios[lang] = len(common_words)

    return ratios


def detect_language(text):

    ratios = _calc_ratios(text)

    most_rated_language = max(ratios, key=ratios.get)
    most_common_words = ratios[most_rated_language]
    del ratios[most_rated_language]
    second_most_rated_language = max(ratios, key=ratios.get)
    second_most_common_words = ratios[second_most_rated_language]

    return most_rated_language


def isEnglish(path):
    enc = get_encoding(episode_path)
    f = open(episode_path, 'r',encoding=enc, errors='ignore')
    text = f.read()
    f.close()
    language = detect_language(text)
    return language == "english"