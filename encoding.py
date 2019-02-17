from bs4 import UnicodeDammit
import chardet


def get_encoding(path_fichier) :
    """
    renvoie l'encoding le plus probable du fichier
    """
    rawdata = open(path_fichier, 'rb').read()
    result = chardet.detect(rawdata)
    return result['encoding']


def normalize(text) :
    """
    renvoie le texte en entrée normalisé (é -> e, etc.)
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()