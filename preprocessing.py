#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import nltk
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def verifier_ligne(ligne):
    """return True si la ligne est un sous-titre, False sinon"""
    timestamp_regex = r'[0-9]{2}:[0-9]{2}:[0-9]{2}' 
    subnumber_regex =r'^[0-9]+$'
    
    liste_regex = [timestamp_regex, subnumber_regex]

    l = ligne.lower()
    if "addic7ed" in l:
        return False
    #if l.startswith("sync"):
        #return False
    for regex in liste_regex:
        if re.match(regex, ligne):
            return False
    return True

def transformer_ligne(ligne):
    """str -> str
    effectue transformations souhaitees sur la ligne"""
    tag_regex = r'<(/)*[a-zA-Z]+>' #to get rif of tags
    alphanum_regex = r'\W+'  #get rid of non alphanumeric characters
    new_line = re.sub(tag_regex, '', ligne)
    new_line = re.sub(alphanum_regex, ' ', new_line)
    return new_line

def scan_folder(parent_folder, corp):
    """retourne corpus des textes contenus dans parent_folder sous forme de liste de string"""
    # iterate over all the files in directory 'parent_folder'
    for file_name in sorted(os.listdir(parent_folder)):
        if file_name.endswith(".txt"):
            path = parent_folder+"/"+file_name
            fichier = open(path, "r", encoding="utf-8")
            lignes = fichier.readlines()
            fichier.close()
            
            texte = ""
            for ligne in lignes :
                #ligne = ligne.lower()
                if verifier_ligne(ligne):
                    new_line = transformer_ligne(ligne)
                    texte += new_line
            corp.append(texte)
        
        else:
            current_path = "".join((parent_folder, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, corp)
    
    return corp


#pour eviter les variables globales utiliser get_corpus qui appelle scan_folder
def get_corpus(parent_folder):
    """retourne corpus des textes contenus dans parent_folder sous forme de liste de string"""
    c = []
    res = scan_folder(parent_folder, c)
    return res

def stemming_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

def lemmatizing_tokenizer(str_input):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(str_input)
    words = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    return words

def lemmatizing_tokenizer_v2(str_input):
    words = []
    wnl = WordNetLemmatizer()
    tokens_tagged =pos_tag(word_tokenize(str_input))
    for word, tag in tokens_tagged:
        if tag.startswith("NN"):
            word = wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            word = wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            word = wnl.lemmatize(word, pos='a')
        else:
            pass
        words.append(word)

    return words


def nbSeriesCorpus(path):
    return sum(os.path.isdir(os.path.join(path, i)) for i in sorted(os.listdir(path)))


def getShowDictList(path):
    liste_dico_series = []
    if nbSeriesCorpus(path) != 1:
        filenames= sorted(os.listdir(path)) # get all files' and folders' names in the current directory
        for filename in filenames: # loop through all the files and folders
            if os.path.isdir(os.path.join(os.path.abspath(path), filename)): # check whether the current object is a folder or not
                show_path = path+"/"+filename
                liste = []
                nb_saisons = sum(os.path.isdir(os.path.join(show_path, i)) for i in sorted(os.listdir(show_path)))
                for i in range(1, nb_saisons+1):
                    if i < 10:
                        path_saison = show_path+"/0"+str(i)
                    else:
                        path_saison = show_path+"/"+str(i)
                    nb_eps_saison = len(fnmatch.filter(os.listdir(path_saison), '*.txt'))
                    liste.append(nb_eps_saison)
                l = np.cumsum(liste)
                seasons_list = list(range(1, nb_saisons+1))
                dico_serie = dict(zip(seasons_list, l))  # key : season id value : max index for this season (maxInd+1 actually) in dataframe
                liste_dico_series.append(dico_serie)
    else:
        liste = []
        nb_saisons = sum(os.path.isdir(os.path.join(path, i)) for i in sorted(os.listdir(path)))
        for i in range(1, nb_saisons+1):
            if i < 10:
                path_saison = path+"/0"+str(i)
            else:
                path_saison = path+"/"+str(i)
            nb_eps_saison = len(fnmatch.filter(os.listdir(path_saison), '*.txt'))
            liste.append(nb_eps_saison)
        l = np.cumsum(liste)
        seasons_list = list(range(1, nb_saisons+1))
        dico_serie = dict(zip(seasons_list, l))
        liste_dico_series.append(dico_serie)

    return liste_dico_series