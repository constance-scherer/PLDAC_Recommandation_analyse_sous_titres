#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:32:47 2019

@author: constancescherer
"""

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.feature_extraction import stop_words
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def verifier_ligne(ligne):
    """
    return True si la ligne est un sous-titre, False sinon
    """
    timestamp_regex = r'[0-9]{2}:[0-9]{2}:[0-9]{2}' 
    subnumber_regex =r'^[0-9]+$'
    liste_regex = [timestamp_regex, subnumber_regex]

    for regex in liste_regex:
        if re.match(regex, ligne):
            return False
    return True

def preprocessing_fichier(nom_fichier) :
    """
    string -> string
    à partir du nom d'un fichier de sous-titres, renvoie le texte des sous-titres
    """
    fichier = open(nom_fichier, 'r', encoding = "ISO-8859-1")
    lignes = fichier.readlines()
    fichier.close()
    texte = ""
    for ligne in lignes :
        if verifier_ligne(ligne) :

            m = re.sub(r'[^\w'+"-"']', ' ', ligne)
            
            texte += m
    
    return texte


def scan_folder(parent_folder, corp):
    """
    retourne corpus des textes contenus dans parent_folder sous forme de liste de string
    """
    # iterate over all the files in directory 'parent_folder'
    for file_name in os.listdir(parent_folder):
        if file_name.endswith(".txt"):
            path = parent_folder+"/"+file_name
            
            texte = preprocessing_fichier(path)
            
            corp.append(texte)
        
        else:
            current_path = "".join((parent_folder, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, corp)
    
    return corp


#pour eviter les variables globales
def get_corpus(parent_folder):
    c = []
    res = scan_folder(parent_folder, c)
    return res


def somme_lignes(tab):
    """somme les lignes d'un tableau"""
    res = []
    for i in range(len(tab[0])):
        res.append(0)
    for k in range(len(tab)):
        for i in range(len(tab[0])):
                res[i] += tab[k][i]
    return np.array(res)


def df_n_plus_presents(n, corpus, stopwords_set):
    """
    int * list[string] * set(string) -> dataFrame
    dataFrame des n mots les plus present dans le corpus
    """
    vectorizer = CountVectorizer(stop_words = stopwords_set)
    X = vectorizer.fit_transform(corpus)
    dico = vectorizer.get_feature_names()
    nb_occ = somme_lignes(X.toarray())
    
    ind = np.argpartition(nb_occ, -n)[-n:]
    ind = ind[np.argsort(-nb_occ[ind])]
    words = [dico[i] for i in ind]

    words_count = []
    i = 0
    for i in range(len(words)):
        words_count.append(nb_occ[ind[i]])

    df = pd.DataFrame(np.column_stack([words, words_count]), columns=['Word', 'Nb_occ'])
    df.Nb_occ=pd.to_numeric(df.Nb_occ)
    
    return df  


def get_hist(df, x_axis, y_axis, titre, colour, font_size=None, horizontal=False):
    """
    
    """
    if horizontal:
        hist = df.plot.barh(x=x_axis, y=y_axis, color=colour, title =titre, fontsize = font_size, edgecolor = "none").get_figure()
    else:
        hist = df.plot.bar(x=x_axis, y=y_axis, color=colour, title =titre, fontsize = font_size, edgecolor = "none").get_figure()
    path_fig = "img/"+titre+'.png'
    hist.savefig(path_fig,  bbox_inches="tight")
    

    
    

def __main__() :
    
    #definition de l'ensemble de stopwords
    nltk_sw = set(stopwords.words('english'))
    sklearn_sw = set(stop_words.ENGLISH_STOP_WORDS)
    stopwords_set = nltk_sw | sklearn_sw
    l_nb = [str(i) for i in range(1000000)]
    l_mots = ["don", "yeah", "hey", "okay", "oh", "uh", "yes", "ok"]
    for mot in l_mots :
        stopwords_set.add(mot)
    for nb in l_nb:
        stopwords_set.add(nb)
    
    
        
    n = 50
    corpus = get_corpus("data/1___Lost/01")
    df_count = df_n_plus_presents(n, corpus, stopwords_set)
    titre = "les "+str(n)+" mots les plus presents dans la premiere saison de Lost"
    get_hist(df_count, "Word", "Nb_occ", titre, "teal", 7, True)
    print(df_count)
    
    
#__main__()

t = "data/2___Heroes/04/04__Acceptance.txt"
print(preprocessing_fichier(t))



