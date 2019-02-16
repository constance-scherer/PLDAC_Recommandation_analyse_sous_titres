#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import nltk
import fnmatch
import numpy as np
import pandas as pd
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from remove_empty_folders import *
from language_detector import *
from remove_empty_folders import *

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


def getRidOfGrabInfo(path):
    filenames= sorted(os.listdir(path)) 
    # loop through all the files and folders
    for filename in filenames:
    # check whether the current object is a folder or not (ie check if it's a show)
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
            path_folder = path+"/"+filename
            l = fnmatch.filter(sorted(os.listdir(path_folder)), '*.txt')
            if l != []:
                for useless_file in l:
                    os.remove(path_folder+"/"+useless_file)


def getRidOfNonEnglishEpisodes(path):
    # get all files' and folders' names in the current directory
    filenames= sorted(os.listdir(path)) 
    # loop through all the files and folders
    for filename in filenames:
         # check whether the current object is a folder or not (ie check if it's a show)
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
                show_path = path+"/"+filename
                nb_seasons = sum(os.path.isdir(os.path.join(show_path, i)) for i in sorted(os.listdir(show_path)))
                
                for i in range(1, nb_seasons+1):
                    if i < 10:
                        season_path = show_path+"/0"+str(i)
                    else:
                        season_path = show_path+"/"+str(i)
                    
                    for episode in sorted(os.listdir(season_path)):
                        episode_path = season_path+"/"+episode
                        if not isEnglish(episode_path):
                            os.remove(episode_path)

def removeFilesAndFoldersThatNeedToGo(path):
    getRidOfGrabInfo(path)
    getRidOfNonEnglishEpisodes(path)
    removeEmptyFolders(path, removeRoot=True)


def get_corpus_as_episodes(path):
    """each text in corpus is an episode"""
    corpus = []
    
    # get all files' and folders' names in the current directory
    filenames= sorted(os.listdir(path)) 

    # loop through all the files and folders
    for filename in filenames:
         # check whether the current object is a folder or not (ie check if it's a show)
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
                show_path = path+"/"+filename
                nb_seasons = sum(os.path.isdir(os.path.join(show_path, i)) for i in sorted(os.listdir(show_path)))
                
                for i in range(1, nb_seasons+1):
                    if i < 10:
                        season_path = show_path+"/0"+str(i)
                    else:
                        season_path = show_path+"/"+str(i)
                    
                    for episode in sorted(os.listdir(season_path)):
                        episode_path = season_path+"/"+episode
    
                        f = open(episode_path, 'r',encoding='utf-8', errors='ignore')
                        lines = f.readlines()
                        f.close()
                        text =""
                        for line in lines :
                            line = line.lower()
                            if verifier_ligne(line):
                                new_line = transformer_ligne(line)
                                text += new_line
                        corpus.append(text)
                            
    return corpus

def get_corpus_as_seasons(path):
    """each text in corpus is a season"""
    corpus = []
    
    # get all files' and folders' names in the current directory
    filenames= sorted(os.listdir(path)) 

    # loop through all the files and folders
    for filename in filenames:
         # check whether the current object is a folder or not (ie check if it's a show)
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
                show_path = path+"/"+filename
                nb_seasons = sum(os.path.isdir(os.path.join(show_path, i)) for i in sorted(os.listdir(show_path)))
                
                for i in range(1, nb_seasons+1):
                    text =""
                    if i < 10:
                        season_path = show_path+"/0"+str(i)
                    else:
                        season_path = show_path+"/"+str(i)
                    
                    for episode in sorted(os.listdir(season_path)):
                        episode_path = season_path+"/"+episode
    
                        f = open(episode_path, 'r',encoding='utf-8', errors='ignore')
                        lines = f.readlines()
                        f.close()
                        for line in lines :
                            line = line.lower()
                            if verifier_ligne(line):
                                new_line = transformer_ligne(line)
                                text += new_line
                    corpus.append(text)
                            
    return corpus

def get_corpus_as_shows(path):
    """each text in corpus is a show"""
    corpus = []
    
    # get all files' and folders' names in the current directory
    filenames= sorted(os.listdir(path)) 

    # loop through all the files and folders
    for filename in filenames:
         # check whether the current object is a folder or not (ie check if it's a show)
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
                text =""
                show_path = path+"/"+filename
                nb_seasons = sum(os.path.isdir(os.path.join(show_path, i)) for i in sorted(os.listdir(show_path)))
                
                for i in range(1, nb_seasons+1):
                    if i < 10:
                        season_path = show_path+"/0"+str(i)
                    else:
                        season_path = show_path+"/"+str(i)
                    
                    for episode in sorted(os.listdir(season_path)):
                        episode_path = season_path+"/"+episode
    
                        f = open(episode_path, 'r',encoding='utf-8', errors='ignore')
                        lines = f.readlines()
                        f.close()
                        for line in lines :
                            line = line.lower()
                            if verifier_ligne(line):
                                new_line = transformer_ligne(line)
                                text += new_line
        corpus.append(text)
                            
    return corpus

def get_corpus(path, texts_as="episodes"):
    if texts_as == "seasons":
        return get_corpus_as_seasons(path)
    if texts_as == "shows":
        return get_corpus_as_shows(path)
    
    return get_corpus_as_episodes(path)           



def getDicts(path):
    res = dict() #  keys : show id     values: dict(key:id season, value: nb  ep season)
    res2 = dict() # keys : show id     values: show title
    j = 1
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
            #l = np.cumsum(liste)
            l = list(liste)
            seasons_list = list(range(1, nb_saisons+1))
            dico_serie = dict(zip(seasons_list, l))
            res[j] = dico_serie
            res2[j] = filename
            j += 1
    
    return res, res2


def getTfidfDataFrame(corpus, my_stopwords=None, my_tokenizer=None):
    vectorizer = TfidfVectorizer(stop_words = my_stopwords, tokenizer=my_tokenizer)
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

def getTfDataFrame(corpus, my_stopwords=None, my_tokenizer=None):
    vectorizer = CountVectorizer(stop_words = my_stopwords, tokenizer=my_tokenizer)
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
            