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
from utils.remove_empty_folders import *
from utils.language_detector import *
from utils.remove_empty_folders import *

def verifier_ligne(ligne):
    """return True si la ligne est un sous-titre, False sinon"""
    timestamp_regex = r'[0-9]{2}:[0-9]{2}:[0-9]{2}' 
    subnumber_regex =r'^[0-9]+$'
    
    liste_regex = [timestamp_regex, subnumber_regex]

    if "addic7ed" in ligne:
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
    tag_regex = r'<(/)*(.)+>' #to get rif of tags
    alphanum_regex = r'\W+'  #get rid of non alphanumeric characters
    new_line = re.sub(tag_regex, '', ligne)
    new_line = re.sub(alphanum_regex, ' ', new_line)
    return new_line

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
                for season in sorted(os.listdir(show_path)):
                    season_path = show_path+"/"+season
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
                for season in sorted(os.listdir(show_path)):
                    season_path = show_path+"/"+season
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
                for season in sorted(os.listdir(show_path)):
                    season_path = show_path+"/"+season
                    text = ""
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
                for season in sorted(os.listdir(show_path)):
                    season_path = show_path+"/"+season
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
    j = 0
    filenames= sorted(os.listdir(path)) # get all files' and folders' names in the current directory
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)): # check whether the current object is a folder or not
            show_path = path+"/"+filename
            l = []
            nb_saisons = sum(os.path.isdir(os.path.join(show_path, i)) for i in sorted(os.listdir(show_path)))
            for season in sorted(os.listdir(show_path)):
                    season_path = show_path+"/"+season
                    nb_eps_saison = len(fnmatch.filter(os.listdir(season_path), '*.txt'))
                    l.append(nb_eps_saison)
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
    
def getTfidfSparseMat(corpus, my_stopwords=None, my_tokenizer=None):
    vectorizer = TfidfVectorizer(stop_words = my_stopwords, tokenizer=my_tokenizer)
    return vectorizer.fit_transform(corpus)

def getTfDataFrame(corpus, my_stopwords=None, my_tokenizer=None):
    vectorizer = CountVectorizer(stop_words = my_stopwords, tokenizer=my_tokenizer)
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    
def getTfSparseMat(corpus, my_stopwords=None, my_tokenizer=None):
    vectorizer = TfidfVectorizer(stop_words = my_stopwords, tokenizer=my_tokenizer)
    return vectorizer.fit_transform(corpus)
            
