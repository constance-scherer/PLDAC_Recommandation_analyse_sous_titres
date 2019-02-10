#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:58:16 2019

@author: constancescherer
"""

from prepocessing import *

def TFIDF(n, corpus, stopwords_set) :
    """
    """
    
    vectorizer = TfidfVectorizer(stop_words = stopwords_set)
    X = vectorizer.fit_transform(corpus)
    dico = vectorizer.get_feature_names()
    dense = X.todense()
    denselist = dense.tolist()
    df_tfidf = pd.DataFrame(denselist, columns=dico)
    
    order = np.argsort(-df_tfidf.values, axis=1)[:, :n]
    result = pd.DataFrame(df_tfidf.columns[order], 
                          columns=['top{}'.format(i) for i in range(1, n+1)],
                          index=df_tfidf.index)
    
    return df_tfidf, result

    
def __main__() :
     
    
    n = 20
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
      
    corpus = get_corpus("data/1___Lost/01")
    df_tfidf, result = TFIDF(n, corpus, stopwords_set)
        
    nbep = 1
    for i in range(nbep):
        lig_df = df_tfidf[i:i+1]
        lig_res = result[i:i+1]
    
        mots = list(np.array(lig_res)[0])
        values = [float(lig_df[mot]) for mot in mots]
        df = pd.DataFrame(np.column_stack([mots, values]), columns=['Word', 'Tfidf'])
        df.Tfidf = pd.to_numeric(df.Tfidf)
        df.sort_values(by ='Tfidf', inplace = True, ascending=True)
        
    
        titre = "top "+str(n)+" tf-idf scores for Lost season 1 episode "+str(i+1)
        get_hist(df, "Word", "Tfidf", titre, "limegreen", horizontal=True)
    
    
__main__()
  


