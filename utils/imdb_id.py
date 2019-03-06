#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import omdb

def get_IMDB_id_from_title(title) :
    """
    title : string, titre de la série dont on recherche les infos
    renvoie l'id IMDB associée à cette série s'il existe, -1 sinon
    """
    info = omdb.search(title)
    series = []
    for dico in info :
        if dico['type'] == 'series' :
            series.append(dico)
    if len(series) == 0 : # si jamais la recherche ne renvoie pas de séries
        #print("Pas de séries pour ce titre")
        return -1
    return series[0]['imdb_id']



def get_ids(apikey, titles) :
	"""
	apikey : string, clé d'identifacation pour OMDB
	titles : list(string), liste des titres des séries
	écrit les id des séries dans un fichier
	limité à 1000 requêtes par jour
	on a tous les id.
	"""
	omdb.set_default('apikey', apikey) # identification
	cpt = 0
	series_rejetees = []
	with open('titles/titles_imdb_id.txt', 'a') as file2: # 'a' pour ajouter à la suite du fichier
	    for titre in log_progress(titles, every=1) :
	        ID = get_IMDB_id_from_title(titre)
	        if ID != -1 :
	            file2.write(titre+" "+ID+"\n")
	        else :
	            series_rejetees.append(titre)
	        cpt += 1