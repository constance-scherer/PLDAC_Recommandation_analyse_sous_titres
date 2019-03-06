#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

def get_titles(path) :
	"""
	path : string
	renvoie un dictionnaire {nom_série : nb_total_de_saisons}
	les noms des séries sont légérement transformés
	"""

	d = dict()
	filenames = sorted(os.listdir(path)) 
	for filename in filenames:
	    if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
	        show_path = path+"/"+filename
	        nb_seasons = sum(os.path.isdir(os.path.join(show_path, i)) for i in sorted(os.listdir(show_path)))
	        r = r'([0-9])+(___)(.+)'
	        titre = re.search(r, filename).group(3).lower()

	        
	        titre = re.sub('(_)+', '-', titre)
	        titre = re.sub('\.', '', titre)
	        titre = re.sub('(-\([0-9]+\))', '', titre)
	        d[titre] = nb_seasons

	return d


def dico_to_file(dico, nom_fichier) :
	"""
	dico : {nom_serie : qqch}
	écrit dans le fichier nom_fichier les infos de dico sous la forme 'nom_serie qqch'
	"""

	with open(nom_fichier, 'w') as file:  # Use file to refer to the file object
	    for nom, qqch in d.items():
	        file.write(nom+" "+str(qqch)+'\n')


