#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils.preprocessing_cleaned_data import *
from utils.swSets import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score



# def get_corpus(path, liste_filenames):
# 	"""each text in corpus is a show"""
# 	corpus = []
	
# 	# get all files' and folders' names in the current directory
# 	filenames= sorted(os.listdir(path)) 

# 	# loop through all the files and folders
# 	for filename in liste_filenames:
# 		if filename[0] == "." : # dossier caché
# 			continue
# 		# check whether the current object is a folder or not (ie check if it's a show)
# 		if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
# 				text =""
# 				show_path = path+"/"+filename
# 				for season in sorted(os.listdir(show_path)):
# 					if season[0] == "." : # dossier caché
# 						continue
# 					season_path = show_path+"/"+season
# 					for episode in sorted(os.listdir(season_path)):
# 						if episode[0] == "." : # dossier caché
# 							continue
# 						episode_path = season_path+"/"+episode
	
# 						f = open(episode_path, 'r',encoding='utf-8', errors='ignore')
# 						lines = f.readlines()
# 						f.close()
# 						for line in lines :
# 							text += line
# 		corpus.append(text)
							
# 	return corpus


def similarities(path, my_stopwords=stopwords_set) :
	#corpus = get_corpus(path, liste_filenames)
	corpus = get_corpus(path, texts_as='shows')
	print("get corpus ok")
	sparse_mat = getTfidfSparseMat(corpus, my_stopwords=stopwords_set)
	print("get tf idf ok")

	similarities = cosine_similarity(sparse_mat)
	print("cosine similarity ok")
	return similarities


def most_similar(path, similarities, nb_reco=3, my_stopwords=stopwords_set) :
	print("reco par similarité start")
	d_info, d_name = getDicts(path)
	print("getdict ok")
	cpt = 100
	most_similar = dict()
	for i in range(len(similarities)):
		if i%cpt == 0 : 
			print("i = ", i)
		show = np.array(similarities[i])
		ind = np.argpartition(show, -(nb_reco+1))[-(nb_reco+1):]
		ind = ind[np.argsort(show[ind])]
		ind = list(ind[:-1])
		ind.reverse()
		most_similar[d_name[i+1]] = [d_name[ind[j]+1] for j in range(0, nb_reco)]

	return most_similar

