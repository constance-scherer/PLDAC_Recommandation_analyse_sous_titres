#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils.preprocessing_cleaned_data import *
from utils.swSets import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score


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
		most_similar[d_name[i]] = [d_name[ind[j]] for j in range(0, nb_reco)]

	return most_similar

