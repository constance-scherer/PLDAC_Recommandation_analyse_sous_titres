from collections import OrderedDict
import numpy as np
import math
print("Import eval_reco.py ok")

def dcg(series_ordonnees, pertinences) :
	"""
	series_ordonnees : liste de séries recommandées pour un utilisateur, dans l'ordre 
	pertinences : jugement de pertinence associé à chaque série pour l'utilisateur
	renvoie la DCG associée à cet ordonnancement et ces pertinences
	"""
	if len(pertinences) == 0 :
		return 0 
	d = pertinences[0]
	for i in range(1, len(pertinences)) :
		d += pertinences[i]/math.log2(i+1)
	return d

def idcg(dico, user) :
	"""
	dico : dict {serie: (note_predite, vraie_note)}
	user : nom de l'utilisateur
	renvoie la DCG sur le meilleur ordonnancement possible pour un user donné
	"""
	# on recupere les series ordonnees selon les notes de l'utilisateur
	sorted_x = sorted(dico.items(), key=lambda kv: kv[1][1])
	sorted_x.reverse()
	sorted_dict = OrderedDict(sorted_x)
	series_ordonnees = list(sorted_dict)
	
	# on recupere les pertinences (remise entre 0 et 1)
	pertinences = []
	for serie in series_ordonnees :
		vraie_note = dico[serie][1]
		if vraie_note >= 7 :
			p = 1
		else :
			p = 0
		pertinences.append(p)
		
	return dcg(series_ordonnees, pertinences)

def get_ndcg_moy_train(predictions, 
				mat, 
				d_id_username, 
				d_id_serie) :
	"""
	predictions : liste de triplets (uid, iid, note_predite)
	mat : (uid, iid), rating
	d_id_username : {id : username}
	d_id_serie : {id : serie}
	renvoie la nDCG moyenne sur tous les utilisateurs de mat
	"""

	#on cree un dictionnaire {u : serie : (note_predite, vraie_note)}
	d = dict()
	for i in range(0, len(predictions)) :
		note_predite = predictions[i]
		(u, i), rating = list(mat.items())[i]
		username = d_id_username[u]
		itemname = d_id_serie[i]
		
		if username not in d.keys() :
			d[username] = {itemname : (note_predite, rating)}
		else :
			d[username][itemname] = (note_predite, rating)

	moy_ndcg = []
	i = 0
	for u, dico in d.items() :
		# on calcule l'ordonnancement sur les notes prédites
		sorted_x = sorted(dico.items(), key=lambda kv: kv[1][0])
		sorted_x.reverse()
		sorted_dict = OrderedDict(sorted_x)
		series_ordonnees = list(sorted_dict)
		
		# on calcule les pertinences associées (entre 0 et 1)
		pertinences = []
		for serie in series_ordonnees :
			vraie_note = dico[serie][1]
			if vraie_note >= 7 :
				p = 1
			else :
				p = 0
			pertinences.append(p)
		# on obtient la nDCG
		DCG = dcg(series_ordonnees, pertinences)
		IDCG = idcg(dico, u)
		if IDCG != 0:
			moy_ndcg.append(DCG/IDCG)
		else : 
			moy_ndcg.append(0)


	return np.mean(moy_ndcg)

def get_ndcg_moy_test(predictions, 
				mat, 
				d_id_username, 
				d_id_serie) :

	"""
	predictions : liste de triplets (uid, iid, note_predite)
	mat : liste de triplets uid, iid, rating
	d_id_username : {id : username}
	d_id_serie : {id : serie}
	renvoie la nDCG moyenne sur tous les utilisateurs de mat
	"""

	#on cree un dictionnaire {u : serie : (note_predite, vraie_note)}
	d = dict()
	for i in range(0, len(predictions)) :
	  
		note_predite = predictions[i]
		u, i, rating = mat[i]
		username = d_id_username[u]
		itemname = d_id_serie[i]
		
		if username not in d.keys() :
			d[username] = {itemname : (note_predite, rating)}
		else :
			d[username][itemname] = (note_predite, rating)


	moy_ndcg = []
	i = 0
	for u, dico in d.items() :
		# on calcule l'ordonnancement sur les notes prédites
		sorted_x = sorted(dico.items(), key=lambda kv: kv[1])
		sorted_x.reverse()
		sorted_dict = OrderedDict(sorted_x)
		series_ordonnees = list(sorted_dict)

		# on calcule les pertinences associées (entre 0 et 1)
		pertinences = []
		for serie in series_ordonnees :
			vraie_note = dico[serie][1]
			if vraie_note >= 7 :
				p = 1
			else :
				p = 0
			pertinences.append(p)

		# on obtient la nDCG
		DCG = dcg(series_ordonnees, pertinences)
		IDCG = idcg(dico, u)
		if IDCG != 0:
			moy_ndcg.append(DCG/IDCG)
		else : 
			moy_ndcg.append(0)

	return np.mean(moy_ndcg)


def tri_par_pop(liste_series, d_pop) :
	"""
	liste_series : liste de series ayant la meme note
	d_pop : dictionnaire des popularités
	"""
	
	l_pop = dict()
	i = 0
	for serie in liste_series :
		if serie not in d_pop.keys() :
			i+= 1
			continue
		l_pop[i] = d_pop[serie]
		i += 1
	sorted_x = sorted(l_pop.items(), key=lambda kv: kv[1])
	sorted_x.reverse()

	sorted_dict = OrderedDict(sorted_x)
	reco = [liste_series[i] for i, pop in sorted_x]
	return reco




def get_ndcg_moy_train_pop(predictions, 
				mat, 
				d_id_username, 
				d_id_serie,
				d_pop) :
	"""
	predictions : liste de triplets (uid, iid, note_predite)
	mat : (uid, iid), rating
	d_id_username : {id : username}
	d_id_serie : {id : serie}
	renvoie la nDCG moyenne sur tous les utilisateurs de mat
	"""

	#on cree un dictionnaire {u : serie : (note_predite, vraie_note)}
	d = dict()
	for i in range(0, len(predictions)) :
		note_predite = predictions[i]
		(u, i), rating = list(mat.items())[i]
		username = d_id_username[u]
		itemname = d_id_serie[i]
		
		if username not in d.keys() :
			d[username] = {itemname : (note_predite, rating)}
		else :
			d[username][itemname] = (note_predite, rating)

	moy_ndcg = []
	i = 0
	for u, dico in d.items() :
		
		i += 1
		# on calcule l'ordonnancement sur les notes prédites
		sorted_x = sorted(dico.items(), key=lambda kv: kv[1][0])
		sorted_x.reverse()
		sorted_dict = OrderedDict(sorted_x)
		series_ordonnees = list(sorted_dict)
		
		series_par_pop = tri_par_pop(series_ordonnees, d_pop)
		# on calcule les pertinences associées (entre 0 et 1)
		pertinences = []
		for serie in series_par_pop :
			vraie_note = dico[serie][1]
			if vraie_note >= 7 :
				p = 1
			else :
				p = 0
			pertinences.append(p)
		# on obtient la nDCG
		DCG = dcg(series_par_pop, pertinences)
		IDCG = idcg(dico, u)
		if IDCG != 0:
			moy_ndcg.append(DCG/IDCG)
		else : 
			moy_ndcg.append(0)


	return np.mean(moy_ndcg)

def get_ndcg_moy_test_pop(predictions, 
				mat, 
				d_id_username, 
				d_id_serie, 
				d_pop) :

	"""
	predictions : liste de triplets (uid, iid, note_predite)
	mat : liste de triplets uid, iid, rating
	d_id_username : {id : username}
	d_id_serie : {id : serie}
	renvoie la nDCG moyenne sur tous les utilisateurs de mat
	"""

	#on cree un dictionnaire {u : serie : (note_predite, vraie_note)}
	d = dict()
	for i in range(0, len(predictions)) :
	  
		note_predite = predictions[i]
		u, i, rating = mat[i]
		username = d_id_username[u]
		itemname = d_id_serie[i]
		
		if username not in d.keys() :
			d[username] = {itemname : (note_predite, rating)}
		else :
			d[username][itemname] = (note_predite, rating)


	moy_ndcg = []
	i = 0
	for u, dico in d.items() :

		# on calcule l'ordonnancement sur les notes prédites
		sorted_x = sorted(dico.items(), key=lambda kv: kv[1][0])
		sorted_x.reverse()
		sorted_dict = OrderedDict(sorted_x)
		series_ordonnees = list(sorted_dict)
		
		series_par_pop = tri_par_pop(series_ordonnees, d_pop)
		# on calcule les pertinences associées (entre 0 et 1)
		pertinences = []
		for serie in series_par_pop :
			vraie_note = dico[serie][1]
			if vraie_note >= 7 :
				p = 1
			else :
				p = 0
			pertinences.append(p)
		# on obtient la nDCG
		DCG = dcg(series_par_pop, pertinences)
		IDCG = idcg(dico, u)
		if IDCG != 0:
			moy_ndcg.append(DCG/IDCG)
		else : 
			moy_ndcg.append(0)

	return np.mean(moy_ndcg)
