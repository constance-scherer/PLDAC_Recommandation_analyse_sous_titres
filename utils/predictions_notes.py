import os
import re
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import dok_matrix
import numpy as np
import pandas as pd
from scipy.spatial.distance import sqeuclidean, cosine
print("Import predictions_notes ok")

# take as input two lists of ratings

def MSE_err(truth,pred):
	"""
	computes MSE from real-pred difference
	"""
	return np.mean((truth-pred)**2)

def MAE_err(truth,pred):
	"""
	computes MAE from real-pred difference
	"""
	return np.mean(abs(np.array(truth-pred)))

def get_d_user(path) :
	"""
	path : str
	renvoie dictionnaire
	username : {titre serie : note}
	"""

	d_user = dict() #{username : {serie: note, serie : note}}

	for user in sorted(os.listdir(path)):
		if user[0] == '.' :
			continue
		#print('user = ', user)
		username = re.sub(".txt", "", user)
		#d_user[username] = dict()
		dico = dict()
		with open(path+"/"+user) as file: 
			lignes = file.readlines()
			for ligne in lignes :
				serie, rating = ligne.split(" ")
				rating = rating.rstrip("\n")
				rating = float(rating)
				dico[serie] = rating
				#d_user[username][serie] = rating
		if len(dico) > 3 :
			d_user[username] = dico

	return d_user


def get_liste_series(d_user) :
	"""
	d_user : username : {titre serie : note}
	renvoie liste des series
	"""
	liste_series = set()
	for username, d_s in d_user.items() :
		for serie, rating in d_s.items() :
			liste_series.add(serie)
	liste_series = list(liste_series)
	len(liste_series)

	return liste_series


def get_data(d_user) :
	"""
	d_user : username : {titre serie : note}
	renvoie data
	liste de tuple (username, serie, rating)
	"""
	data = []
	for username, d_s in d_user.items() :
		for serie, rating in d_s.items() :
			data.append( (username, serie, rating) )
	return data



def get_all_data(data) :
	"""

	"""
	# We first remap users and item to ids between (0,len(user)) and (0,len(item))
	u_dic = {} #{username : user id}
	i_dic = {} #{item title : item id}
			
	all_data = [] #[(user id, item id, rating)]
		
	d_username_id = dict()
	d_itemname_id = dict()
	for uid,iid,rating in data:  # iterating on all data
		
		uk = u_dic.setdefault(uid,len(u_dic))
		ik = i_dic.setdefault(iid,len(i_dic))
		all_data.append((uk,ik,float(rating)))
		d_username_id[uid] = uk
		d_itemname_id[iid] = ik

	num_user = len(u_dic)
	num_item = len(i_dic)
	return all_data, num_user, num_item


def get_train_test(num_user, num_item, all_data, test_size=10) :
	"""
	"""
	# We take 10% of the train set as test data
	train_mat = dok_matrix((num_user, num_item), dtype=np.float32)
	test = []
	train = []
		
	for i,(uid,iid,rating) in enumerate(all_data):
		if i%test_size == 0: #one out of 10 is for test
			test.append((uid,iid,rating))
		else:
			train.append((uid,iid,rating))
			train_mat[uid,iid] = rating

	return train, train_mat, test


def create_sparse_mat(data) : 
	# We first remap users and item to ids between (0,len(user)) and (0,len(item))
	u_dic = {}
	i_dic = {}

	all_data = []

	d_username_id = dict()
	d_itemname_id = dict()
	for uid,iid,rating in data:  # iterating on all data

		uk = u_dic.setdefault(uid,len(u_dic))
		ik = i_dic.setdefault(iid,len(i_dic))
		all_data.append((uk,ik,float(rating)))
		d_username_id[uid] = uk
		d_itemname_id[iid] = ik

	num_user = len(u_dic)
	num_item = len(i_dic)
	
	# (1) Create sparse matrix from all ratings
	Full = dok_matrix((num_user, num_item), dtype=np.float32)

	for uid,iid,rating in all_data:
		Full[uid,iid] = float(rating)
		
	return d_username_id, d_itemname_id, Full



def reverse_dict(d) :
	r_d = {v: k for k, v in d.items()}
	return r_d


def get_d_titre_filename(path) :
	d_titre_filename = dict()
	with open(path) as file :
		lignes = file.readlines()
		for ligne in lignes :
			l = ligne.split(" ")
			titre = l[0]
			filename = l[1].rstrip('\n')
			d_titre_filename[titre] = filename

	return d_titre_filename