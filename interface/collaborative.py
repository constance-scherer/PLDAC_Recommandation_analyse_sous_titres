import os
import re
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import dok_matrix
import numpy as np
import pandas as pd
from scipy.spatial.distance import sqeuclidean, cosine
from sklearn.decomposition import NMF, TruncatedSVD


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
	d_user = dict() #{username : {serie: note, serie : note}}

	for user in sorted(os.listdir(path)):
	    username = re.sub(".txt", "", user)
	    d_user[username] = dict()
	    with open(path+"/"+user) as file: 
	        lignes = file.readlines()
	        for ligne in lignes :
	            serie, rating = ligne.split(" ")
	            rating = rating.rstrip("\n")
	            rating = float(rating)
	            
	            d_user[username][serie] = rating
	return d_user




def get_liste_series(d_user) :
	liste_series = set()
	for username, d_s in d_user.items() :
	    for serie, rating in d_s.items() :
	        liste_series.add(serie)
	liste_series = list(liste_series)

	return liste_series



def get_data(d_user) :
	data = []
	for username, d_s in d_user.items() :
	    for serie, rating in d_s.items() :
	        data.append( (username, serie, rating) )
	return data


def group_by_user(tuple_list):
    r_dic = {}
    for uid,iid,rating in tuple_list:
        list_rev = r_dic.get(uid,[])
        list_rev.append(rating)
    
        r_dic[uid] =list_rev
    return r_dic


def group_by_item(tuple_list):
    r_dic = {}
    for uid,iid,rating in tuple_list:
        list_rev = r_dic.get(iid,[])
        list_rev.append(rating)
    
        r_dic[iid] =list_rev
    return r_dic


def get_all_data(data) :
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

def get_Uksvd_Iksvd(train, train_mat, num_user, num_item) :
	model =TruncatedSVD(n_components=150)

	# (1) compute means of training set
	mean = np.mean([rating for (uid,iid),rating in train_mat.items()])

	# user and item deviation to mean
	u_means = {u:(np.mean(ratings - mean)) for u,ratings in group_by_user(train).items()}
	i_means = {i:(np.mean(ratings) - mean) for i,ratings in group_by_item(train).items()}




	# (2) normalize training matrix
	tmn_k = dok_matrix((num_user, num_item), dtype=np.float32)

	for (uid,iid), rating in train_mat.items():
	    tmn_k[uid,iid] = rating - mean - u_means.get(uid,0) - i_means.get(iid,0)
	    
	# (3) factorize matrix
	model_kor = TruncatedSVD(n_components=150)


	U_ksvd = model.fit_transform(tmn_k)
	I_ksvd = model.components_.transpose()

	return mean, u_means, i_means, U_ksvd, I_ksvd



def pred_func_ksvd(uid,iid, U_ksvd, I_ksvd, u_means, i_means, mean):
	
    Uu = U_ksvd[uid]
    Ii = I_ksvd[iid]
    Bu = u_means.get(uid,0)
    Bi = i_means.get(iid, 0)
    
    return np.dot(Uu, Ii) + mean + Bu + Bi



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



def create_reversed_dic(u_dic, i_dic) :
	reversed_u_dic =  dict([[v,k] for k,v in u_dic.items()]) #{user id : username}
	reversed_i_dic = dict([[v,k] for k,v in i_dic.items()]) #{item id: item title}
	return reversed_u_dic, reversed_i_dic





def plus_proche_voisin(username, d_username_id, Full) :
    user_id = d_username_id[username]
    f = Full[user_id].todense()
    d_min = 1000
    i_min = -1
    for i in range(0, Full.shape[0]) :
        if i != user_id :
            fi = Full[i].todense()
            d = cosine(f, fi)
            if d < d_min :
                i_min = i
                d_min = d
    return i_min



def series_pas_en_commun(username1, username2, d_user, d_itemname_id) :
    """id series que user2 a vu et pas user1"""
    series_u1 = d_user[username1].keys()
    series_u2 = d_user[username2].keys()
    
    s = set()
    #for s1 in series_u1 :
     #   if s1 not in series_u2 and d_user[username1][s1] >= 7:
            
      #      s.add(d_itemname_id[s1])
            
    for s2 in series_u2:
        if s2 not in series_u1 and d_user[username2][s2] >= 7:
            s.add(d_itemname_id[s2])
            
    return list(s)




def recommandation(username1, data, d_user, nb_pred, U_ksvd, I_ksvd, u_means, i_means, mean) :
    d_username_id, d_itemname_id, Full = create_sparse_mat(data)
    reversed_u_dic, reversed_i_dic = create_reversed_dic(d_username_id, d_itemname_id)
    ppv = plus_proche_voisin(username1, d_username_id, Full)
    username2 = ""
    for username, ID in d_username_id.items() :
        if ID == ppv :
            username2 = username
            break
    if username2 == "" :
        print("problème avec le nom du plus proche voisin")
        return

    series_non_vues = series_pas_en_commun(username1, username2, d_user, d_itemname_id)

    predictions =dict()
    for i in series_non_vues:
        predictions[i] = pred_func_ksvd(d_username_id[username1], i, U_ksvd, I_ksvd, u_means, i_means, mean)  #{id serie: note predite}
    sorted_rec = [(k, predictions[k]) for k in sorted(predictions, key=predictions.get, reverse=True)]
    res = [reversed_i_dic[sorted_rec[i][0]] for i in range(nb_pred)]
    
    return res, sorted_rec


