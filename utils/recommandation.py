from utils.collaborative import *
from utils.similarities import *
from utils.predictions_content import *
from utils.ndcg import *
import operator
import pickle
from collections import OrderedDict
from utils.predictions_notes import *


def reco_fc_kSVD(username, 
	d_username_id, 
	d_itemname_id, 
	d_user,
	U_ksvd,
	I_ksvd,
	u_means,
	i_means,
	mean,
	nb_reco=10):
	"""
	renvoie nb_reco recommandation pour l'utilisateur username
	filtrage collaboratif, kSVD
	"""
	uid = d_username_id[username]
	d_notes = dict()
	for serie, iid in d_itemname_id.items() :
	    if serie not in d_user[username].keys() :
	        # prediction
	        p = pred_func_ksvd(uid, iid, U_ksvd, I_ksvd, u_means, i_means, mean)
	        if p > 10 :
	            p = 10
	        d_notes[serie] = p 


	sorted_x = sorted(d_notes.items(), key=lambda kv: kv[1])
	sorted_x.reverse()

	sorted_dict = OrderedDict(sorted_x)
	reco = list(sorted_dict)
	top_reco = reco[:nb_reco]
		
	return top_reco


def reco_content(username,
	d_username_id,
	d_itemname_id,
	d_name,
	d_user,
	d_ind, 
	d_titre_filename, 
	d_filename_titre, 
	d_id_username, 
	d_id_serie, 
	sim, 
	nb_reco=10):
	"""
	renvoie nb_reco recommandation pour l'utilisateur username
	contenu
	"""
	uid = d_username_id[username]
	d_notes = dict()
	for serie, iid in d_itemname_id.items() :
	    if serie not in d_user[username].keys() :
	        # prediction
	        p = pred_content(uid, iid, d_name,
		d_user, 
		d_ind, 
		d_titre_filename, 
		d_filename_titre, 
		d_id_username, 
		d_id_serie, 
		sim)
	        if p > 10 :
	            p = 10
	        d_notes[serie] = p 


	sorted_x = sorted(d_notes.items(), key=lambda kv: kv[1])
	sorted_x.reverse()

	sorted_dict = OrderedDict(sorted_x)
	reco = list(sorted_dict)
	top_reco = reco[:nb_reco]

	return top_reco
