from utils.collaborative import *
from utils.similarities import *
from utils.predictions_content import *
from utils.ndcg import *
import operator
import pickle
from collections import OrderedDict
from utils.predictions_notes import *
from utils.recommandation import *


def dcg(series_ordonnees, pertinences) :
    d = pertinences[0]
    for i in range(1, len(pertinences)) :
        d += pertinences[i]/math.log2(i+1)
    return d

def idcg(d, user) :
    sorted_x = sorted(dico.items(), key=lambda kv: kv[1][1])
    sorted_x.reverse()
    sorted_dict = OrderedDict(sorted_x)
    series_ordonnees = list(sorted_dict)
    
    pertinences = []
    
    for serie in series_ordonnees :
        vraie_note = dico[serie][1]
        if vraie_note >= 7 :
            p = 1
        else :
            p = 0
        pertinences.append(p)
        
    return dcg(series_ordonnees, pertinences)

def get_ndcg_moy(predictions, 
				mat, 
				d_id_username, 
				d_id_serie
				) :

	
	d = dict()
	for i in range(0, len(predictions)) :
	  
	    note_predite = prediction[i]
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

	    sorted_x = sorted(dico.items(), key=lambda kv: kv[1])
	    sorted_x.reverse()
	    sorted_dict = OrderedDict(sorted_x)
	    series_ordonnees = list(sorted_dict)
	    
	    pertinences = []
	   
	    for serie in series_ordonnees :
	        vraie_note = dico[serie][1]
	        if vraie_note >= 7 :
	            p = 1
	        else :
	            p = 0
	        pertinences.append(p)
	       
	        
	    DCG = dcg(series_ordonnees, pertinences)
	    IDCG = idcg(d, u)
	    if IDCG != 0:
	        moy_ndcg.append(DCG/IDCG)
	    else : 
	        moy_ndcg.append(0)


	return np.mean(moy_ndcg)