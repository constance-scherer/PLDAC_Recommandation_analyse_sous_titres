import os
import re
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import dok_matrix
import numpy as np
import pandas as pd
from scipy.spatial.distance import sqeuclidean, cosine
from collections import OrderedDict
from utils.preprocessing_cleaned_data import *
from utils.predictions_notes import *
import pickle
print("Import predictions_content ok")



def pred_content(uid, 
	iid, 
	d_name,
	d_user, 
	d_ind, 
	d_titre_filename, 
	d_filename_titre, 
	d_id_username, 
	d_id_serie, 
	similarities,
	k=3) :
    """
    prédire la note de l'utilisateur uid pour la serie iid 
    (moyenne sur les k ppv de iid chez uid)
    """
    
    # récupérer toutes les séries notées par uid
    u = d_id_username[uid]
    series_notes = d_user[u]
    series = series_notes.keys()
    #notes = series_notes.values()
    
    # récupérer le vecteur de similarité entre iid et toutes les autres séries
    i = d_id_serie[iid]
    f = d_titre_filename[i]
    n_iid = -1
    if f in d_ind.keys() :
        n_iid = d_ind[f]
    
    simil = similarities[n_iid]
    
    # on parcourt les séries que l'utilisateur a vu
    series_ind = []
    for s in series :
        f = d_titre_filename[s]
        if f in d_ind.keys() :
            series_ind.append(d_ind[f])
        
    
    d_simil = {}
    for ind in series_ind :
        d_simil[ind] = simil[ind]
    
    
    sorted_x = sorted(d_simil.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
     
    sorted_dict = OrderedDict(sorted_x)
    
    series_plus_similaires = list(sorted_dict)
    
    kppv = series_plus_similaires[:k]
    
    notes = []
    
    for ind in kppv :
        f = d_name[ind]
        t = d_filename_titre[f]
        n = series_notes[t]
        notes.append(n)
        
    return np.mean(notes)