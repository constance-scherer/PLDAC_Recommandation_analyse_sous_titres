import pickle
import math
from collections import OrderedDict
from utils.recommandation import *
print("Import ndcg ok")


def dcg(reco, user, d_pert_user) :
    """
    reco : liste de titres de series, dans l'ordre de la recommandation
    user : utilisateur pour qui on fait la recommandation
    d_pert_user : dictionnaire des pertinences par utilisateurs
    renvoie la DCG pour les recommandations reco pour l'utilisateur user
    """
    d_pert = d_pert_user[user]
    if reco[0] not in d_pert.keys() :
        res = 0
    else :
        res = d_pert[reco[0]]
    for i in range(1, len(reco)) :
        serie = reco[i]
        if serie not in d_pert.keys() :
            continue
        res += d_pert[serie]/math.log2(i+1)
        
    return res


def ideal_reco(user, k, d_pert_user, d_user) :
    """
    user : utilisateur pour lequel on veut faire la recommandation
    k : nombre de series a predire
    d_pert_user : dictionnaire des pertinences utilisateurs
    d_user : dictionnaire user
    renvoie la meilleure recommandation pour user (pour la ndcg)
    """
    d_notes = d_user[user]
    series_vues = list(d_notes.keys())
    d_pert = d_pert_user[user]
    sorted_x = sorted(d_pert.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    sorted_dict = OrderedDict(sorted_x)
    ideal_reco = []
    i = 0
    while len(ideal_reco) < k and i < len(d_pert) :
        serie, pert = sorted_x[i]
        if serie not in series_vues :
            ideal_reco.append(serie)
        i += 1
        
    return ideal_reco


def ndcg(reco, user, d_pert_user, d_user) :
    """
    reco : liste de titres de series, dans l'ordre de la recommandation
    user : utilisateur pour qui on fait la recommandation
    d_pert_user : dictionnaire des pertinences par utilisateurs
    renvoie la nDCG pour les recommandations reco pour l'utilisateur user
    """
    DCG = dcg(reco, user, d_pert_user)
    k = len(reco)
    d_pert = d_pert_user[user]
    sorted_x = sorted(d_pert.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    sorted_dict = OrderedDict(sorted_x)
    ideal_r = ideal_reco(user, k, d_pert_user, d_user)
    
    IDCG = dcg(ideal_r, user, d_pert_user,)
    if IDCG != 0 :
        return DCG/IDCG
    return 0.

def ndcg_moy_fc(d_pert_user,
        d_username_id, 
        d_itemname_id, 
        d_user,
        U_ksvd,
        I_ksvd,
        u_means,
        i_means,
        mean) :
    ndcg_fc = []
    for username in d_user.keys() :
        top_reco = reco_fc_kSVD(username,   d_username_id, 
        d_itemname_id, 
        d_user,
        U_ksvd,
        I_ksvd,
        u_means,
        i_means,
        mean)
        ndcg_fc.append(ndcg(top_reco, username, d_pert_user, d_user))

    return np.mean(ndcg_fc)


def ndcg_moy_content(d_pert_user,
        d_username_id,
        d_itemname_id,
        d_name,
        d_user,
        d_ind, 
        d_titre_filename, 
        d_filename_titre, 
        d_id_username, 
        d_id_serie, 
        sim ) :
    ndcg_c = []
        
    for username in d_user.keys() :
        top_reco = reco_content(username,
            d_username_id,
            d_itemname_id,
            d_name,
            d_user,
            d_ind, 
            d_titre_filename, 
            d_filename_titre, 
            d_id_username, 
            d_id_serie, 
            sim)
        ndcg_c.append(ndcg(top_reco, username, d_pert_user, d_user))

    return np.mean(ndcg_c)