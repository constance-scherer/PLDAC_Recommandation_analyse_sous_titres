import os
import re
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import dok_matrix
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
#import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import dok_matrix
import numpy as np
import pandas as pd
from scipy.spatial.distance import sqeuclidean, cosine
import tensorflow as tf
import math
np.random.seed(0)
#import matplotlib.pyplot as plt



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
     



def pred_func_svd(U_svd, I_svd, uid,iid):
    
    Uu = U_svd[uid]
    Ii = I_svd[iid]
    
    return np.dot(Uu, Ii)  


def pred_func_mean(uid,iid):
    
    
    return mean


def pred_func_msvd(U_msvd, I_msvd, uid,iid, mean): 
    
    Uu = U_msvd[uid]
    Ii = I_msvd[iid]
    
    return np.dot(Uu, Ii) + mean


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
    p = np.dot(Uu, Ii) + mean + Bu + Bi
    #on ramène les notes au dessus de 10 à 10
    if p > 10 :
        p = 10
    
    return p



def pred_func_ksvd2(U_ksvd, I_ksvd, uid,iid, mean, u_means, i_means):
    Uu = U_ksvd[uid]
    Ii = I_ksvd[iid]
    Bu = u_means.get(uid,0)
    Bi = i_means.get(iid, 0)
    return np.dot(Uu, Ii) + mean + Bu + Bi




def pred_func(W, H, u, i):
    H_t = H.T
    return np.dot(W[u], H_t[i])

def replace_sup10(l):
    res = []
    for i in range(len(l)):
        if l[i] > 10.0:
            res.append(10.0)
        else:
            res.append(l[i])
    return res

def pred_func_bias(W, H, mean, user_bias, item_bias, u, i):
    """
    fonction de prediction avec biais pour la NMF
    """
    H_t = H.T
    return np.dot(W[u], H_t[i])+mean+ user_bias[u]+item_bias[i]



def predictions_NMF_biais(train_mat,test, nb_comp, num_user, num_item) :

    truth_tr = np.array([rating for (uid,iid),rating in train_mat.items()])
    truth_te = np.array([rating for uid,iid,rating in test])

    train_mat_mean = np.array(list(train_mat.values())).mean() #moyenne des notes dans la matrice train
    item_bias = [] #biais item : moyenne des notes reçues l'item - moyenne du dataset
    user_bias = [] #biais user : moyenne des notes données par l'utilisateur - moyenne du dataset

    for i in range(0, num_user):
        user = train_mat[i]
        user_mean = np.array(list(user.values())).mean()
        bias = user_mean - train_mat_mean
        if np.isnan(bias):
            user_bias.append(0)
        else:
            user_bias.append(bias)

    train_mat_T = train_mat.copy().transpose()
    for i in range(0, num_item):
        item = train_mat_T[i]
        item_mean = np.array(list(item.values())).mean()
        bias = item_mean - train_mat_mean
        if np.isnan(bias):
            item_bias.append(0)
        else:
            item_bias.append(bias)

    train_bias = dok_matrix((num_user, num_item), dtype=np.float32)

    for (u,i), rating in train_mat.items():
        train_bias[u,i] = rating - train_mat_mean - user_bias[u] - item_bias[i]


    A_orig = np.array(train_bias.todense())

    #on remplace les zeros (notes absentes) par des NaN
    for i in range(num_user):
        for j in range(num_item):
            if A_orig[i][j] == 0.0:
                A_orig[i][j] = np.NaN
                
    A_df = pd.DataFrame(A_orig)

    #on utilise un masque
    np_mask = A_df.notnull()

    # Boolean mask for computing cost only on valid (not missing) entries
    tf_mask = tf.Variable(np_mask.values)

    A = tf.constant(A_df.values)
    shape = A_df.values.shape

    #latent factors : nombre de dimensions latente
    rank = 200

    # Initializing random H and W
    temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.random.randn(shape[0], rank).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H =  tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)

    #cost of Frobenius norm
    cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))

    # Learning rate
    lr = 0.001
    # Number of steps
    steps = 1000
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()

    # Clipping operation. This ensure that W and H learnt are non-negative
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)

    steps = 1000
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            sess.run(train_step)
            sess.run(clip)
            #if i%100==0:
                #print("\nCost: %f" % sess.run(cost))
                #print("*"*40)
        learnt_W = sess.run(W)
        learnt_H = sess.run(H)

    prediction_tr = np.array([pred_func_bias(learnt_W, learnt_H, train_mat_mean,  user_bias, item_bias, u, i,) for (u,i),rating in train_mat.items()])
    prediction_te = np.array([pred_func_bias(learnt_W, learnt_H, train_mat_mean,  user_bias, item_bias, u, i) for u,i,rating in test])

    #on arrondi les ratings predits
    prediction_tr = prediction_tr.round()
    prediction_te = prediction_te.round()

    #on remplace les ratings > 10 par 10, < 1 par 1
    prediction_tr = replace_sup10_inf1(prediction_tr)
    prediction_te = replace_sup10_inf1(prediction_te)

    return prediction_tr, prediction_te

def replace_sup10_inf1(l):
    res = []
    for i in range(len(l)):
        if l[i] > 10.0:
            res.append(10.0)
        elif l[i] < 1.0:
            res.append(1.0)
        else:
            res.append(l[i])
    return res

def predictions_NMF(train_mat,test, nb_comp, num_user, num_item) :
    truth_tr = np.array([rating for (uid,iid),rating in train_mat.items()])
    truth_te = np.array([rating for uid,iid,rating in test])

    A_orig = np.array(train_mat.todense())

    #on remplace les zeros (notes absentes) par des NaN
    for i in range(num_user):
        for j in range(num_item):
            if A_orig[i][j] == 0.0:
                A_orig[i][j] = np.NaN
                
    A_df = pd.DataFrame(A_orig)
    
    np_mask = A_df.notnull()
    np_mask.head()
    
    # Boolean mask for computing cost only on valid (not missing) entries
    tf_mask = tf.Variable(np_mask.values)

    A = tf.constant(A_df.values)
    shape = A_df.values.shape

    #latent factors : nombre de dimensions latente
    rank = nb_comp

    # Initializing random H and W
    temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.random.randn(shape[0], rank).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H =  tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)
    
    #cost of Frobenius norm
    cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))
    
    
    # Learning rate
    lr = 0.001
    # Number of steps
    steps = 1000
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()
    
    
    
    # Clipping operation. This ensure that W and H learnt are non-negative
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)
    
    
    
    steps = 1000
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            sess.run(train_step)
            sess.run(clip)
            #if i%100==0:
                #print("\nCost: %f" % sess.run(cost))
                #print("*"*40)
        learnt_W = sess.run(W)
        learnt_H = sess.run(H)
    
    
    prediction_tr = np.array([pred_func(learnt_W, learnt_H, u, i) for (u,i),rating in train_mat.items()])
    prediction_te = np.array([pred_func(learnt_W, learnt_H, u, i) for u,i,rating in test])
    
    #on arrondi les ratings predits
    prediction_tr = prediction_tr.round()
    prediction_te = prediction_te.round()

    #on remplace les ratings > 10 par 10
    prediction_tr = replace_sup10(prediction_tr)
    prediction_te = replace_sup10(prediction_te)

    # rescale pour les notes au dessus de 10
    # m_te = np.min(prediction_te)
    # M_te = np.max(prediction_te)
    # prediction_te_rescaled = np.array([(x - m_te)/(M_te - m_te) for x in prediction_te])
    
    return prediction_tr, prediction_te




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



def plot_courbes_erreur(n, train_mse, train_mae, test_mse, test_mae, t) :
    plt.plot(n, train_mse, label='train mse', color='darkorange')
    plt.plot(n, train_mae, label='train mae', color='orangered')

    plt.plot(n, test_mse, label='test mse', color='turquoise')
    plt.plot(n, test_mae, label='test mae', color='darkcyan')

    plt.xlabel("nombre de composantes")
    plt.ylabel("erreur")

    plt.title("Erreur en fonction du nombre de composantes - "+str(t))

    plt.legend()
    l = t.lower()
    plt.savefig("img/error/erreur_"+str(l)+".png")
    plt.show()
