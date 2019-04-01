import os
import re
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import dok_matrix
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt



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



def pred_func_ksvd(U_ksvd, I_ksvd, uid,iid, mean, u_means, i_means):
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
    
    return prediction_tr, prediction_te



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
