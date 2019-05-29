import numpy as np

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

def error_ksvd(train_mat, test) :
    """returns train and test error for ksvd using MSE and MAE metrics"""
    truth_tr = np.array([rating for (uid,iid),rating in train_mat.items()])
    truth_te = np.array([rating for uid,iid,rating in test])

    prediction_tr = np.array([pred_func_ksvd(u,i, U_ksvd, I_ksvd, u_means, i_means, mean) for (u,i),rating in train_mat.items()])
    prediction_te = np.array([pred_func_ksvd(u,i, U_ksvd, I_ksvd, u_means, i_means, mean) for u,i,rating in test])


    print("Training Error:")
    print("MSE:",  MSE_err(prediction_tr,truth_tr))
    print("MAE:",  MAE_err(prediction_tr,truth_tr))

    print("Test Error:")
    print("MSE:",  MSE_err(prediction_te,truth_te))
    print("MAE:",  MAE_err(prediction_te,truth_te))

def error_NMF(train_mat, test, num_user, num_item) :
    """returns train and test error for NMF using MSE and MAE metrics"""
    truth_tr = np.array([rating for (uid,iid),rating in train_mat.items()])
    truth_te = np.array([rating for uid,iid,rating in test])
    
    prediction_tr, prediction_te = predictions_NMF(train_mat,test, 100, num_user, num_item)
    print("Training Error:")
    print("MSE:",  MSE_err(prediction_tr,truth_tr))
    print("MAE:",  MAE_err(prediction_tr,truth_tr))

    print("Test Error:")
    print("MSE:",  MSE_err(prediction_te,truth_te))
    print("MAE:",  MAE_err(prediction_te,truth_te))

def error_NMF_biais(train_mat,test, nb_comp, num_user, num_item) :
    """returns train and test error for NMF with user/item biases using MSE and MAE metrics"""
    truth_tr = np.array([rating for (uid,iid),rating in train_mat.items()])
    truth_te = np.array([rating for uid,iid,rating in test])
    
    prediction_tr, prediction_te = predictions_NMF_biais(train_mat,test, 100, num_user, num_item)
    print("Training Error:")
    print("MSE:",  MSE_err(prediction_tr,truth_tr))
    print("MAE:",  MAE_err(prediction_tr,truth_tr))

    print("Test Error:")
    print("MSE:",  MSE_err(prediction_te,truth_te))
    print("MAE:",  MAE_err(prediction_te,truth_te))

def error_content(train_mat, test, d_name, d_user, d_ind, d_titre_filename, d_filename_titre, d_id_username, d_id_serie, similarities) :
    """returns train and test error for content based recommandation with user/item biases using MSE and MAE metrics"""
    truth_tr = np.array([rating for (uid,iid),rating in train_mat.items()])
    truth_te = np.array([rating for uid,iid,rating in test])

    prediction_tr = np.array([pred_content(u, i, d_name, d_user, d_ind, d_titre_filename, d_filename_titre, d_id_username, d_id_serie, similarities) for (u,i),rating in train_mat.items()])
    prediction_te = np.array([pred_content(u, i, d_name, d_user, d_ind, d_titre_filename, d_filename_titre, d_id_username, d_id_serie, similarities) for u,i,rating in test])


    print("Training Error:")
    print("MSE:",  MSE_err(prediction_tr,truth_tr))
    print("MAE:",  MAE_err(prediction_tr,truth_tr))

    print("Test Error:")
    print("MSE:",  MSE_err(prediction_te,truth_te))
    print("MAE:",  MAE_err(prediction_te,truth_te))

def error_mean_only(train_mat, test) :
    """returns train and test error for mean only model with user/item biases using MSE and MAE metrics"""
    train_mat_mean = np.array(list(train_mat.values())).mean()
    
    truth_tr = np.array([rating for (uid,iid),rating in train_mat.items()])
    truth_te = np.array([rating for uid,iid,rating in test])
    
    prediction_tr = np.array([train_mat_mean for (u,i),rating in train_mat.items()])
    prediction_te = np.array([train_mat_mean for u,i,rating in test])


    print("Training Error:")
    print("MSE:",  MSE_err(prediction_tr,truth_tr))
    print("MAE:",  MAE_err(prediction_tr,truth_tr))

    print("Test Error:")
    print("MSE:",  MSE_err(prediction_te,truth_te))
    print("MAE:",  MAE_err(prediction_te,truth_te))
