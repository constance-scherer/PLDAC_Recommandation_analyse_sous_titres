from utils.ndcg import *
import pickle

path_d_user = "/Users/constancescherer/Desktop/pickles/d_user.p"
path_sim = "/Users/constancescherer/Desktop/pickles/sim.p"
path_most_sim = "/Users/constancescherer/Desktop/pickles/most_sim.p"
path_d_pert_user = "/Users/constancescherer/Desktop/pickles/d_pert_user_k3.p"

#path_d_user = "/Vrac/PLDAC_addic7ed/pickles/d_user.p"
#path_sim = "/Vrac/PLDAC_addic7ed/pickles/sim.p"
#path_most_sim = "/Vrac/PLDAC_addic7ed/pickles/most_sim.p"
#path_d_pert_user = "/Vrac/PLDAC_addic7ed/pickles/d_pert_user_k3.p"


# dictionnaire d_users
# {username : {serie : rating}}
with open(path_d_user, 'rb') as pickle_file:
    d_user = pickle.load(pickle_file)

# matrice des similarités cosinus
with open(path_sim, 'rb') as pickle_file:
    sim = pickle.load(pickle_file)

# dictionnaire des séries les plus similaires
with open(path_most_sim, 'rb') as pickle_file:
    most_similar = pickle.load(pickle_file)
    
with open(path_d_pert_user, 'rb') as pickle_file:
    d_pert_user = pickle.load(pickle_file)




user = 'shannen-l-c'
reco = ['doctor-who', 'house']


DCG = dcg(reco, user, d_pert_user)
print("DCG = ", DCG)

k = 2
ideal_r = ideal_reco(user, k, d_pert_user, d_user)
print("Reco idéale = ", ideal_r)

nDCG = ndcg(reco, user, d_pert_user, d_user)
print("nDCG = ", nDCG)