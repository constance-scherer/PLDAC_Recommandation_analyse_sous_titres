"""Premier exemple avec Tkinter.

On crée une fenêtre simple qui souhaite la bienvenue à l'utilisateur.

"""

# On importe Tkinter
from tkinter import *
from utils.collaborative import *
from utils.similarities import *
import operator
import pickle
from collections import OrderedDict
from utils.predictions_notes import *

print("**************************************************")
print("\n")


# ======== PICKLE LOADS ========
print('----- Start pickle loads -----')

path_d_user = "/Users/constancescherer/Desktop/pickles/d_user.p"
path_sim = "/Users/constancescherer/Desktop/pickles/sim.p"
path_most_sim = "/Users/constancescherer/Desktop/pickles/most_sim.p"

# dictionnaire d_users
# {username : {serie : rating}}
with open(path_d_user, 'rb') as pickle_file:
    d_user = pickle.load(pickle_file)

# matrice des similarités cosinus
with open(path_sim, 'rb') as pickle_file:
    similarities = pickle.load(pickle_file)

# dictionnaire des séries les plus similaires
with open(path_most_sim, 'rb') as pickle_file:
    most_similar = pickle.load(pickle_file)

print('----- Pickle loads finished -----')

# ======== COLLABORATIVE ========
print('----- Start collaborative -----')

path_ratings = "/Users/constancescherer/Desktop/ratings/ratings_imdb/users"
#path_ratings = "/Vrac/PLDAC_addic7ed/ratings/ratings_imdb/users"

liste_series = get_liste_series(d_user)
data = get_data(d_user)
all_data, num_user, num_item = get_all_data(data)
train, train_mat, test = get_train_test(num_user, num_item, all_data, test_size=10)
mean, u_means, i_means,U_ksvd, I_ksvd =  get_Uksvd_Iksvd(train, train_mat, num_user, num_item)
d_username_id, d_itemname_id, Full = create_sparse_mat(data)


path_series = "/Users/constancescherer/Desktop/addic7ed_good_encoding"
#path_series = '/Vrac/PLDAC_addic7ed/addic7ed_clean'

d_info, d_name = getDicts(path_series)
d_ind = reverse_dict(d_name)
d_titre_filename = get_d_titre_filename("titles/title-filename.txt")
d_filename_titre = reverse_dict(d_titre_filename)
d_id_username = reverse_dict(d_username_id)
d_id_serie = reverse_dict(d_itemname_id)

reversed_u_dic, reversed_i_dic = create_reversed_dic(d_username_id, d_itemname_id)

print('----- Collaborative finished -----')

# ======== CONTENT ========
# liste_filenames = []
# for s in liste_series : 
# 	liste_filenames.append(d_titre_filename[s])

#path2 = "/Vrac/PLDAC_addic7ed/addic7ed_clean_ok"
#new_path = "/Vrac/PLDAC_addic7ed/addic7ed_final"
# new_path = "/Users/constancescherer/Desktop/addic7ed_final"
# print("create clean data start")
# createCleanedData(path_series, new_path)
# print("end create clean data")
# print("starting remove")
# removeFilesAndFoldersThatNeedToGo(path_series) 
# print("Remove finished")






class Interface(Frame):
	
	"""Notre fenêtre principale.
	Tous les widgets sont stockés comme attributs de cette fenêtre."""
	
	def __init__(self, fenetre, **kwargs):
		fenetre.title("Recommandation de séries TV")
		fenetre.geometry("800x800")
		Frame.__init__(self, fenetre, **kwargs)

		#self.pack(fill=BOTH)

		self.username1 = ""
		
		self.titre = Label(fenetre, text="Veuillez choisir un utilisateur : ")
		#self.titre.pack(side='top')


		self.liste = Listbox(fenetre)
		#self.liste.pack()

		for u in d_user.keys() :
			self.liste.insert(END, u)
		
		# self.bouton_quitter = Button(self, text="Quitter", command=self.quit)
		# self.bouton_quitter.pack(side="left")


		
		self.bouton_collabo = Button(fenetre, text="filtrage collaboratif",command=self.filtrage_collaboratif)
		#self.bouton_collabo.pack(side='top')


		self.bouton_content = Button(fenetre, text="content",command=self.content)
		#self.bouton_content.pack(side='top')

		self.message = ""
		self.liste2 = ""
		self.liste3 = ""

		self.message2 = ""



		self.titre.grid(row=1, column=2)
		self.liste.grid(row=2, column=2)
		self.bouton_collabo.grid(row=4, column=3)
		self.bouton_content.grid(row=4, column=1)

		self.best_show = ""


	
	
	def filtrage_collaboratif(self):
		"""Il y a eu un clic sur le bouton.
		
		On change la valeur du label message."""
		user_selectionne = reversed_u_dic[int(self.liste.curselection()[0])]
		username = reversed_u_dic[int(self.liste.curselection()[0])]
		print('user = ', username)

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
		top3_reco = reco[:3]
		#top3_reco, p = recommandation(user_selectionne, data, d_user, 3, U_ksvd, I_ksvd, u_means, i_means, mean)
		if self.message != "":
			self.message.destroy()
		if self.liste2 != "":
			self.liste2.destroy()

		self.message = Label(fenetre, text="Top 3 des recommandations (filtrage collaboratif)")
		
		self.liste2 = Listbox(fenetre)
		#self.liste2.pack()
		for r in top3_reco :
			self.liste2.insert(END, r)
		self.message.grid(row=5, column=3)
		self.liste2.grid(row=6, column=3)

		

	def content(self):
		user_selectionne = reversed_u_dic[int(self.liste.curselection()[0])]

		top1_user = max(d_user[user_selectionne].items(), key=operator.itemgetter(1))[0]

		show = d_titre_filename[top1_user]
		if self.best_show != "" :
			self.best_show.destroy()
		self.best_show = Label(fenetre, text="best show : "+show)

		top3_reco = most_similar[show]
		if self.message2 != "":
			self.message2.destroy()

		self.message2 = Label(fenetre, text="Top 3 des recommandations (content)")
		self.message2.grid(row=5, column=1)

		if self.liste3 != "":
			self.liste3.destroy()

		self.liste3 = Listbox(fenetre)

		for r in top3_reco :
			if r in d_filename_titre.keys() :
				t = d_filename_titre[r]
			else :
				t = r
			self.liste3.insert(END, t)
		self.message2.grid(row=5, column=1)
		self.liste3.grid(row=6, column=1)
		self.best_show.grid(row=7, column=2)


fenetre = Tk()
interface = Interface(fenetre)

interface.mainloop()
