"""Premier exemple avec Tkinter.

On crée une fenêtre simple qui souhaite la bienvenue à l'utilisateur.

"""

# On importe Tkinter
from tkinter import *
from collaborative import *
from similarities import *
import operator


# ======== COLLABORATIVE ========

#path = "/Users/constancescherer/Desktop/ratings/ratings_60"
path = "/Vrac/PLDAC_addic7ed/ratings/ratings_60"
d_user = get_d_user(path)
liste_series = get_liste_series(d_user)
data = get_data(d_user)
all_data, num_user, num_item = get_all_data(data)
train, train_mat, test = get_train_test(num_user, num_item, all_data, test_size=10)
mean, u_means, i_means,U_ksvd, I_ksvd =  get_Uksvd_Iksvd(train, train_mat, num_user, num_item)
d_username_id, d_itemname_id, Full = create_sparse_mat(data)

reversed_u_dic, reversed_i_dic = create_reversed_dic(d_username_id, d_itemname_id)



# ======== CONTENT ========
d_titre_filename = dict()
with open("title-filename.txt") as file :
	lignes = file.readlines()
	for ligne in lignes :
		l = ligne.split(" ")
		titre = l[0]
		filename = l[1].rstrip('\n')
		d_titre_filename[titre] = filename



liste_filenames = []
for s in liste_series : 
	liste_filenames.append(d_titre_filename[s])

#path2 = "/Users/constancescherer/Desktop/addic7ed_clean"
path2 = "/Vrac/PLDAC_addic7ed/addic7ed_clean_ok"
new_path = "/Vrac/PLDAC_addic7ed/addic7ed_final"
print("create clean data start")
createCleanedData(path2, new_path)
print("end create clean data")
#print("starting remove")
#removeFilesAndFoldersThatNeedToGo(path2) 
#print("Remove finished")


# recommandation(username1, data, d_user, nb_pred, U_ksvd, I_ksvd)
print("start similarite")
similarities = similarities(new_path)
print("end similarites")
print("start most similar")
most_similar = most_similar(new_path, similarities)
print("end most similar")

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

		top3_reco, p = recommandation(user_selectionne, data, d_user, 3, U_ksvd, I_ksvd, u_means, i_means, mean)
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
			self.liste3.insert(END, r)
		self.message2.grid(row=5, column=1)
		self.liste3.grid(row=6, column=1)
		self.best_shoW.grid(row=7, column=4)


fenetre = Tk()
interface = Interface(fenetre)

interface.mainloop()
#interface.destroy()


# On crée une fenêtre, racine de notre interface
# fenetre = Tk()
# fenetre.title("Recommandation de séries TV")
# fenetre.geometry("800x800")

# titre = Label(fenetre, text="Veuillez choisir un utilisateur : ")
# titre.pack(side='top')

# liste = Listbox(fenetre)
# liste.pack()

# for u in d_user.keys() :
# 	liste.insert(END, u)


# user_selectionne = reversed_u_dic[int(liste.curselection())[0]]

# top3_reco, p = recommandation(user_selectionne, data, d_user, 3, U_ksvd, I_ksvd)

# liste2 = Listbox(fenetre)
# liste2.pack()
# for r in top3_reco :
# 	liste2.insert(END, r)


# # On démarre la boucle Tkinter qui s'interompt quand on ferme la fenêtre
# fenetre.mainloop()


# On crée un label (ligne de texte) souhaitant la bienvenue
# Note : le premier paramètre passé au constructeur de Label est notre
# interface racine
#champ_label = Label(fenetre, text="Recommandation de séries TV")

# On affiche le label dans la fenêtre
#champ_label.pack()


# var_texte = StringVar()
# ligne_texte = Entry(fenetre, textvariable=var_texte, width=30)
# ligne_texte.pack()


# var_case = IntVar()
# case = Checkbutton(fenetre, text="Ne plus poser cette question", variable=var_case)
# case.pack()



# bouton_quitter = Button(fenetre, text="Quitter", command=fenetre.quit)
# bouton_quitter.pack()

