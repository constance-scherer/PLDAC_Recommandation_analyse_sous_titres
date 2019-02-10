import os
import re

def verifier_ligne(ligne):
    """return True si la ligne est un sous-titre, False sinon"""
    timestamp_regex = r'[0-9]{2}:[0-9]{2}:[0-9]{2}' 
    subnumber_regex =r'^[0-9]+$'
    
    liste_regex = [timestamp_regex, subnumber_regex]

    l = ligne.lower()
    if "addic7ed" in l:
        return False
    #if l.startswith("sync"):
        #return False
    for regex in liste_regex:
        if re.match(regex, ligne):
            return False
    return True

def transformer_ligne(ligne):
    """str -> str
    effectue transformation souhaitees sur la ligne"""
    opening_tag_regex =  r'^<[a-zA-Z]+>' #to get rid of opening tag
    closing_tag_regex = r'</[a-zA-Z]+>$' #to get rif of closing tag
    new_line = re.sub(opening_tag_regex, '', ligne)
    new_line = re.sub(closing_tag_regex, '', ligne)
    return new_line

def scan_folder(parent_folder, corp):
    """retourne corpus des textes contenus dans parent_folder sous forme de liste de string"""
    
    # iterate over all the files in directory 'parent_folder'
    for file_name in os.listdir(parent_folder):
        if file_name.endswith(".txt"):
            path = parent_folder+"/"+file_name
            fichier = open(path, "r")
            lignes = fichier.readlines()
            fichier.close()
            
            texte = ""
            for ligne in lignes :
                if verifier_ligne(ligne):
                    new_line = transformer_ligne(ligne)
                    texte += new_line
            corp.append(texte)
        
        else:
            current_path = "".join((parent_folder, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, corp)
    
    return corp


#pour eviter les variables globales
def get_corpus(parent_folder):
    c = []
    res = scan_folder(parent_folder, c)
    return res