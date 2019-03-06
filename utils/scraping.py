#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from multidl_mieux import *
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver      
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options


def get_headers(url):
	"""
	url : string, url de la page dont on veut les headers
	renvoie les headers
	"""
	return {'Referer' : url, 'User-Agent' : 'Mozilla/5.0 (X11; U; Linux x86_64; en-US; rv:1.9.1.3)'}

def get_headers2(url):
	"""
	url : string, url de la page dont on veut les headers
	renvoie les headers qui fonctionnent pour Metacritic
	"""
	user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
	return {'Referer' : url, 'User-Agent' : user_agent}


def get_request(url): 
	"""
	"""
	return urllib.request.Request(url, None, get_headers2(url))

def my_cb(page,code,url,**kwargs): 
	"""
	"""
	print("%s\n%s\n param sup : %s" % (url,page,str(kwargs)))

def my_cb2(page,code,url,**kwargs):
	"""
	"""
	print("")


def get_page_code(url, dm) :
	"""
	url : string, l'url de la page que l'on veut
	dm : MultiDLManager
	renvoie la page (en bytes) associée à l'url en entrée et le code 
	"""
	dm.add_task(get_request(url),my_cb2)
	page, code = dm.get_request(url)
	return page, code


def get_html_soup(page) :
	"""
	page : bytes, code html de la page que l'on veut
	renvoie la page sous une forme exploitable avec BeautifulSoup
	"""
	# page est en bytes (string avec un 'b' devant)
	response = str(page, 'utf-8') # pour convertir en string classique
	html_soup = BeautifulSoup(response, 'html.parser')
	return html_soup



def get_url_metacritic(nom_serie, nb_saison, nb_page) :
	"""
	nom_serie : string, nom de la série 
	nb_saison : int, saison de la série
	nb_page : int, numéro de la page de review que l'on veut
	renvoie l'url de la page correspondant à la série, la saison et le numero de page correspondant sur Metacritic
	"""
	url = "https://www.metacritic.com/tv/"+nom_serie+"/season-"+str(nb_saison)+"/user-reviews?page="+str(nb_page)
	return url


def get_reviews_from_metacritic(nom_serie, dm,  nb_total_saisons) :
	""" 
	nom_serie : string, nom de la serie
	dm : MultiDLManager
	nb_total_saisons : int, nb totals de saisons de cette série
	renvoie un dictionnaire {username : {serie : list(note)}} pour une seule série
	"""

	d = dict()
	for nb_saison in range(1, nb_total_saisons+1) : # pour chaque saison 
	    nb_page = 0
	    # Sur Metacritic, quand on essaye d'accéder à un numéro de page de reviews qui n'existent pas, on n'a pas une erreur 404 
	    # mais simplement une page sans reviews
	    while True : # on va regarder les pages jusqu'à ce qu'il n'y ait plus de reviews
	        # print("Saison "+str(nb_saison)+" page "+str(nb_page))
	        url = get_url_metacritic(nom_serie, nb_saison,nb_page) 

	        page, code = get_page_code(url, dm)
	        html_soup = get_html_soup(page)

	        reviews = html_soup.find_all('div', class_ = 'review_stats') # on récupère toutes les div de la classe reviews_stats (c'est là dedans qu'on trouve les notes+username)
	        
	        nb_reviews = len(reviews) - 3 # il y a juste quelques div en plus avec ce nom (et pas de classes plus fines)
	        
	        if nb_reviews <= 0 : # s'il n'y a aucune reviews, c'est-à-dire qu'on est sur une page sans reviews, donc à la fin des reviews
	            #print('\tPlus aucune review pour '+nom_serie+" saison "+str(nb_saison)+" page "+str(nb_page))
	            break # on sort

	        for i in range(0, nb_reviews) :
	            name = reviews[i].find('div', class_='name')
	            user = list(name.children)[1].text # nom de l'utilisateur
	            r = reviews[i].find('div', class_="review_grade")
	            rating = int(r.div.text) # sa note pour la saison 

	            if nb_saison == 1 or user not in d.keys(): # si c'est la première saison ou que l'utilisateur n'a jamais été vu, alors on crée la liste de ces notes
	                d[user] = {nom_serie : [rating]}
	            else  :
	                d[user][nom_serie].append(rating)
	                
	        nb_page += 1

	return d


def get_reviews_from_imdb(nom_serie, id_serie) :
    """
    nom_serie : string, nom de la série
    id_serie : string, id imdb de la série
    renvoie un dictionnaire {username : {serie : note}}
    """
    options = Options()
    options.add_argument('--headless')
    
    url = "https://www.imdb.com/title/"+id_serie+"/reviews?ref_=tt_ov_rt"
    
    # Tor needs to be running
    profile=webdriver.FirefoxProfile()
    profile.set_preference('network.proxy.type', 1)
    profile.set_preference('network.proxy.socks', '127.0.0.1')
    profile.set_preference('network.proxy.socks_port', 9050)

    driver = webdriver.Firefox(profile, executable_path='/Users/constancescherer/Downloads/geckodriver', options=options)
    driver.get(url)

    while True:
        try:
        
            loadMoreButton = driver.find_element_by_id("load-more-trigger")

            driver.execute_script("arguments[0].scrollIntoView();", loadMoreButton)
            
            time.sleep(2)
            loadMoreButton.click()
            time.sleep(2)
            
        except Exception as e:
            break

    html = driver.page_source
    
    driver.quit()
    
    html_soup = BeautifulSoup(html, 'html.parser')
    
    reviews = html_soup.find_all('div', class_="lister-item-content")
    nb_reviews = len(reviews)
    
    d = dict()
    for i in range(0, nb_reviews) :
        
        try : # on essaye de récupérer le rating
            rating = int(reviews[i].find('div', class_="ipl-ratings-bar").span.span.text)
        except Exception as e: # s'il existe pas on passe à la review suivante
            continue
        #rating = int(reviews[i].find('div', class_="ipl-ratings-bar").span.span.text)
        name = list(reviews[i].find('span', class_="display-name-link").children)[0].text
        
        d[name] = {nom_serie : rating}
        
    return d





