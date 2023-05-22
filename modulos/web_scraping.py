# Librerías
import json # Para lectura de archivo de configuración

import time #para sleep
import pandas as pd
from datetime import datetime


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import requests
from bs4 import BeautifulSoup

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Modelo pre entrenado de análisis de sentimientos.

def extraer_noticias(config):
    s=Service(r"C:\Users\Hp\Documents\GitHub\ALM-DE-DATOS\SELENIUM\chromedriver_win32\chromedriver.exe")
    driver = webdriver.Chrome(service=s)
    url=config["sentimientos"]["url"]
    driver.get(url)
    time.sleep(5) 

    botonAcceptAll = driver.find_element(By.CLASS_NAME,("btn.secondary.accept-all")) 
    botonAcceptAll.click()

    time.sleep(5)

    buscar = driver.find_element(By.ID,"yfin-usr-qry")
    buscar.send_keys(config["alpha_vantage"]["symbol"])

    time.sleep(5) 

    elegirEmpresa = driver.find_element(By.XPATH,'//*[@id="header-search-form"]/div[2]/div[1]/div/ul[1]/li[1]')
    elegirEmpresa.click()

    time.sleep(5) 

    elegirNews = driver.find_element(By.XPATH,'//*[@id="Col1-3-Summary-Proxy"]/section/div/div/div[2]')
    elegirNews.click()

    html = driver.page_source

    soup = BeautifulSoup(html) #Conexión entre Selenium y BeautifulSoup

    noticias = soup.find_all("h3", class_="Mb(5px)")

    titulares = []
    for noticia in noticias:
        titulares.append(noticia.get_text())
    return titulares

def classify_positive(text, threshold=0):
    # Score text
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    # Get compound score from dictionary
    score = score.get('compound')
    # Classify text according to threshold
    if score > threshold:
        pred_class = 1
    else:
        pred_class = 0
    # Return prediciton
    return pred_class

def predicciones_sentimientos(titulares,threshold_value):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    polarity= []
    for headline in titulares:
        sentiment = classify_positive(headline, threshold_value)
        sentiments.append(sentiment)
        polarity.append(analyzer.polarity_scores(headline))
    df = pd.DataFrame(list(map(list, zip(titulares,sentiments,polarity))),columns=['Titulares','Positivos', 'Detalles'])
    df2 = pd.json_normalize(df.Detalles)
    df_result = pd.concat([df, df2], axis=1, join='inner').drop('Detalles',axis=1)
    df_result['Threshold'] =  threshold_value
    df_result['Timestamp'] =  datetime.now()
    return df_result