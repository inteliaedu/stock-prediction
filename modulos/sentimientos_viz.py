from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def bar_plot(df):
    df[['neu','pos','neg']].sum().plot(kind='bar')
    plt.xticks(np.arange(3), ['Neutral', 'Positivo', 'Negativo'], rotation=0)
    plt.show();

def casos_plot(df,config):
    df["Positivos_texto"] =np.where(df.Positivos==1,"Positivo","Negativo/ Neutral")
    fig = sns.countplot(x=df["Positivos_texto"]).set(title='Casos de '+config['alpha_vantage']['symbol'])
    plt.xlabel("Casos")
    plt.ylabel("Frecuencia");

def word_cloud(df):
    text = " ".join(i for i in df.Titulares)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.figure( figsize=(15,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show();