import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("movies_data.csv", index_col = [0])
crew = pd.read_csv("Crew.csv", index_col = [0])
genres = pd.read_csv("genres.csv", index_col = [0])
paises = pd.read_csv("production_countries.csv")
saga = pd.read_csv("saga.csv")
mask = (crew["job"] == "Director")
crew = crew[mask]

'''Se eliminan los valores que estén duplicados en todas sus columnas, al inspeccionar la máscara 'p' aplicada al df
se ve que las pelóculas que comparten ID son las mismas, por lo que se pueden eliminar'''
p = df.duplicated("id_movie")
df = df.drop(df[p].index)

df = df.dropna(subset=["title"])


#Vamos filtrar las películas ya que buscamos que no se recomiende una película con baja calificación
#Es decir vamos a seleccionar películas con puntuación por encima de la media

df_recomendado = df.loc[lambda df:(df["vote_average"] > 6)]
df_recomendado["release_year"] = df_recomendado["release_year"].to_string()
df_recomendado = df_recomendado[["id_movie","overview","title","vote_average","vote_count","release_year","genres"]]
df_recomendado["overview"] = df_recomendado["overview"].str.lower()

df_recomendado = pd.merge(df_recomendado, crew[["name","id_movie"]], on = "id_movie")
df_recomendado["r"] = df_recomendado["genres"] + df_recomendado["overview"] + df_recomendado["name"] 
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_recomendado["r"])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recomendacion(titulo, cosine_sim = cosine_sim):
    titulo = str.lower(titulo)
    idx = df_recomendado[df_recomendado["title"].str.lower() == titulo].index.to_list()[0]
    puntaje_coseno = enumerate(cosine_sim[idx])
    puntaje_coseno = sorted(puntaje_coseno, key = lambda x: x[1], reverse = True)    
    puntaje_coseno = puntaje_coseno[1:5]
    cos_indices = [i[0] for i in puntaje_coseno]
    print (df_recomendado["title"].iloc[cos_indices])

