#!/usr/bin/env python3
import sys, json, pickle, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

# Load artifacts
base = "/mnt/data/project_extracted/Project/output"
knn = pickle.load(open(base + "/knn_model.pkl","rb"))
movies_df = pickle.load(open(base + "/movies_dataframe.pkl","rb"))
scaler = pickle.load(open(base + "/scaler.pkl","rb"))

feature_cols = knn['feature_cols']

def prepare_input(input_json):
    # input_json expected keys: genre (string or list), year (int), average_rating (float)
    genres_input = input_json.get('genre', [])
    if isinstance(genres_input, str):
        genres_input = genres_input.split('|') if genres_input else []
    year = input_json.get('year', movies_df['year'].median())
    avg = input_json.get('average_rating', movies_df['average_rating'].mean())
    # create vector
    vec = []
    for col in feature_cols:
        if col in ['year','average_rating']:
            vec.append([year, avg][['year','average_rating'].index(col)])
        else:
            vec.append(1.0 if col in genres_input else 0.0)
    vec = np.array(vec, dtype=float).reshape(1,-1)
    # scale last two cols
    vec[:,-2:] = scaler.transform(vec[:,-2:])
    return vec

def recommend(input_json, top_n=10):
    vec = prepare_input(input_json)
    distances, indices = knn['nn'].kneighbors(vec, n_neighbors=knn['k']+1)
    # exclude self-match if exists, take top_n unique movies
    idxs = indices[0]
    dists = distances[0]
    res = []
    for i,idx in enumerate(idxs):
        movie = movies_df.iloc[idx]
        res.append({"movieId": int(movie['movieId']), "title": movie['title'], "distance": float(dists[i])})
        if len(res) >= top_n:
            break
    return res

if __name__ == "__main__":
    # read JSON string from argv[1] or stdin
    if len(sys.argv) > 1:
        input_json = json.loads(sys.argv[1])
    else:
        input_json = json.load(sys.stdin)
    recs = recommend(input_json, top_n=10)
    print(json.dumps({"recommendations": recs}, ensure_ascii=False))
