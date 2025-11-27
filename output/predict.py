import sys
import json
import pickle
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# ============== Paths ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model files
knn = pickle.load(open(os.path.join(BASE_DIR, "knn_model.pkl"), "rb"))
movies_df = pickle.load(open(os.path.join(BASE_DIR, "movies_dataframe.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

feature_cols = knn["feature_cols"]   # your saved feature columns


# ============== Prepare Input Vector ==============
def prepare_input(input_json):
    genres_input = input_json.get("genre", "")
    if isinstance(genres_input, str):
        genres_input = genres_input.split("|") if genres_input else []

    year = input_json.get("year", movies_df["year"].median())
    avg = input_json.get("average_rating", movies_df["average_rating"].mean())

    vec = []

    # fill vector based on feature columns
    for col in feature_cols:
        if col in ["year", "average_rating"]:
            vec.append(year if col == "year" else avg)
        else:
            vec.append(1.0 if col in genres_input else 0.0)

    vec = np.array(vec, dtype=float).reshape(1, -1)

    # scale last 2 cols
    vec[:, -2:] = scaler.transform(vec[:, -2:])

    return vec


# ============== Get Recommendations (now 50) ==============
def recommend(input_json, top_n=50):
    vec = prepare_input(input_json)

    distances, indices = knn["nn"].kneighbors(vec, n_neighbors=top_n + 1)

    idxs = indices[0][1: top_n+1]       # skip first (same movie)
    dists = distances[0][1: top_n+1]

    results = []

    for i, idx in enumerate(idxs):
        movie = movies_df.iloc[idx]

        # return full movie info for front-end
        results.append({
            "movieId": int(movie["movieId"]),
            "title": movie["title"],
            "genres": movie["genres"],
            "year": int(movie["year"]),
            "rating": float(movie["average_rating"]),
            "distance": float(dists[i])
        })

    return results


# ============== Entry ==============
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_json = json.loads(sys.argv[1])
    else:
        input_json = json.load(sys.stdin)

    recs = recommend(input_json)
    print(json.dumps({"recommendations": recs}, ensure_ascii=False))
