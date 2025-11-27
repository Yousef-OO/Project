import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =======================================================
# 1) LOAD MOVIES DATA
# =======================================================
# Example CSV columns required:
# movieId | title | genres | year | average_rating
df = pd.read_csv("movies.csv")

# Ensure genres are strings
df["genres"] = df["genres"].fillna("Unknown")

# =======================================================
# 2) ONE-HOT ENCODE GENRES
# =======================================================
# Split genres by |  →  create multiple binary columns
df_genres = df["genres"].str.get_dummies(sep="|")

# =======================================================
# 3) SELECT NUMERIC FEATURES
# =======================================================
numeric_features = df[["year", "average_rating"]]

# Scale numeric features
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_features)
scaled_numeric = pd.DataFrame(scaled_numeric, columns=["year", "average_rating"])

# =======================================================
# 4) BUILD FULL FEATURE MATRIX
# =======================================================
features = pd.concat([df_genres, scaled_numeric], axis=1)

feature_cols = list(features.columns)

X = features.values   # final training matrix

# =======================================================
# 5) TRAIN KNN MODEL
# =======================================================
k = 50   # choose top 50 similar movies (adjust as needed)

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(X)

# We pack model + metadata in one dictionary
model_package = {
    "nn": knn,
    "k": k,
    "feature_cols": feature_cols
}

# =======================================================
# 6) SAVE OUTPUT FILES
# =======================================================
with open("output/knn_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

with open("output/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("output/movies_dataframe.pkl", "wb") as f:
    pickle.dump(df, f)

print("🎉 Training complete!")
print("Saved:")
print("- output/knn_model.pkl")
print("- output/scaler.pkl")
print("- output/movies_dataframe.pkl")
