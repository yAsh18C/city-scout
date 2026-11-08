"""
Player similarity: UMAP dimensionality reduction + nearest-neighbour lookup.
Includes Man City specific scoring.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import umap
import joblib
import os

MODEL_DIR = "models"
MAN_CITY_TEAM = "Manchester City"


def build_umap_embedding(feat_df, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Fit UMAP on feature matrix. Returns (embedding_array, reducer, scaler).
    """
    print("Fitting UMAP...")
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X = imputer.fit_transform(feat_df.values)
    X_s = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric="euclidean",
    )
    embedding = reducer.fit_transform(X_s)
    print(f"  UMAP shape: {embedding.shape}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(reducer,  os.path.join(MODEL_DIR, "umap_reducer.joblib"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "umap_scaler.joblib"))
    joblib.dump(imputer,  os.path.join(MODEL_DIR, "umap_imputer.joblib"))

    return embedding, reducer, scaler, imputer


def compute_city_fit(embedding, meta_df):
    """
    City Fit score: cosine similarity of each player's UMAP embedding to
    the centroid of Man City players' embeddings.
    Returns a Series indexed by player name (0-100 score).
    """
    city_mask = meta_df["team"] == MAN_CITY_TEAM
    city_players = meta_df[city_mask].index.tolist()

    if not city_players:
        print("  Warning: No Man City players found in dataset")
        return pd.Series(0.0, index=meta_df.index)

    city_indices = [meta_df.index.get_loc(p) for p in city_players if p in meta_df.index]
    city_centroid = embedding[city_indices].mean(axis=0)

    # Cosine similarity to centroid
    def cosine_sim(vec, centroid):
        num = np.dot(vec, centroid)
        denom = np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-9
        return num / denom

    sims = np.array([cosine_sim(embedding[i], city_centroid)
                     for i in range(len(embedding))])

    # Scale to 0-100
    sims_min, sims_max = sims.min(), sims.max()
    scores = 100 * (sims - sims_min) / (sims_max - sims_min + 1e-9)

    return pd.Series(scores, index=meta_df.index, name="city_fit")


def find_similar_players(player_name, embedding, meta_df, top_n=8):
    """
    Find top_n most similar players to player_name using Euclidean distance
    in UMAP embedding space.
    Returns DataFrame with columns: player, team, competition, umap_distance.
    """
    if player_name not in meta_df.index:
        return pd.DataFrame()

    idx = meta_df.index.get_loc(player_name)
    player_vec = embedding[idx]

    dists = np.linalg.norm(embedding - player_vec, axis=1)

    result = meta_df.copy()
    result["umap_distance"] = dists
    result = result[result.index != player_name]  # exclude self
    result = result.sort_values("umap_distance").head(top_n)
    result["similarity_pct"] = 100 * (
        1 - result["umap_distance"] / (result["umap_distance"].max() + 1e-9)
    )
    # Index is the player name; drop the redundant 'player' data column if present
    result = result.reset_index()
    result.rename(columns={"index": "player_key"}, inplace=True)
    if "player" not in result.columns:
        result["player"] = result["player_key"]
    return result
