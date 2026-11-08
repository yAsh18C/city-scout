"""
precompute.py — Run once to generate all data files for the Streamlit app.

Output files (all in data/):
  player_features.parquet    — 25-feature matrix, one row per player
  player_meta.parquet        — name, team, competition, minutes, city_fit
  player_heatmap.parquet     — (player, x, y, minute) for heat map rendering
  umap_embeddings.npy        — 2D UMAP coordinates, row-ordered with player_meta
  model_metrics.json         — action value model ROC-AUC for display in app

Models (in models/):
  action_value_model.joblib
  action_value_scaler.joblib
  umap_reducer.joblib
  umap_scaler.joblib
  umap_imputer.joblib
"""

import os, json
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pipeline.load_data import load_all_events
from pipeline.features import build_player_features
from pipeline.action_value import train_action_value_model, score_player_events
from pipeline.similarity import build_umap_embedding, compute_city_fit

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. Load events ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading StatsBomb events")
print("=" * 60)
df = load_all_events()
print(f"On-ball events: {len(df):,}")

# ── 2. Train action value model ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Action value model")
print("=" * 60)
model, scaler, auc = train_action_value_model(df)
avg_av = score_player_events(df, model, scaler)

with open(os.path.join(DATA_DIR, "model_metrics.json"), "w") as f:
    json.dump({"action_value_roc_auc": round(auc, 4)}, f)

# ── 3. Feature engineering ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Feature engineering")
print("=" * 60)
feat_df, meta_df = build_player_features(df)

# Attach action value to features
feat_df["avg_action_value"] = avg_av.reindex(feat_df.index).fillna(avg_av.median())

print(f"Player feature matrix: {feat_df.shape}")

# ── 4. UMAP + City Fit ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: UMAP embedding + City Fit scoring")
print("=" * 60)
embedding, reducer, umap_scaler, imputer = build_umap_embedding(feat_df)

city_fit = compute_city_fit(embedding, meta_df)
meta_df["city_fit"] = city_fit

print(f"Man City players in dataset: {(meta_df['team'] == 'Manchester City').sum()}")
print(f"City Fit range: {city_fit.min():.1f} - {city_fit.max():.1f}")

# ── 5. Heat map data ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Saving heat map event data")
print("=" * 60)
eligible = set(feat_df.index.tolist())
heatmap_df = df[df["player"].isin(eligible)][["player", "x", "y", "minute"]].copy()
print(f"Heat map events: {len(heatmap_df):,}")

# ── 6. Save everything ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving data files")
print("=" * 60)

feat_df.to_parquet(os.path.join(DATA_DIR, "player_features.parquet"))
meta_df.to_parquet(os.path.join(DATA_DIR, "player_meta.parquet"))
heatmap_df.to_parquet(os.path.join(DATA_DIR, "player_heatmap.parquet"))
np.save(os.path.join(DATA_DIR, "umap_embeddings.npy"), embedding)

print("Saved:")
print(f"  player_features.parquet  — {feat_df.shape}")
print(f"  player_meta.parquet      — {meta_df.shape}")
print(f"  player_heatmap.parquet   — {heatmap_df.shape}")
print(f"  umap_embeddings.npy      — {embedding.shape}")
print(f"  model_metrics.json       — ROC-AUC {auc:.4f}")
print("\nPrecompute complete.")
