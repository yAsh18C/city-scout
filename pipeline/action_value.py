"""
Action Value Model: logistic regression estimating P(shot within N events)
for each on-ball action, given its location and type.

Training: label each event 1 if any of the next MAX_HORIZON events in the
same match is a Shot, else 0. Train logistic regression. Validate on
held-out matches.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import os

MAX_HORIZON = 5   # events to look ahead for a shot
MODEL_DIR = "models"


def _build_training_data(df):
    """
    For each event, label 1 if a shot occurs within MAX_HORIZON events
    in the same match.
    """
    df = df.sort_values(["match_id", "index"] if "index" in df.columns
                        else ["match_id", "time_seconds"]).copy()

    # Create a binary shot flag
    df["is_shot"] = (df["type"] == "Shot").astype(int)

    # For each event, check if the next MAX_HORIZON events in the same
    # match contain a shot (rolling forward sum)
    labels = []
    for match_id, group in df.groupby("match_id"):
        shot_arr = group["is_shot"].values
        n = len(shot_arr)
        label_arr = np.zeros(n, dtype=int)
        for i in range(n):
            end = min(i + MAX_HORIZON + 1, n)
            if shot_arr[i + 1:end].sum() > 0:
                label_arr[i] = 1
        labels.extend(label_arr)

    df["label"] = labels
    return df


def _make_features(df):
    """
    Feature vector for action value model:
      x_norm, y_norm, dist_to_goal, angle_to_goal, in_att_third,
      is_pass, is_carry, is_dribble, is_shot, is_pressure,
      time_pct (0=kickoff, 1=90th min)
    """
    # Normalize coordinates
    x_norm = df["x"] / 120.0
    y_norm = df["y"] / 80.0

    # Distance and angle to goal (goal at x=120, y=40)
    dx = 120.0 - df["x"]
    dy = df["y"] - 40.0
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    type_col = df["type"]
    feats = pd.DataFrame({
        "x_norm":      x_norm.values,
        "y_norm":      y_norm.values,
        "dist_goal":   dist.values / 120.0,
        "angle_goal":  np.abs(angle.values) / np.pi,
        "in_att_third": (df["x"] > 80).astype(int).values,
        "is_pass":     (type_col == "Pass").astype(int).values,
        "is_carry":    (type_col == "Carry").astype(int).values,
        "is_dribble":  (type_col == "Dribble").astype(int).values,
        "is_shot":     (type_col == "Shot").astype(int).values,
        "is_pressure": (type_col == "Pressure").astype(int).values,
        "time_pct":    (df["minute"].clip(0, 90) / 90.0).values,
    })
    return feats


def train_action_value_model(df, test_frac=0.2):
    """
    Train logistic regression action value model.
    Returns (model, scaler, roc_auc).
    """
    print("Building training data for action value model...")
    df = _build_training_data(df)

    # Train/test split by match
    matches = df["match_id"].unique()
    np.random.seed(42)
    test_matches = np.random.choice(
        matches, size=int(len(matches) * test_frac), replace=False
    )
    train_df = df[~df["match_id"].isin(test_matches)]
    test_df  = df[df["match_id"].isin(test_matches)]

    X_train = _make_features(train_df).values
    y_train = train_df["label"].values
    X_test  = _make_features(test_df).values
    y_test  = test_df["label"].values

    print(f"  Train: {len(X_train):,} events  |  Test: {len(X_test):,} events")
    print(f"  Positive rate (train): {y_train.mean():.3f}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    model.fit(X_train_s, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    print(f"  Action value model ROC-AUC (test): {auc:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "action_value_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "action_value_scaler.joblib"))

    return model, scaler, auc


def score_player_events(df, model, scaler):
    """
    Score every event in df, return mean action value per player.
    """
    X = _make_features(df).values
    X_s = scaler.transform(X)
    df = df.copy()
    df["action_value"] = model.predict_proba(X_s)[:, 1]

    return df.groupby("player")["action_value"].mean().rename("avg_action_value")


def load_model():
    """Load saved model and scaler."""
    model  = joblib.load(os.path.join(MODEL_DIR, "action_value_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "action_value_scaler.joblib"))
    return model, scaler
