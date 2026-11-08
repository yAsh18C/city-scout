"""
Feature engineering: build one feature vector per player.

Features (25 total):
  Zone activity (6): % of actions in each of 6 pitch zones
  Temporal (6):      % of actions in each 15-minute interval
  Action mix (5):    % of each action type (pass, carry, shot, pressure, other)
  Efficiency (4):    pass success rate, dribble success rate, shot-on-target rate,
                     pressure + interception rate (defensive activity)
  Volume (4):        total actions per 90, progressive actions per 90,
                     attacking-third actions per 90, final-third shot rate
"""

import numpy as np
import pandas as pd

# Pitch dimensions: 120 x 80 yards (StatsBomb)
PITCH_X = 120.0
PITCH_Y = 80.0

# 6-zone grid: 2 halves × 3 lateral bands
#   Zones: Left-Def, Centre-Def, Right-Def, Left-Att, Centre-Att, Right-Att
ZONE_NAMES = ["left_def", "centre_def", "right_def", "left_att", "centre_att", "right_att"]

TEMPORAL_WINDOWS = [
    (0, 15, "t0_15"),
    (15, 30, "t15_30"),
    (30, 45, "t30_45"),
    (45, 60, "t45_60"),
    (60, 75, "t60_75"),
    (75, 999, "t75_90"),
]

MIN_MINUTES = 450  # Minimum playing time for inclusion


def assign_zone(x, y):
    """Return zone index (0-5) for a pitch coordinate."""
    # Halves
    half = 0 if x < PITCH_X / 2 else 1
    # Lateral thirds
    if y < PITCH_Y / 3:
        lat = 0  # left
    elif y < 2 * PITCH_Y / 3:
        lat = 1  # centre
    else:
        lat = 2  # right
    return half * 3 + lat


def compute_minutes_played(df, player_name):
    """
    Approximate minutes: use the max time_seconds across matches for the player,
    capped at 90 minutes per match.
    """
    player_events = df[df["player"] == player_name]
    matches = player_events["match_id"].unique()
    total = 0
    for mid in matches:
        match_events = player_events[player_events["match_id"] == mid]
        max_min = match_events["minute"].max()
        total += min(max_min, 90)
    return total


def build_player_features(df):
    """
    Build feature matrix: one row per player.
    Returns feature DataFrame + metadata DataFrame.
    """
    # Only keep players with valid names
    df = df[df["player"].notna()].copy()

    # Assign zones and time windows
    df["zone"] = df.apply(lambda r: assign_zone(r["x"], r["y"]), axis=1)
    df["time_window"] = pd.cut(
        df["minute"],
        bins=[0, 15, 30, 45, 60, 75, 999],
        labels=[tw[2] for tw in TEMPORAL_WINDOWS],
        right=True,
    )

    # Determine player minutes (approximate)
    print("Computing player minutes...")
    player_minutes = {}
    for player in df["player"].unique():
        player_minutes[player] = compute_minutes_played(df, player)

    eligible_players = [p for p, m in player_minutes.items() if m >= MIN_MINUTES]
    print(f"Players with >= {MIN_MINUTES} minutes: {len(eligible_players)}")

    df = df[df["player"].isin(eligible_players)].copy()

    rows = []
    meta_rows = []

    for player in eligible_players:
        pdf = df[df["player"] == player]
        minutes = player_minutes[player]
        per90 = 90.0 / minutes

        # --- Zone features ---
        zone_counts = pdf["zone"].value_counts()
        total_actions = len(pdf)
        zone_feats = {f"pct_{z}": zone_counts.get(i, 0) / max(total_actions, 1)
                      for i, z in enumerate(ZONE_NAMES)}

        # --- Temporal features ---
        time_counts = pdf["time_window"].value_counts()
        time_feats = {f"pct_{tw[2]}": time_counts.get(tw[2], 0) / max(total_actions, 1)
                      for tw in TEMPORAL_WINDOWS}

        # --- Action type mix ---
        type_counts = pdf["type"].value_counts()
        action_feats = {
            "pct_pass":     type_counts.get("Pass", 0) / max(total_actions, 1),
            "pct_carry":    type_counts.get("Carry", 0) / max(total_actions, 1),
            "pct_shot":     type_counts.get("Shot", 0) / max(total_actions, 1),
            "pct_pressure": type_counts.get("Pressure", 0) / max(total_actions, 1),
            "pct_dribble":  type_counts.get("Dribble", 0) / max(total_actions, 1),
        }

        # --- Efficiency features ---
        # Pass success rate
        passes = pdf[pdf["type"] == "Pass"]
        if "pass_outcome" in pdf.columns and len(passes) > 0:
            incomplete = passes["pass_outcome"].notna().sum()
            pass_success = 1 - incomplete / len(passes)
        else:
            pass_success = np.nan

        # Dribble success rate
        dribbles = pdf[pdf["type"] == "Dribble"]
        if "dribble_outcome" in pdf.columns and len(dribbles) > 0:
            complete = (dribbles["dribble_outcome"] == "Complete").sum()
            dribble_success = complete / len(dribbles)
        else:
            dribble_success = np.nan

        # Shot on target rate
        shots = pdf[pdf["type"] == "Shot"]
        if "shot_outcome" in pdf.columns and len(shots) > 0:
            on_target = shots["shot_outcome"].isin(
                ["Goal", "Saved", "Saved To Post"]
            ).sum()
            shot_on_target = on_target / len(shots)
        else:
            shot_on_target = np.nan

        # Defensive action rate
        def_actions = pdf[pdf["type"].isin(
            ["Pressure", "Interception", "Ball Recovery", "Clearance", "Duel"]
        )]
        def_rate = len(def_actions) / max(total_actions, 1)

        efficiency_feats = {
            "pass_success_rate":   pass_success,
            "dribble_success_rate": dribble_success,
            "shot_on_target_rate": shot_on_target,
            "defensive_action_rate": def_rate,
        }

        # --- Volume features ---
        # Attacking third: x > 80
        att_third = (pdf["x"] > 80).sum()

        # Progressive actions: x increases by >= 10 yards (use carries)
        carries = pdf[pdf["type"] == "Carry"]
        # We don't have carry end location easily, so use x > 60 as proxy for
        # "progressive" (entering opposition half)
        progressive_actions = (pdf["x"] > 60).sum()

        volume_feats = {
            "actions_per90":          total_actions * per90,
            "progressive_actions_p90": progressive_actions * per90,
            "att_third_actions_p90":  att_third * per90,
            "shots_per90":            len(shots) * per90,
        }

        # --- Combine ---
        feats = {**zone_feats, **time_feats, **action_feats,
                 **efficiency_feats, **volume_feats}
        rows.append(feats)

        # Metadata
        team = pdf["team"].iloc[0] if len(pdf) > 0 else "Unknown"
        comp = pdf["competition_label"].iloc[0] if "competition_label" in pdf.columns else "Unknown"
        meta_rows.append({
            "player":      player,
            "team":        team,
            "competition": comp,
            "minutes":     minutes,
            "total_actions": total_actions,
        })

    feat_df = pd.DataFrame(rows, index=eligible_players)
    meta_df = pd.DataFrame(meta_rows, index=eligible_players)

    # Fill NaN efficiency stats with column median
    for col in ["pass_success_rate", "dribble_success_rate", "shot_on_target_rate"]:
        feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    return feat_df, meta_df
