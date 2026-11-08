"""
Load StatsBomb open data - memory-efficient version.
Processes each match individually and extracts only needed columns.
"""

import os
import pickle
import numpy as np
import pandas as pd
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")

# Competitions to load
COMPETITIONS = [
    {"competition_id": 2,  "season_id": 27, "label": "Premier League 2015/16"},
    {"competition_id": 43, "season_id": 106,"label": "FIFA World Cup 2022"},
    {"competition_id": 55, "season_id": 43, "label": "UEFA Euro 2020"},
    {"competition_id": 11, "season_id": 42, "label": "La Liga 2019/20"},
]

RELEVANT_TYPES = {
    "Pass", "Shot", "Dribble", "Carry", "Pressure",
    "Ball Recovery", "Clearance", "Interception", "Duel"
}

COLS_NEEDED = [
    "id", "index", "match_id", "player", "team",
    "type", "minute", "second", "location",
    "pass_outcome", "dribble_outcome", "shot_outcome",
]

CACHE_DIR = "data/cache"


def _extract_match_events(match_id, competition_id, season_id, label):
    """Load one match, extract only needed columns."""
    ev = sb.events(match_id=match_id)
    ev["match_id"] = match_id
    ev["competition_id"] = competition_id
    ev["season_id"] = season_id
    ev["competition_label"] = label

    # Expand type name
    if "type" in ev.columns:
        ev["type"] = ev["type"].apply(
            lambda t: t.get("name", "") if isinstance(t, dict) else str(t)
        )

    # Expand player name
    if "player" in ev.columns:
        ev["player"] = ev["player"].apply(
            lambda p: p.get("name", "") if isinstance(p, dict) else str(p)
        )

    # Expand team name
    if "team" in ev.columns:
        ev["team"] = ev["team"].apply(
            lambda t: t.get("name", "") if isinstance(t, dict) else str(t)
        )

    # Expand outcome columns if they're dicts
    for col in ["pass_outcome", "dribble_outcome", "shot_outcome"]:
        if col in ev.columns:
            ev[col] = ev[col].apply(
                lambda x: x.get("name", None) if isinstance(x, dict) else x
            )
        else:
            ev[col] = None

    # Keep only needed columns + extras
    keep = COLS_NEEDED + ["competition_label"]
    ev = ev[[c for c in keep if c in ev.columns]].copy()

    # Filter to on-ball types only
    ev = ev[ev["type"].isin(RELEVANT_TYPES)]

    # Unpack location
    ev = ev[ev["location"].notna()].copy()
    ev["x"] = ev["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 2 else None
    )
    ev["y"] = ev["location"].apply(
        lambda loc: loc[1] if isinstance(loc, list) and len(loc) >= 2 else None
    )
    ev = ev.drop(columns=["location"])
    ev = ev[ev["x"].notna() & ev["y"].notna()]

    ev["time_seconds"] = ev["minute"] * 60 + ev["second"].fillna(0)

    return ev


def load_competition_events(competition_id, season_id, label=""):
    """Load all events for one competition/season, with caching."""
    path = os.path.join(CACHE_DIR, f"events_{competition_id}_{season_id}.parquet")
    if os.path.exists(path):
        print(f"  [cache] {label}")
        return pd.read_parquet(path)

    print(f"  [fetch] {label} ...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"    {len(matches)} matches")

    frames = []
    for i, match_id in enumerate(matches["match_id"]):
        if i % 50 == 0:
            print(f"    match {i+1}/{len(matches)}")
        try:
            ev = _extract_match_events(match_id, competition_id, season_id, label)
            frames.append(ev)
        except Exception as e:
            print(f"    skip {match_id}: {e}")

        # Flush every 100 matches to manage memory
        if len(frames) >= 100:
            partial = pd.concat(frames, ignore_index=True)
            frames = [partial]

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(path)
    print(f"    done: {len(df):,} events")
    return df


def load_all_events():
    """Load events for all configured competitions."""
    frames = []
    for comp in COMPETITIONS:
        df = load_competition_events(
            comp["competition_id"], comp["season_id"], comp["label"]
        )
        frames.append(df)
        print(f"    {comp['label']}: {len(df):,} on-ball events")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal on-ball events: {len(combined):,}")
    return combined
