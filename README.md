# City Scout — Football Player Intelligence Platform

A recruitment and player profiling tool built on StatsBomb open data.

## What it does

**Spatial Profile** — Pitch heat map showing where a player operates, plus a 15-minute activity breakdown across the full 90 minutes (spatio-temporal analysis).

**Style & Action Value** — Action Value score estimated by a logistic regression model trained on 1.26M events. Predicts P(shot within 5 events) for each on-ball action. ROC-AUC 0.74 on held-out matches.

**Recruitment** — UMAP-based player similarity. Each of 611 eligible players is embedded as a 25-feature playing style vector. Nearest neighbours in UMAP space are shown as similar players. Man City players from the Premier League 2015/16 season are highlighted, and a City Fit score measures how closely any player's style matches City's playing style.

## Data

StatsBomb open data via `statsbombpy`:
- Premier League 2015/16 (380 matches, includes Manchester City)
- FIFA World Cup 2022 (64 matches)
- UEFA Euro 2020 (51 matches)
- La Liga 2019/20 (33 matches)

Total: 1,260,071 on-ball events · 611 players with 450+ minutes

## Running locally

```bash
pip install -r requirements.txt
python precompute.py   # generates data/ and models/ — run once
streamlit run app.py
```

## Limitations

Built on historical open data only. Man City's playing style under Guardiola has evolved significantly since 2015/16. City Fit scores should be read as a stylistic proxy, not a definitive recruitment recommendation.

---

Data: StatsBomb Open Data — used in accordance with the StatsBomb open data licence.
