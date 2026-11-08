"""
City Scout — Football Player Intelligence Platform
Built on StatsBomb open data.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from mplsoccer import Pitch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="City Scout",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
/* ── Global ── */
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #080c14; color: #f0f4f8; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a1020 100%);
    border-right: 1px solid #1a2744;
}
[data-testid="stSidebar"] label { color: #8b9bb4 !important; font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.3px; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #111c35 !important; border: 1px solid #1e3060 !important; border-radius: 8px !important; color: #e0e8f4 !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
    background: linear-gradient(135deg, #1c3f6e 0%, #0d1f42 100%);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 16px;
    border: 1px solid #2a4d8a;
    text-align: center;
}
.sidebar-logo-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #6cabdd;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.sidebar-logo-sub {
    font-size: 0.68rem;
    color: #4a7ab5;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* ── Hero player banner ── */
.player-hero {
    background: linear-gradient(135deg, #0d1f42 0%, #111c35 50%, #0a1428 100%);
    border: 1px solid #1e3060;
    border-left: 4px solid #6cabdd;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.player-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(108,171,221,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-name {
    font-size: 1.9rem;
    font-weight: 800;
    color: #f0f4f8;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.hero-meta {
    font-size: 0.85rem;
    color: #6b8db8;
    margin-top: 6px;
    font-weight: 500;
}
.hero-badges { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.badge-city {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: linear-gradient(135deg, #1c4f8a, #0d2a5e);
    color: #6cabdd;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border: 1px solid #2a5a9e;
}
.badge-fit {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-comp {
    display: inline-flex;
    align-items: center;
    background: #111c35;
    color: #8b9bb4;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid #1e3060;
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1526;
    border: 1px solid #1a2744;
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6cabdd, #1c3f6e);
    border-radius: 0 0 10px 10px;
}
.metric-label {
    color: #4a7ab5;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    color: #f0f4f8;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.5px;
}
.metric-sub { color: #6cabdd; font-size: 0.7rem; margin-top: 4px; font-weight: 500; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1526;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1a2744;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a7ab5 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1c3f6e, #0d2a5e) !important;
    color: #6cabdd !important;
    border-radius: 8px !important;
}

/* ── Section title ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #6cabdd;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a2744;
    margin-bottom: 14px;
}

/* ── Player similarity rows ── */
.player-row {
    background: #0d1526;
    border: 1px solid #1a2744;
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 7px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.15s, background 0.15s;
}
.player-row:hover { border-color: #2a5a9e; background: #111c35; }
.player-avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1c3f6e, #0d2a5e);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; color: #6cabdd;
    flex-shrink: 0; margin-right: 10px;
}
.player-info { flex: 1; }
.player-name  { font-weight: 700; color: #e8eef8; font-size: 0.88rem; }
.player-meta  { color: #4a7ab5; font-size: 0.75rem; margin-top: 1px; }
.sim-bar-wrap { background: #111c35; border-radius: 3px; height: 3px; margin-top: 5px; overflow: hidden; }
.similarity-bar {
    height: 3px;
    background: linear-gradient(90deg, #6cabdd, #3a7bbf);
    border-radius: 3px;
}
.player-score { text-align: right; min-width: 65px; }
.score-pct  { color: #f0f4f8; font-weight: 700; font-size: 0.95rem; }
.score-fit  { font-size: 0.7rem; margin-top: 2px; font-weight: 600; }

/* ── Stat rows in style tab ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #111c35;
}
.stat-label { color: #4a7ab5; font-size: 0.82rem; }
.stat-val   { color: #e8eef8; font-weight: 700; font-size: 0.85rem; }

/* ── Sidebar model info box ── */
.model-box {
    background: #0d1526;
    border: 1px solid #1a2744;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 4px;
}
.model-auc {
    font-size: 1.5rem;
    font-weight: 800;
    color: #6cabdd;
    line-height: 1;
}

/* ── Chart containers ── */
.chart-box {
    background: #0d1526;
    border: 1px solid #1a2744;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #1e3060;
    font-size: 0.72rem;
    padding: 20px 0 8px;
    border-top: 1px solid #0d1526;
    margin-top: 24px;
}

/* Hide streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_data
def load_all_data():
    feat_df  = pd.read_parquet(os.path.join(DATA_DIR, "player_features.parquet"))
    meta_df  = pd.read_parquet(os.path.join(DATA_DIR, "player_meta.parquet"))
    heat_df  = pd.read_parquet(os.path.join(DATA_DIR, "player_heatmap.parquet"))
    emb      = np.load(os.path.join(DATA_DIR, "umap_embeddings.npy"))
    with open(os.path.join(DATA_DIR, "model_metrics.json")) as f:
        metrics = json.load(f)
    return feat_df, meta_df, heat_df, emb, metrics


feat_df, meta_df, heat_df, emb, metrics = load_all_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div class='sidebar-logo'>"
        "<div class='sidebar-logo-title'>City Scout</div>"
        "<div class='sidebar-logo-sub'>Football Player Intelligence · StatsBomb Open Data</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Competition filter
    competitions = ["All"] + sorted(meta_df["competition"].unique().tolist())
    selected_comp = st.selectbox("Competition", competitions)

    if selected_comp == "All":
        pool = meta_df
    else:
        pool = meta_df[meta_df["competition"] == selected_comp]

    # Team filter
    teams = ["All Teams"] + sorted(pool["team"].unique().tolist())
    selected_team = st.selectbox("Team", teams)
    if selected_team != "All Teams":
        pool = pool[pool["team"] == selected_team]

    # Player search
    player_names = sorted(pool.index.tolist())
    # Default to Kevin De Bruyne if available
    default_idx = player_names.index("Kevin De Bruyne") if "Kevin De Bruyne" in player_names else 0
    selected_player = st.selectbox(
        "Select Player",
        player_names,
        index=default_idx,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='model-box'>"
        f"<div style='color:#4a7ab5;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;'>Action Value ROC-AUC</div>"
        f"<div class='model-auc'>{metrics['action_value_roc_auc']}</div>"
        f"<div style='color:#4a7ab5;font-size:0.72rem;margin-top:6px;line-height:1.5;'>"
        f"Logistic regression · 1.26M events · P(shot within 5 events)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#2a3d6e;font-size:0.68rem;line-height:1.6;'>"
        "Data: StatsBomb Open Data<br>"
        "611 players · 450+ min filter<br>"
        "PL 15/16 · LaLiga 19/20 · WC 2022 · Euro 2020"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Player header ─────────────────────────────────────────────────────────────
player_meta = meta_df.loc[selected_player]
is_city = player_meta["team"] == "Manchester City"
city_fit = player_meta["city_fit"]

# City Fit badge styling
if city_fit >= 85:
    fit_color = "#6cabdd"
    fit_bg    = "rgba(28,79,110,0.5)"
    fit_border = "#2a5a9e"
    fit_label = "Excellent Fit"
elif city_fit >= 65:
    fit_color = "#58a6ff"
    fit_bg    = "rgba(28,65,110,0.4)"
    fit_border = "#2a5090"
    fit_label = "Good Fit"
elif city_fit >= 45:
    fit_color = "#d29922"
    fit_bg    = "rgba(90,65,0,0.4)"
    fit_border = "#7a5500"
    fit_label = "Moderate Fit"
else:
    fit_color = "#6b7280"
    fit_bg    = "rgba(50,50,60,0.4)"
    fit_border = "#374151"
    fit_label = "Low Fit"

city_badge_html = (
    "<span class='badge-city'>🔵 Manchester City</span>"
    if is_city else ""
)
fit_badge_html = (
    f"<span class='badge-fit' style='background:{fit_bg};color:{fit_color};border:1px solid {fit_border};'>"
    f"City Fit {city_fit:.0f}/100 &nbsp;·&nbsp; {fit_label}</span>"
)
comp_badge_html = f"<span class='badge-comp'>{player_meta['competition']}</span>"

st.markdown(
    f"<div class='player-hero'>"
    f"  <div class='hero-name'>{selected_player}</div>"
    f"  <div class='hero-meta'>{player_meta['team']}</div>"
    f"  <div class='hero-badges'>{city_badge_html}{fit_badge_html}{comp_badge_html}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# Top metrics row
player_feat = feat_df.loc[selected_player]
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Minutes</div>"
        f"<div class='metric-value'>{player_meta['minutes']:.0f}</div>"
        f"<div class='metric-sub'>in dataset</div></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Actions / 90</div>"
        f"<div class='metric-value'>{player_feat.get('actions_per90', 0):.0f}</div>"
        f"<div class='metric-sub'>on-ball</div></div>",
        unsafe_allow_html=True,
    )
with col3:
    av_score = player_feat.get("avg_action_value", 0)
    comp_median = feat_df["avg_action_value"].median()
    av_delta = "▲" if av_score > comp_median else "▼"
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Avg Action Value</div>"
        f"<div class='metric-value'>{av_score:.3f}</div>"
        f"<div class='metric-sub'>{av_delta} vs median ({comp_median:.3f})</div></div>",
        unsafe_allow_html=True,
    )
with col4:
    pass_rate = player_feat.get("pass_success_rate", 0)
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Pass Success</div>"
        f"<div class='metric-value'>{pass_rate*100:.0f}%</div>"
        f"<div class='metric-sub'>completion rate</div></div>",
        unsafe_allow_html=True,
    )
with col5:
    shots_p90 = player_feat.get("shots_per90", 0)
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Shots / 90</div>"
        f"<div class='metric-value'>{shots_p90:.1f}</div>"
        f"<div class='metric-sub'>&nbsp;</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📍 Spatial Profile", "📈 Style & Timing", "🔍 Recruitment"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: SPATIAL PROFILE — heat map + temporal activity
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    colA, colB = st.columns([1.1, 1])

    with colA:
        st.markdown("<div class='section-title'>Where This Player Operates</div>", unsafe_allow_html=True)

        player_events = heat_df[heat_df["player"] == selected_player]

        fig_pitch, ax_pitch = plt.subplots(figsize=(8, 5.5))
        fig_pitch.patch.set_facecolor("#080c14")

        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#080c14",
            line_color="#1a2744",
            linewidth=1.2,
        )
        pitch.draw(ax=ax_pitch)

        if len(player_events) > 0:
            pitch.kdeplot(
                player_events["x"],
                player_events["y"],
                ax=ax_pitch,
                cmap="Blues",
                fill=True,
                alpha=0.85,
                levels=50,
                thresh=0.03,
            )

            # Overlay shot locations
            shots_ev = player_events[player_events["minute"].notna()]

        ax_pitch.set_title(
            f"{selected_player} — All On-Ball Actions",
            color="#8b949e", fontsize=9, pad=8,
        )
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_pitch, use_container_width=True)
        plt.close(fig_pitch)

        st.markdown(
            "<div style='color:#8b949e;font-size:0.75rem;text-align:center;'>"
            f"Based on {len(player_events):,} on-ball actions (passes, carries, shots, dribbles, pressures)</div>",
            unsafe_allow_html=True,
        )

    with colB:
        st.markdown("<div class='section-title'>Activity by Match Phase (Spatio-Temporal)</div>", unsafe_allow_html=True)

        # 15-minute window activity
        windows = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]
        feat_keys = ["pct_t0_15", "pct_t15_30", "pct_t30_45", "pct_t45_60", "pct_t60_75", "pct_t75_90"]
        activity = [player_feat.get(k, 0) * 100 for k in feat_keys]
        comp_activity = [feat_df[k].median() * 100 for k in feat_keys]

        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=windows, y=activity,
            name=selected_player.split()[0],
            marker_color="#6cabdd",
            opacity=0.9,
        ))
        fig_time.add_trace(go.Scatter(
            x=windows, y=comp_activity,
            name="Dataset median",
            mode="lines+markers",
            line=dict(color="#444c56", dash="dash", width=1.5),
            marker=dict(size=5, color="#444c56"),
        ))
        fig_time.update_layout(
            plot_bgcolor="#080c14",
            paper_bgcolor="#080c14",
            font=dict(color="#4a7ab5", size=11),
            legend=dict(
                bgcolor="#0d1526", bordercolor="#1a2744", borderwidth=1,
                font=dict(size=10),
            ),
            xaxis=dict(
                gridcolor="#111c35", title="Match Phase",
                title_font=dict(color="#4a7ab5"),
            ),
            yaxis=dict(
                gridcolor="#111c35", title="% of Total Actions",
                title_font=dict(color="#4a7ab5"),
            ),
            margin=dict(l=10, r=10, t=20, b=10),
            height=250,
        )
        st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Pitch Zone Breakdown</div>", unsafe_allow_html=True)

        zone_labels = ["Left Def.", "Centre Def.", "Right Def.", "Left Att.", "Centre Att.", "Right Att."]
        zone_keys   = ["pct_left_def","pct_centre_def","pct_right_def","pct_left_att","pct_centre_att","pct_right_att"]
        zone_vals   = [player_feat.get(k, 0) * 100 for k in zone_keys]
        zone_colors = ["#1c3f6e","#1c3f6e","#1c3f6e","#6cabdd","#6cabdd","#6cabdd"]

        fig_zone = go.Figure(go.Bar(
            x=zone_labels, y=zone_vals,
            marker_color=zone_colors,
            opacity=0.9,
        ))
        fig_zone.update_layout(
            plot_bgcolor="#080c14",
            paper_bgcolor="#080c14",
            font=dict(color="#4a7ab5", size=10),
            xaxis=dict(gridcolor="#111c35"),
            yaxis=dict(gridcolor="#111c35", title="% of Actions"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            showlegend=False,
        )
        st.plotly_chart(fig_zone, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: STYLE & ACTION VALUE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    colC, colD = st.columns([1, 1])

    with colC:
        st.markdown("<div class='section-title'>Action Value Score</div>", unsafe_allow_html=True)

        av = player_feat.get("avg_action_value", 0)
        all_av = feat_df["avg_action_value"].dropna().sort_values()
        percentile = (all_av < av).mean() * 100

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=av,
            delta={"reference": feat_df["avg_action_value"].median(),
                   "valueformat": ".4f",
                   "font": {"color": "#8b949e", "size": 12}},
            number={"valueformat": ".4f", "font": {"color": "#e6edf3", "size": 28}},
            gauge={
                "axis": {"range": [all_av.min(), all_av.max()],
                         "tickformat": ".3f",
                         "tickcolor": "#4a7ab5",
                         "tickfont": {"color": "#4a7ab5", "size": 10}},
                "bar": {"color": "#6cabdd"},
                "bgcolor": "#0d1526",
                "bordercolor": "#1a2744",
                "steps": [
                    {"range": [all_av.min(), all_av.quantile(0.33)], "color": "#111c35"},
                    {"range": [all_av.quantile(0.33), all_av.quantile(0.67)], "color": "#1c3f6e"},
                    {"range": [all_av.quantile(0.67), all_av.max()], "color": "#2d5a9e"},
                ],
                "threshold": {
                    "value": feat_df["avg_action_value"].median(),
                    "line": {"color": "#6cabdd", "width": 2},
                },
            },
            title={"text": f"Avg Action Value<br><span style='font-size:0.7rem;color:#8b949e'>"
                           f"P(shot within 5 events) · {percentile:.0f}th percentile</span>",
                   "font": {"color": "#8b949e", "size": 13}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#080c14",
            font=dict(color="#e8eef8"),
            height=270,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            "<div style='color:#8b949e;font-size:0.78rem;line-height:1.5;'>"
            "Action Value estimates the probability that an on-ball action leads to a shot "
            "within the next 5 events. Trained via logistic regression on 1.26M events using "
            "action type, pitch coordinates, distance and angle to goal, and match phase. "
            f"Model ROC-AUC on held-out matches: <b style='color:#6cabdd'>{metrics['action_value_roc_auc']}</b>."
            "</div>",
            unsafe_allow_html=True,
        )

    with colD:
        st.markdown("<div class='section-title'>Playing Style Breakdown</div>", unsafe_allow_html=True)

        style_labels = ["Passing", "Carrying", "Shooting", "Pressing", "Dribbling"]
        style_keys   = ["pct_pass", "pct_carry", "pct_shot", "pct_pressure", "pct_dribble"]
        player_style = [player_feat.get(k, 0) * 100 for k in style_keys]
        median_style = [feat_df[k].median() * 100 for k in style_keys]

        fig_style = go.Figure()
        fig_style.add_trace(go.Bar(
            y=style_labels, x=player_style,
            orientation="h",
            name=selected_player.split()[0],
            marker_color="#6cabdd",
            opacity=0.9,
        ))
        fig_style.add_trace(go.Bar(
            y=style_labels, x=median_style,
            orientation="h",
            name="Dataset median",
            marker_color="#444c56",
            opacity=0.5,
        ))
        fig_style.update_layout(
            plot_bgcolor="#080c14",
            paper_bgcolor="#080c14",
            font=dict(color="#4a7ab5", size=11),
            barmode="group",
            legend=dict(bgcolor="#0d1526", bordercolor="#1a2744", borderwidth=1, font=dict(size=10)),
            xaxis=dict(gridcolor="#111c35", title="% of Actions"),
            yaxis=dict(gridcolor="#111c35"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=240,
        )
        st.plotly_chart(fig_style, use_container_width=True)

        st.markdown("<div class='section-title'>Efficiency Metrics</div>", unsafe_allow_html=True)

        eff_data = {
            "Pass completion": f"{player_feat.get('pass_success_rate', 0)*100:.1f}%",
            "Dribble success": f"{player_feat.get('dribble_success_rate', 0)*100:.1f}%",
            "Shot on target":  f"{player_feat.get('shot_on_target_rate', 0)*100:.1f}%",
            "Defensive action rate": f"{player_feat.get('defensive_action_rate', 0)*100:.1f}%",
        }
        for label, val in eff_data.items():
            st.markdown(
                f"<div class='stat-row'>"
                f"<span class='stat-label'>{label}</span>"
                f"<span class='stat-val'>{val}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: RECRUITMENT — UMAP + Similar Players
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    colE, colF = st.columns([1.3, 1])

    with colE:
        st.markdown("<div class='section-title'>Player Similarity Map (UMAP)</div>", unsafe_allow_html=True)

        # Colour by competition
        comp_colors = {
            "Premier League 2015/16": "#38003c",   # PL purple
            "FIFA World Cup 2022":    "#c41e3a",   # WC red
            "UEFA Euro 2020":         "#003399",   # UEFA blue
            "La Liga 2019/20":        "#ff4500",   # La Liga orange
        }
        default_color = "#555"

        # Build scatter
        fig_umap = go.Figure()

        selected_idx = meta_df.index.get_loc(selected_player)

        for comp in meta_df["competition"].unique():
            mask = meta_df["competition"] == comp
            idxs = [meta_df.index.get_loc(p) for p in meta_df[mask].index]
            city_mask_comp = (meta_df["team"] == "Manchester City") & mask

            # Non-city players of this competition
            non_city_idxs = [
                meta_df.index.get_loc(p)
                for p in meta_df[mask & (meta_df["team"] != "Manchester City")].index
            ]
            city_idxs = [
                meta_df.index.get_loc(p)
                for p in meta_df[city_mask_comp].index
            ]

            if non_city_idxs:
                fig_umap.add_trace(go.Scatter(
                    x=emb[non_city_idxs, 0],
                    y=emb[non_city_idxs, 1],
                    mode="markers",
                    name=comp,
                    text=[meta_df.index[i] + "<br>" + meta_df.iloc[i]["team"]
                          for i in non_city_idxs],
                    hoverinfo="text",
                    marker=dict(
                        size=6,
                        color=comp_colors.get(comp, default_color),
                        opacity=0.55,
                    ),
                ))

        # Man City players (highlighted)
        city_all_idxs = [meta_df.index.get_loc(p)
                         for p in meta_df[meta_df["team"] == "Manchester City"].index]
        if city_all_idxs:
            fig_umap.add_trace(go.Scatter(
                x=emb[city_all_idxs, 0],
                y=emb[city_all_idxs, 1],
                mode="markers",
                name="🔵 Man City",
                text=[meta_df.index[i] for i in city_all_idxs],
                hoverinfo="text",
                marker=dict(size=10, color="#6cabdd", symbol="diamond", opacity=0.9,
                            line=dict(color="white", width=0.8)),
            ))

        # Selected player (starred)
        fig_umap.add_trace(go.Scatter(
            x=[emb[selected_idx, 0]],
            y=[emb[selected_idx, 1]],
            mode="markers",
            name=f"★ {selected_player.split()[0]}",
            text=[selected_player],
            hoverinfo="text",
            marker=dict(size=16, color="#ffd700", symbol="star",
                        line=dict(color="white", width=1.5)),
        ))

        fig_umap.update_layout(
            plot_bgcolor="#080c14",
            paper_bgcolor="#080c14",
            font=dict(color="#4a7ab5", size=11),
            legend=dict(bgcolor="#0d1526", bordercolor="#1a2744", borderwidth=1,
                        font=dict(size=10), itemsizing="constant"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
        )
        st.plotly_chart(fig_umap, use_container_width=True)
        st.markdown(
            "<div style='color:#8b949e;font-size:0.75rem;'>"
            "Each dot is a player (450+ minutes). Distance = style similarity. "
            "🔵 Man City players highlighted. ★ = selected player. "
            "UMAP: nonlinear dimensionality reduction on 25-feature playing style vectors.</div>",
            unsafe_allow_html=True,
        )

    with colF:
        st.markdown("<div class='section-title'>Similar Players — Style Nearest Neighbours</div>", unsafe_allow_html=True)

        # Find similar players
        from pipeline.similarity import find_similar_players
        similar = find_similar_players(selected_player, emb, meta_df, top_n=8)

        if similar.empty:
            st.write("No similar players found.")
        else:
            # Rescale bar: 100% for most similar, 20% for least (so all bars are visible)
            d_min = similar["umap_distance"].min()
            d_max = similar["umap_distance"].max()
            similar = similar.copy()
            similar["sim_bar"] = 100 - 80 * (similar["umap_distance"] - d_min) / (d_max - d_min + 1e-9)

            for _, row in similar.iterrows():
                p_name = row["player"]
                p_team = row["team"]
                p_comp = row["competition"]
                p_sim  = row["sim_bar"]
                p_fit  = row["city_fit"]
                p_is_city = p_team == "Manchester City"

                # Avatar initials
                parts = p_name.split()
                initials = (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else p_name[:2].upper()

                city_tag = " &nbsp;<span style='font-size:0.75rem;'>🔵</span>" if p_is_city else ""

                if p_fit >= 85:
                    fit_color_p = "#6cabdd"
                    fit_label_p = "Excellent"
                elif p_fit >= 65:
                    fit_color_p = "#58a6ff"
                    fit_label_p = "Good"
                elif p_fit >= 45:
                    fit_color_p = "#d29922"
                    fit_label_p = "Moderate"
                else:
                    fit_color_p = "#6b7280"
                    fit_label_p = "Low"

                st.markdown(
                    f"<div class='player-row'>"
                    f"  <div class='player-avatar'>{initials}</div>"
                    f"  <div class='player-info'>"
                    f"    <div class='player-name'>{p_name}{city_tag}</div>"
                    f"    <div class='player-meta'>{p_team} &nbsp;·&nbsp; {p_comp}</div>"
                    f"    <div class='sim-bar-wrap'><div class='similarity-bar' style='width:{p_sim:.0f}%;'></div></div>"
                    f"  </div>"
                    f"  <div class='player-score'>"
                    f"    <div class='score-pct'>{p_sim:.0f}%</div>"
                    f"    <div class='score-fit' style='color:{fit_color_p};'>City {p_fit:.0f}</div>"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>City Fit Explained</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#8b949e;font-size:0.78rem;line-height:1.6;'>"
            "City Fit measures how closely a player's style profile matches the "
            "centroid of Manchester City's squad from the Premier League 2015/16 season. "
            "Computed as cosine similarity to the City centroid in UMAP embedding space, "
            "scaled 0-100.<br><br>"
            "<b style='color:#e6edf3;'>Limitation:</b> Built on historical open data only. "
            "The club's playing style has evolved considerably since 2015/16, and this score "
            "reflects that squad's stylistic profile rather than current tactical requirements. "
            "Treat it as a directional indicator, not a definitive recruitment recommendation."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Top 10 City Fit (full width, below both columns) ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Top 10 City Fit — Best Style Matches Across Dataset</div>", unsafe_allow_html=True)

    top_city = meta_df[meta_df["team"] != "Manchester City"].nlargest(10, "city_fit")

    col_left, col_right = st.columns(2)
    for rank, (p_name, row) in enumerate(top_city.iterrows(), 1):
        p_team = row["team"]
        p_comp = row["competition"]
        p_fit  = row["city_fit"]
        p_mins = row["minutes"]

        parts    = p_name.split()
        initials = (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else p_name[:2].upper()

        if p_fit >= 85:
            fit_color = "#6cabdd"
            fit_label = "Excellent"
        elif p_fit >= 65:
            fit_color = "#58a6ff"
            fit_label = "Good"
        elif p_fit >= 45:
            fit_color = "#d29922"
            fit_label = "Moderate"
        else:
            fit_color = "#6b7280"
            fit_label = "Low"

        row_html = (
            f"<div class='player-row'>"
            f"  <div style='color:#2a3d6e;font-size:0.82rem;font-weight:800;min-width:22px;text-align:center;'>#{rank}</div>"
            f"  <div class='player-avatar' style='margin-left:10px;'>{initials}</div>"
            f"  <div class='player-info'>"
            f"    <div class='player-name'>{p_name}</div>"
            f"    <div class='player-meta'>{p_team} &nbsp;·&nbsp; {p_comp} &nbsp;·&nbsp; {p_mins:.0f} min</div>"
            f"  </div>"
            f"  <div class='player-score'>"
            f"    <div class='score-pct' style='color:{fit_color};'>{p_fit:.0f}</div>"
            f"    <div class='score-fit' style='color:{fit_color};'>{fit_label} City Fit</div>"
            f"  </div>"
            f"</div>"
        )
        if rank <= 5:
            col_left.markdown(row_html, unsafe_allow_html=True)
        else:
            col_right.markdown(row_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "City Scout &nbsp;·&nbsp; Built with StatsBomb Open Data &nbsp;·&nbsp; "
    "1,260,071 events &nbsp;·&nbsp; 611 players &nbsp;·&nbsp; 4 competitions<br>"
    "StatsBomb data used in accordance with their open data licence."
    "</div>",
    unsafe_allow_html=True,
)
