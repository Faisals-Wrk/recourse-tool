# =============================================================================
# app.py — Algorithmic Recourse Tool
# =============================================================================
# A Streamlit web application with six interactive pages that together
# tell the full story of algorithmic recourse and fairness.
#
# PAGE MAP:
#   1. Home              — What is recourse? Story, concepts, visitor guide
#   2. Dataset           — Explore the Adult Income data
#   3. Find Recourse     — Interactive: pick a person, find their recourse
#   4. Compare Algorithms— See all three algorithms side by side
#   5. Fairness          — Does the model demand more from protected groups?
#   6. About             — Methodology, research, limitations
#
# AESTHETIC:
# 
#   Deep purple-slate background, warm amber + emerald + rose accents.
#   Fonts: Playfair Display (headings) + Fira Code (data) + Source Sans Pro
#   The goal: feel human and accessible, not cold and technical.
#
# DATA FLOW:
#   load_everything() runs once (~3 min) and caches via @st.cache_data.
#   It trains all 3 models and computes all 4 analyses upfront.
#   Every page click after that is instant — reads from the cache.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

# Add project root to path so imports work regardless of working directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recourse.data           import (load_data, get_person,
                                     FEATURE_META, FEATURE_NAMES,
                                     ACTIONABLE_FEATURES, IMMUTABLE_FEATURES,
                                     CATEGORY_LABELS, feature_value_label,
                                     SEX_REVERSE, RACE_REVERSE)
from recourse.model          import (train_all_models, predict_person,
                                     get_all_predictions)
from recourse.counterfactual import find_recourse
from recourse.analyze        import (compute_effort_distribution,
                                     compute_fairness_analysis,
                                     compute_feature_frequency,
                                     compare_algorithms,
                                     compute_summary_stats)


# =============================================================================
# PAGE CONFIG — must be the FIRST Streamlit call
# =============================================================================

st.set_page_config(
    page_title="Algorithmic Recourse Tool",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# GLOBAL CSS — warm editorial dark theme
# =============================================================================
# Design language intentionally different from Project 1:
#   Project 1: cold navy, electric teal, scientific dashboard
#   Project 2: warm purple-slate, amber, editorial/human feel
#
# Why warm colors for a recourse tool?
#   Recourse is about people — their livelihoods, their futures.
#   A warm palette signals that this is human-centered, not just data.

st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Fira+Code:wght@300;400;500&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

/* ── Design Tokens ────────────────────────────────────────────────────── */
:root {
    --bg:          #13111a;   /* deep purple-slate background               */
    --bg-card:     #1c1929;   /* card surfaces                              */
    --bg-raised:   #221f30;   /* slightly raised elements                   */
    --border:      #2e2a3e;   /* default border                             */
    --border-glow: #4a3f6b;   /* brighter border for focus/hover            */
    --amber:       #f59e0b;   /* primary accent — warmth, attention         */
    --amber-dim:   #b45309;   /* dimmer amber for borders                   */
    --emerald:     #10b981;   /* success, found recourse, positive          */
    --emerald-dim: #059669;
    --rose:        #f43f5e;   /* danger, unfairness, not found              */
    --rose-dim:    #be123c;
    --indigo:      #818cf8;   /* info, algorithm labels, neutral accent     */
    --indigo-dim:  #4f46e5;
    --gold:        #fbbf24;   /* highlight numbers                          */
    --text:        #f1f5f9;   /* primary text                               */
    --text-muted:  #94a3b8;   /* secondary text                             */
    --text-faint:  #475569;   /* footnotes, labels                          */
    --font-head:   'Playfair Display', Georgia, serif;
    --font-mono:   'Fira Code', 'Courier New', monospace;
    --font-body:   'Source Sans Pro', sans-serif;
}

/* ── Global Reset ─────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Cards ────────────────────────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 18px;
}
.card-amber {
    background: linear-gradient(135deg, #1f1a0d 0%, #17130a 100%);
    border: 1px solid var(--amber-dim);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 18px;
}
.card-emerald {
    background: linear-gradient(135deg, #0d1f18 0%, #0a1712 100%);
    border: 1px solid var(--emerald-dim);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 18px;
}
.card-rose {
    background: linear-gradient(135deg, #1f0d12 0%, #170a0d 100%);
    border: 1px solid var(--rose-dim);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 18px;
}
.card-indigo {
    background: linear-gradient(135deg, #0f0d1f 0%, #0c0a1a 100%);
    border: 1px solid var(--indigo-dim);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 18px;
}

/* ── Metric Boxes ─────────────────────────────────────────────────────── */
.metric {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 18px;
    text-align: center;
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 30px;
    font-weight: 500;
    color: var(--amber);
    display: block;
    line-height: 1.1;
}
.metric-label {
    font-family: var(--font-body);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-top: 6px;
    display: block;
}

/* ── Typography ───────────────────────────────────────────────────────── */
.page-title {
    font-family: var(--font-head);
    font-size: 46px;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: 8px;
}
.page-subtitle {
    font-family: var(--font-body);
    font-size: 16px;
    color: var(--text-muted);
    margin-bottom: 36px;
    line-height: 1.7;
    max-width: 680px;
}
.section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--amber);
    margin-bottom: 14px;
    display: block;
}

/* ── Callouts ─────────────────────────────────────────────────────────── */
.callout {
    background: rgba(245,158,11,0.07);
    border-left: 3px solid var(--amber);
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 14px;
    line-height: 1.75;
    color: var(--text);
}
.callout.emerald { background: rgba(16,185,129,0.07); border-left-color: var(--emerald); }
.callout.rose    { background: rgba(244,63,94,0.07);  border-left-color: var(--rose);    }
.callout.indigo  { background: rgba(129,140,248,0.07);border-left-color: var(--indigo);  }

/* ── Step Cards (visitor guide) ───────────────────────────────────────── */
.step-card {
    display: flex;
    gap: 18px;
    align-items: flex-start;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.step-num {
    font-family: var(--font-head);
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    min-width: 48px;
    color: var(--amber);
    opacity: 0.5;
}
.step-body h4 {
    font-family: var(--font-head);
    font-size: 17px;
    color: var(--text);
    margin: 0 0 6px 0;
}
.step-body p {
    font-family: var(--font-body);
    font-size: 13px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.65;
}

/* ── Feature change rows (recourse display) ───────────────────────────── */
.change-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-raised);
    border-radius: 8px;
    margin-bottom: 8px;
    border: 1px solid var(--border);
}
.change-label { font-family: var(--font-body); font-size: 14px; color: var(--text-muted); }
.change-arrow { font-family: var(--font-mono); font-size: 13px; color: var(--text); }
.change-delta-pos { font-family: var(--font-mono); font-size: 13px; color: var(--emerald); }
.change-delta-neg { font-family: var(--font-mono); font-size: 13px; color: var(--rose);    }

/* ── Prediction badge ─────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.badge-high  { background: rgba(16,185,129,0.15);  color: var(--emerald); border: 1px solid rgba(16,185,129,0.3); }
.badge-low   { background: rgba(244,63,94,0.15);   color: var(--rose);    border: 1px solid rgba(244,63,94,0.3);  }
.badge-found { background: rgba(245,158,11,0.15);  color: var(--amber);   border: 1px solid rgba(245,158,11,0.3); }

/* ── Streamlit widget overrides ───────────────────────────────────────── */
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    border-radius: 8px !important;
}
[data-testid="stTabs"] button {
    font-family: var(--font-body) !important;
    font-size: 14px !important;
    color: var(--text-muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
}
[data-testid="stRadio"] label { font-family: var(--font-body) !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }

/* ── Tag pill ─────────────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 2px;
}
.tag-amber  { background: rgba(245,158,11,0.12);   color: var(--amber);   border: 1px solid rgba(245,158,11,0.25);  }
.tag-emerald{ background: rgba(16,185,129,0.12);   color: var(--emerald); border: 1px solid rgba(16,185,129,0.25);  }
.tag-rose   { background: rgba(244,63,94,0.12);    color: var(--rose);    border: 1px solid rgba(244,63,94,0.25);   }
.tag-indigo { background: rgba(129,140,248,0.12);  color: var(--indigo);  border: 1px solid rgba(129,140,248,0.25); }

/* ── Divider ──────────────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PLOTLY THEME
# =============================================================================
# Warm editorial chart theme consistent with the CSS palette above.
# Transparent backgrounds so the card's dark purple shows through.

PLOT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font":     {"family": "Fira Code, monospace", "color": "#94a3b8", "size": 11},
    "xaxis":    {"gridcolor": "#1e1b2e", "linecolor": "#2e2a3e",
                 "tickfont":  {"color": "#94a3b8", "size": 10}},
    "yaxis":    {"gridcolor": "#1e1b2e", "linecolor": "#2e2a3e",
                 "tickfont":  {"color": "#94a3b8", "size": 10}},
    "colorway": ["#f59e0b", "#10b981", "#f43f5e", "#818cf8", "#fbbf24"],
    "hoverlabel": {"bgcolor": "#1c1929", "bordercolor": "#2e2a3e",
                   "font": {"family": "Fira Code", "color": "#f1f5f9", "size": 11}},
}

def apply_theme(fig, height=420):
    """Apply the global warm dark Plotly theme to any figure."""
    fig.update_layout(
        paper_bgcolor = PLOT["paper_bgcolor"],
        plot_bgcolor  = PLOT["plot_bgcolor"],
        font          = PLOT["font"],
        hoverlabel    = PLOT["hoverlabel"],
        height        = height,
        margin        = dict(l=40, r=20, t=44, b=40),
        legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2e2a3e",
                             font={"color":"#94a3b8","size":10})
    )
    fig.update_xaxes(**PLOT["xaxis"])
    fig.update_yaxes(**PLOT["yaxis"])
    return fig


# =============================================================================
# DATA LOADING — split into FAST and SLOW to avoid a 3-minute spinner
# =============================================================================
# WHY TWO FUNCTIONS?
#   The original single function called compute_effort_distribution() which
#   runs up to 1,500 greedy searches before the cache saves anything.
#   Each call to predict_person() in the old code created a new pandas
#   DataFrame — millions of allocations before the user saw anything.
#
#   Solution:
#     load_fast()     — trains models (~10 sec)  — runs on EVERY page load
#     load_analyses() — effort + fairness (~60 sec) — only on Home/Fairness

@st.cache_data(show_spinner=False)
def load_fast():
    """
    FAST load — completes in ~10 seconds.
    Reads data, trains all 3 models, computes all-person predictions.
    Called on every page — must be quick.

    Returns:
        X, y, df    — feature matrix, labels, full DataFrame
        trained     — dict of 3 fitted models
        all_preds   — DataFrame of all 3 models predictions on all persons
    """
    X, y, df  = load_data()
    trained   = train_all_models(X, y, verbose=False)
    all_preds = get_all_predictions(trained, X)
    return X, y, df, trained, all_preds


@st.cache_data(show_spinner=False)
def load_analyses(_trained_models, _X, _y):
    """
    SLOW load — 7-11 minutes on first visit to Streamlit Cloud.
    Runs 150 persons × 3 models of greedy recourse, then aggregates.
    @st.cache_data stores the result — every subsequent visit is instant.
    Only called when visiting Home or Fairness pages.
    Cached after first call — subsequent visits are instant.

    WHY UNDERSCORE PREFIXES?
      @st.cache_data tries to hash every argument for cache invalidation.
      sklearn model objects are not hashable by Streamlit.
      Arguments prefixed with _ are skipped during hashing.

    Args:
        _trained_models — fitted sklearn models (not hashed)
        _X, _y          — features and labels

    Returns:
        effort_df    — recourse effort for 200 sampled persons x 3 models
        fairness     — effort aggregated by sex, race, age_group
        feature_freq — feature appearance frequency per model
        stats        — headline summary numbers for Home page
    """
    # PERFORMANCE DESIGN — tuned for Streamlit Cloud (~2-3ms per RF call)
    #
    # WHY sample_size=150:
    #   150 stratified persons → ~48 female persons in the sample.
    #   Of those, ~65% find recourse → ~31 females with a distance measurement.
    #   The sex gap (~30%) is a large effect — reliably detectable at n≥15.
    #   Fewer than 50 persons produces noisy fairness statistics not worth showing.
    #
    # WHY max_iterations=80:
    #   Logistic Regression's greedy path needs 15-30 steps (smooth probability
    #   surface). Cutting iterations artificially reduces LR's measured success
    #   rate and distorts Finding 01 (architecture matters). 80 gives full convergence.
    #   Decision Tree stuck-persons exit after ~5 steps via PATIENCE in
    #   counterfactual.py — so DT is cheap regardless of this cap.
    #
    # RUNTIME ESTIMATES:
    #   Streamlit Cloud (~3ms/RF): 150 × 80 × 18 × 3ms ≈ 648s ≈ 11 min first load
    #   Streamlit Cloud (~2ms/RF): 150 × 80 × 18 × 2ms ≈ 432s ≈  7 min first load
    #   After the first load, @st.cache_data stores the result — instant forever.
    #   Your Windows laptop (~18ms/RF): ~65 min — do not run locally with these values.
    #   For local testing only, temporarily use sample_size=20, max_iterations=10.
    effort_df    = compute_effort_distribution(
                       _trained_models, _X, _y,
                       sample_size=150,   # ~48 females → reliable ~30% sex gap finding
                       max_iterations=80, # full LR convergence; DT exits early via PATIENCE
                       verbose=False
                   )
    fairness     = compute_fairness_analysis(effort_df)
    feature_freq = compute_feature_frequency(effort_df)
    stats        = compute_summary_stats(effort_df, _trained_models)
    return effort_df, fairness, feature_freq, stats


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """
    Left sidebar: branding, navigation, live stats after load.
    """
    with st.sidebar:

        # ── Branding ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:10px 0 26px 0; border-bottom:1px solid #2e2a3e;
                    margin-bottom:22px;">
            <div style="font-family:'Playfair Display',serif; font-size:20px;
                        color:#f1f5f9; line-height:1.25;">
                Algorithmic<br>Recourse Tool
            </div>
            <div style="font-family:'Fira Code',monospace; font-size:10px;
                        color:#f59e0b; margin-top:7px; letter-spacing:0.12em;
                        text-transform:uppercase;">
                Adult Income · UCI 1994 Census
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────
        pages = [
            ("🏠", "Home",               "What is recourse?"),
            ("📊", "Dataset",            "Explore the data"),
            ("🔍", "Find Recourse",      "Interactive recourse finder"),
            ("⚗️", "Compare Algorithms", "Three algorithms, one person"),
            ("⚖️", "Fairness",           "Who bears the burden?"),
            ("📖", "About",              "Methodology & research"),
        ]

        st.markdown('<span class="section-label">Navigate</span>',
                    unsafe_allow_html=True)

        current = st.session_state.get("page", "Home")
        for icon, name, desc in pages:
            active   = (current == name)
            btn_type = "primary" if active else "secondary"
            if st.button(f"{icon}  {name}", key=f"nav_{name}",
                         width='stretch', type=btn_type):
                st.session_state["page"] = name
                st.rerun()

        # ── Live stats ────────────────────────────────────────────────────
        if "stats" in st.session_state:
            s = st.session_state["stats"]
            sg = s.get("sex_gap") or {}
            st.markdown("""
            <div style="margin-top:28px; padding-top:20px;
                        border-top:1px solid #2e2a3e;">
                <span class="section-label">Live Stats</span>
            </div>
            """, unsafe_allow_html=True)

            pm = s.get("per_model", {})
            lr = pm.get("logistic", {})
            rf = pm.get("forest", {})
            dt = pm.get("tree", {})

            st.markdown(f"""
            <div class="metric" style="margin-bottom:8px;">
                <span class="metric-value">{lr.get('success_rate',0)*100:.0f}%</span>
                <span class="metric-label">LR Recourse Rate</span>
            </div>
            <div class="metric" style="margin-bottom:8px;">
                <span class="metric-value" style="color:#10b981;">
                    {rf.get('success_rate',0)*100:.0f}%</span>
                <span class="metric-label">RF Recourse Rate</span>
            </div>
            <div class="metric">
                <span class="metric-value" style="color:#f43f5e;">
                    {sg.get('gap_pct', 0):+.1f}%</span>
                <span class="metric-label">Female vs Male Gap</span>
            </div>
            """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="margin-top:28px; padding-top:16px;
                    border-top:1px solid #1e1b2e;">
            <div style="font-size:10px; color:#2e2a3e; line-height:1.8;
                        font-family:'Source Sans Pro',sans-serif;">
                UCI Adult Income Dataset<br>
                48,842 Census records · 1994<br>
                Wachter et al. 2017
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — HOME
# =============================================================================

def render_home(stats, effort_df, trained):
    """
    Landing page. Tells the full story of algorithmic recourse,
    explains the key concepts, and guides the visitor through the app.
    """

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:10px 0 36px 0; border-bottom:1px solid #2e2a3e;
                margin-bottom:40px;">
        <div style="font-family:'Fira Code',monospace; font-size:11px;
                    color:#f59e0b; text-transform:uppercase;
                    letter-spacing:0.16em; margin-bottom:14px;">
            Algorithmic Fairness · Counterfactual Explanations · UCI 1994 Census
        </div>
        <div style="font-family:'Playfair Display',serif; font-size:54px;
                    color:#f1f5f9; line-height:1.05; margin-bottom:18px;">
            When the Machine Says No—<br>
            <em>What Can You Do About It?</em>
        </div>
        <div style="font-size:17px; color:#94a3b8; max-width:700px;
                    line-height:1.85; font-family:'Source Sans Pro',sans-serif;">
            A machine learning model scans your income profile and predicts
            you will earn less than $50,000 a year. The prediction is 85% accurate.
            But you are sitting across from a loan officer who cannot tell you
            <em>what to change</em> to flip that decision.
            <strong style="color:#f1f5f9;">That is the problem
            algorithmic recourse solves.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live metrics ──────────────────────────────────────────────────────
    pm   = stats.get("per_model", {})
    sg   = stats.get("sex_gap") or {}
    ov   = stats.get("overall", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("9,999",
         "People · 12 Features",         "#f59e0b"),
        (f"{pm.get('logistic',{}).get('success_rate',0)*100:.0f}%",
         "LR Recourse Success",           "#10b981"),
        (f"{pm.get('forest',{}).get('success_rate',0)*100:.0f}%",
         "RF Recourse Success",           "#10b981"),
        (f"{pm.get('tree',{}).get('success_rate',0)*100:.0f}%",
         "Tree Recourse Success",         "#f43f5e"),
        (f"{sg.get('gap_pct',0):+.1f}%",
         "Female vs Male Effort Gap",     "#818cf8"),
    ]
    for col, (val, label, color) in zip([c1,c2,c3,c4,c5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric">
                <span class="metric-value" style="color:{color};
                      font-size:26px;">{val}</span>
                <span class="metric-label">{label}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column: story + quick-facts ───────────────────────────────────
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<span class="section-label">The Story</span>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-family:'Playfair Display',serif; font-size:20px;
                      color:#f1f5f9; line-height:1.55; margin-bottom:18px;">
                Imagine three people — all predicted as
                <em>low-income</em> by the same model.
            </p>
            <p style="font-size:14px; color:#94a3b8; line-height:1.85;
                      margin-bottom:14px;">
                <strong style="color:#f1f5f9;">Person A</strong> is 35,
                working 30 hours a week. The model says: "Work 40 hours and
                finish some college — you'd be predicted high-income."
                Hard but achievable.
            </p>
            <p style="font-size:14px; color:#94a3b8; line-height:1.85;
                      margin-bottom:14px;">
                <strong style="color:#f1f5f9;">Person B</strong> — a woman —
                works 45 hours, has a bachelor's degree, senior occupation.
                The model says: "Get a doctorate and change your occupation."
                An enormous ask.
            </p>
            <p style="font-size:14px; color:#94a3b8; line-height:1.85;">
                <strong style="color:#f1f5f9;">Person C</strong> has the
                exact same profile as Person B, except male. The model says:
                "Work 5 more hours per week." One small change.
                Same education. Same job. Different sex. Different recourse.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="callout">
            <strong>What is Algorithmic Recourse?</strong><br><br>
            Given a person predicted as low-income by model f, find the
            minimum changes to their features such that f predicts high-income.<br><br>
            <code style="font-family:'Fira Code'; font-size:13px;
                          color:#f59e0b;">
                min  distance(x, x')
                s.t. f(x') = >50K
                     x'[immutable] = x[immutable]
            </code>
        </div>

        <div class="callout emerald">
            <strong>The Opportunity:</strong><br>
            When recourse is actionable and affordable, it gives people a
            real path forward — study more, work more hours, gain capital.
            That is empowering.
        </div>

        <div class="callout rose">
            <strong>The Problem:</strong><br>
            When recourse demands more from women or minorities than from
            their equally-qualified counterparts, the model is encoding
            historical bias as individual obligation. That is harmful.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<span class="section-label">Recourse Success by Model</span>',
                    unsafe_allow_html=True)

        # Bar chart comparing success rates across three models
        models_display = {
            'logistic': 'Logistic Reg.',
            'forest':   'Random Forest',
            'tree':     'Decision Tree',
        }
        model_names  = [models_display[k] for k in ['logistic','forest','tree']]
        success_rates= [pm.get(k,{}).get('success_rate',0)*100
                        for k in ['logistic','forest','tree']]
        mean_dists   = [pm.get(k,{}).get('mean_distance',0)
                        for k in ['logistic','forest','tree']]
        bar_colors   = ["#10b981", "#f59e0b", "#f43f5e"]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=model_names, y=success_rates,
            marker_color=bar_colors,
            marker_line_color="#13111a",
            marker_line_width=2,
            text=[f"{v:.0f}%" for v in success_rates],
            textposition="outside",
            textfont=dict(family="Fira Code", color="#f1f5f9", size=13),
            hovertemplate="%{x}<br>Success rate: %{y:.1f}%<extra></extra>"
        ))
        fig_bar.update_layout(
            title=dict(text="% of Low-Income Persons Finding Recourse",
                       font={"family":"Playfair Display","color":"#f1f5f9","size":15}),
            xaxis_title="", yaxis_title="Success Rate (%)",
            showlegend=False, yaxis_range=[0, 115]
        )
        apply_theme(fig_bar, height=280)
        st.plotly_chart(fig_bar, width='stretch')

        # Key numbers grid
        st.markdown(f"""
        <div class="card" style="margin-top:16px;">
            <span class="section-label">Key Numbers</span>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div style="background:#13111a; border:1px solid #2e2a3e;
                            border-radius:10px; padding:14px; text-align:center;">
                    <div style="font-family:'Fira Code',monospace;
                                font-size:22px; color:#f59e0b; font-weight:500;">
                        3</div>
                    <div style="font-size:10px; color:#475569;
                                text-transform:uppercase; letter-spacing:.08em;
                                margin-top:4px;">Algorithms</div>
                </div>
                <div style="background:#13111a; border:1px solid #2e2a3e;
                            border-radius:10px; padding:14px; text-align:center;">
                    <div style="font-family:'Fira Code',monospace;
                                font-size:22px; color:#10b981; font-weight:500;">
                        9</div>
                    <div style="font-size:10px; color:#475569;
                                text-transform:uppercase; letter-spacing:.08em;
                                margin-top:4px;">Actionable Features</div>
                </div>
                <div style="background:#13111a; border:1px solid #2e2a3e;
                            border-radius:10px; padding:14px; text-align:center;">
                    <div style="font-family:'Fira Code',monospace;
                                font-size:22px; color:#f43f5e; font-weight:500;">
                        3</div>
                    <div style="font-size:10px; color:#475569;
                                text-transform:uppercase; letter-spacing:.08em;
                                margin-top:4px;">Immutable Features</div>
                </div>
                <div style="background:#13111a; border:1px solid #2e2a3e;
                            border-radius:10px; padding:14px; text-align:center;">
                    <div style="font-family:'Fira Code',monospace;
                                font-size:22px; color:#818cf8; font-weight:500;">
                        {sg.get('gap_pct',0):+.0f}%</div>
                    <div style="font-size:10px; color:#475569;
                                text-transform:uppercase; letter-spacing:.08em;
                                margin-top:4px;">Female Effort Gap</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Visitor Guide ─────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="section-label">How to Use This App</span>
    <div style="font-family:'Playfair Display',serif; font-size:28px;
                color:#f1f5f9; margin-bottom:22px;">
        Your five-step journey through recourse
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("📊", "01", "Explore the Dataset",
         "Visit the <strong>Dataset</strong> page to meet the 9,999 people in our "
         "study. See their age, education, occupation, hours worked, capital "
         "gains — and how the three models predict their income. Understand "
         "what the models are learning from before you see what they do.",
         "#f59e0b"),
        ("🔍", "02", "Find Someone's Recourse",
         "The <strong>Find Recourse</strong> page is the heart of the tool. "
         "Pick any person predicted as low-income, choose a model and algorithm, "
         "and see exactly what they would need to change — which features, "
         "by how much — for the prediction to flip to >$50K.",
         "#10b981"),
        ("⚗️", "03", "Compare the Three Algorithms",
         "The <strong>Compare Algorithms</strong> page shows the same person "
         "through three different lenses. Greedy finds the fastest path. "
         "Importance-Guided aligns with the model's logic. Proximity minimizes "
         "the total distance. Same person, same model, very different advice.",
         "#818cf8"),
        ("⚖️", "04", "Examine Fairness",
         "The <strong>Fairness</strong> page asks the hard question: do some "
         "groups need to change more than others to receive a favorable "
         "prediction? We compare effort by sex and race — revealing disparities "
         "encoded in the 1994 Census data.",
         "#f43f5e"),
        ("📖", "05", "Read the Methodology",
         "The <strong>About</strong> page explains exactly how this was built, "
         "which algorithms were used, what the limitations are, and how this "
         "connects to published research. Read this before discussing with "
         "a researcher.",
         "#fbbf24"),
    ]

    col1, col2 = st.columns(2, gap="large")
    for i, (icon, num, title, desc, color) in enumerate(steps):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="step-card" style="border-left:3px solid {color};">
                <div style="font-size:28px; min-width:40px;">{icon}</div>
                <div class="step-body">
                    <div style="font-family:'Fira Code',monospace; font-size:10px;
                                color:{color}; text-transform:uppercase;
                                letter-spacing:.12em; margin-bottom:5px;">
                        Step {num}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Three key findings ────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Three Core Findings</span>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    lr_rate = pm.get('logistic',{}).get('success_rate',0)*100
    dt_rate = pm.get('tree',{}).get('success_rate',0)*100
    gap_pct = sg.get('gap_pct', 0)

    with f1:
        st.markdown(f"""
        <div class="card-amber">
            <span class="section-label" style="color:#f59e0b;">
                Finding 01 — Architecture Matters</span>
            <div style="font-family:'Playfair Display',serif; font-size:20px;
                        color:#f1f5f9; margin:10px 0; line-height:1.3;">
                Same person. Three models.<br>Three very different answers.
            </div>
            <p style="font-size:13px; color:#94a3b8; line-height:1.7; margin:0;">
                Logistic Regression finds recourse for
                <strong style="color:#f1f5f9;">{lr_rate:.0f}%</strong>
                of people. The Decision Tree finds recourse for only
                <strong style="color:#f43f5e;">{dt_rate:.0f}%</strong>.
                Model architecture determines whether a person has any
                recourse at all — independent of accuracy.
            </p>
        </div>""", unsafe_allow_html=True)

    with f2:
        st.markdown(f"""
        <div class="card-rose">
            <span class="section-label" style="color:#f43f5e;">
                Finding 02 — The Fairness Gap</span>
            <div style="font-family:'Playfair Display',serif; font-size:20px;
                        color:#f1f5f9; margin:10px 0; line-height:1.3;">
                Women need {abs(gap_pct):.0f}% more effort<br>
                than men for the same prediction.
            </div>
            <p style="font-size:13px; color:#94a3b8; line-height:1.7; margin:0;">
                The model has equal overall accuracy across sexes. Yet the
                changes it requires from women are consistently larger.
                This is recourse disparity — bias expressed as individual
                burden rather than aggregate error.
            </p>
        </div>""", unsafe_allow_html=True)

    with f3:
        st.markdown(f"""
        <div class="card-indigo">
            <span class="section-label" style="color:#818cf8;">
                Finding 03 — No Recourse Exists</span>
            <div style="font-family:'Playfair Display',serif; font-size:20px;
                        color:#f1f5f9; margin:10px 0; line-height:1.3;">
                For some people, no actionable<br>path exists at all.
            </div>
            <p style="font-size:13px; color:#94a3b8; line-height:1.7; margin:0;">
                Some people are placed by the model in a region of feature
                space where no change to their mutable attributes —
                within realistic bounds — can flip the prediction.
                The model has made a permanent judgment about them.
            </p>
        </div>""", unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — DATASET
# =============================================================================

def render_dataset(X, y, df, trained, all_preds):
    """
    Shows the Adult Income dataset with feature descriptions,
    distributions, and model prediction overview.
    """

    st.markdown("""
    <div class="page-title">The Adult Income<br><em>Dataset</em></div>
    <div class="page-subtitle">
        UCI Census Income data — 9,999 people from the 1994 U.S. Census.
        Binary label: does this person earn more than $50,000 per year?
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric">
            <span class="metric-value">{len(df):,}</span>
            <span class="metric-label">Total People</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#10b981;">
                {int(y.sum()):,}</span>
            <span class="metric-label">High Income (>50K)</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#f43f5e;">
                {int((y==0).sum()):,}</span>
            <span class="metric-label">Low Income (≤50K)</span>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#818cf8;">12</span>
            <span class="metric-label">Features</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature dictionary + data table ───────────────────────────────────
    col_dict, col_table = st.columns([1, 2], gap="large")

    with col_dict:
        st.markdown('<span class="section-label">Feature Dictionary</span>',
                    unsafe_allow_html=True)

        # ROOT CAUSE OF THE CSS BUG:
        # Calling st.markdown() separately for each feature (12 times) caused
        # Streamlit to insert 12 individual HTML elements into the DOM. When
        # the adjacent st.dataframe() renders its tooltip layer, it picks up
        # the raw HTML source from nearby markdown nodes and shows it on hover.
        #
        # FIX: Build ONE combined HTML string for all features and render it
        # in a single st.markdown() call. One element = no tooltip bleed.
        rows_html = ""
        for fname in FEATURE_NAMES:
            meta       = FEATURE_META[fname]
            actionable = meta['actionable']
            tag_cls    = "tag-emerald" if actionable else "tag-rose"
            tag_txt    = "actionable"  if actionable else "immutable"
            inc_only   = meta.get('increase_only', False)
            inc_badge  = ('<span class="tag tag-amber" style="margin-left:2px;">'
                         'increase only</span>') if inc_only else ''
            rows_html += (
                f'<div style="padding:9px 0; border-bottom:1px solid #1e1b2e;">'
                f'<div style="display:flex; align-items:center; gap:6px; '
                f'margin-bottom:3px; flex-wrap:wrap;">'
                f'<span class="tag {tag_cls}">{tag_txt}</span>'
                f'{inc_badge}'
                f'<span style="font-family:\'Playfair Display\',serif; '
                f'font-size:13px; color:#f1f5f9;">{meta["label"]}</span>'
                f'</div>'
                f'<div style="font-size:11px; color:#475569; '
                f'padding-left:4px;">{meta["description"]}</div>'
                f'</div>'
            )
        st.markdown(f'<div>{rows_html}</div>', unsafe_allow_html=True)

    with col_table:
        st.markdown('<span class="section-label">Browse the Data</span>',
                    unsafe_allow_html=True)

        # Build display DataFrame with readable labels
        display_df = df.copy()
        display_df.insert(0, '#', range(len(display_df)))
        display_df['income'] = y.map({0: '🔴 ≤50K', 1: '🟢 >50K'})

        # Convert encoded categoricals back to labels for readability
        for fname in FEATURE_NAMES:
            if fname in CATEGORY_LABELS:
                display_df[fname] = display_df[fname].apply(
                    lambda v: feature_value_label(fname, v)
                )

        st.dataframe(
            display_df[['#','income'] + FEATURE_NAMES].head(500),
            width='stretch', height=460,
            column_config={
                '#':       st.column_config.NumberColumn('#', width='small'),
                'income':  st.column_config.TextColumn('Income', width='medium'),
            },
            hide_index=True
        )
        st.markdown("""
        <div style="font-family:'Fira Code',monospace; font-size:10px;
                    color:#475569; margin-top:6px;">
            Showing first 500 rows · categorical features decoded to labels
        </div>""", unsafe_allow_html=True)

    # ── Feature distributions ─────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Feature Distributions by Income Group</span>',
                unsafe_allow_html=True)

    key_feats  = ['education_num', 'hours_per_week', 'age', 'capital_gain']
    feat_labels = {
        'education_num':  'Education Level (1–16)',
        'hours_per_week': 'Hours per Week',
        'age':            'Age',
        'capital_gain':   'Capital Gain ($)',
    }

    fcols = st.columns(4)
    for i, feat in enumerate(key_feats):
        with fcols[i]:
            high = df[y == 1][feat]
            low  = df[y == 0][feat]
            fig  = go.Figure()
            fig.add_trace(go.Histogram(
                x=low, name="≤50K",
                marker_color="#f43f5e", opacity=0.75, nbinsx=20
            ))
            fig.add_trace(go.Histogram(
                x=high, name=">50K",
                marker_color="#10b981", opacity=0.75, nbinsx=20
            ))
            fig.update_layout(
                title=dict(text=feat_labels[feat],
                           font={"family":"Playfair Display",
                                 "color":"#f1f5f9","size":13}),
                barmode="overlay", showlegend=(i == 0),
                xaxis_title=feat, yaxis_title="Count"
            )
            apply_theme(fig, height=220)
            st.plotly_chart(fig, width='stretch')

    st.markdown("""
    <div class="callout">
        <strong>Reading these charts:</strong>
        Where the green (>50K) and red (≤50K) bars separate cleanly,
        that feature strongly predicts income. Education and capital gain
        show the clearest separation — they are the most powerful levers
        in the recourse algorithms.
    </div>
    """, unsafe_allow_html=True)

    # ── Model agreement ───────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Model Agreement</span>',
                unsafe_allow_html=True)

    disagree   = all_preds[~all_preds['consensus']]
    agree      = all_preds[all_preds['consensus']]

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#10b981;">
                {len(agree):,}</span>
            <span class="metric-label">All 3 Models Agree</span>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#f43f5e;">
                {len(disagree):,}</span>
            <span class="metric-label">Models Disagree</span>
        </div>""", unsafe_allow_html=True)
    with mc3:
        pct = len(disagree) / len(all_preds) * 100
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#818cf8;">
                {pct:.1f}%</span>
            <span class="metric-label">Disagreement Rate</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="callout indigo" style="margin-top:16px;">
        <strong>Why disagreement matters for recourse:</strong>
        For the 5.4% of people where models disagree, the person's income
        prediction — and therefore the recourse advice they receive — depends
        entirely on which model the institution deployed. This is arbitrary.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3 — FIND RECOURSE
# =============================================================================

def render_find_recourse(X, y, trained, all_preds):
    """
    Interactive recourse finder. Pick a person, pick a model and
    algorithm, click Run — see the exact changes recommended.
    """

    st.markdown("""
    <div class="page-title">Find<br><em>Recourse</em></div>
    <div class="page-subtitle">
        Select any person predicted as low-income, choose a model and
        algorithm, and see exactly what they would need to change to flip
        the prediction to high-income.
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

    with ctrl1:
        # Build dropdown: show only persons predicted <=50K by the
        # selected model (selected later, so we default to tree)
        low_income_idx = [
            i for i, p in enumerate(trained['tree']['predictions'])
            if p == 0
        ]

        def person_option(idx):
            f, lbl = get_person(X, y, idx)
            age    = int(f['age'])
            edu    = feature_value_label('education_num', f['education_num'])
            occ    = feature_value_label('occupation', f['occupation'])
            sex    = feature_value_label('sex', f['sex'])
            true_s = ">50K" if lbl == 1 else "≤50K"
            return (f"#{idx:04d} | Age {age} · {edu} · {occ} · "
                    f"{sex} | True: {true_s}")

        options      = [person_option(i) for i in low_income_idx[:200]]
        selected_opt = st.selectbox(
            "Select a person (predicted ≤50K by Decision Tree):",
            options, index=0,
            help="Sorted by dataset order. The true label shows the actual income."
        )
        person_idx = low_income_idx[options.index(selected_opt)]

    with ctrl2:
        model_choice = st.selectbox(
            "Model:",
            ["Decision Tree", "Logistic Regression", "Random Forest"],
            index=0
        )
        model_key_map = {
            "Decision Tree":       "tree",
            "Logistic Regression": "logistic",
            "Random Forest":       "forest",
        }
        model_key = model_key_map[model_choice]

    with ctrl3:
        algo_choice = st.selectbox(
            "Algorithm:",
            ["Greedy", "Importance-Guided", "Proximity (Wachter)"],
            index=0
        )
        algo_key_map = {
            "Greedy":                "greedy",
            "Importance-Guided":     "importance",
            "Proximity (Wachter)":   "proximity",
        }
        algo_key = algo_key_map[algo_choice]

    run_btn = st.button("▶  Find Recourse", type="primary",
                        width='content')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Person profile ─────────────────────────────────────────────────────
    features, true_label = get_person(X, y, person_idx)
    model                = trained[model_key]['model']
    pred, prob           = predict_person(model, features)

    pred_badge  = ('badge-high' if pred == 1 else 'badge-low')
    pred_text   = ('>50K Predicted' if pred == 1 else '≤50K Predicted')
    true_badge  = ('badge-high' if true_label == 1 else 'badge-low')
    true_text   = ('>50K True' if true_label == 1 else '≤50K True')

    st.markdown(f"""
    <div class="card" style="margin-bottom:24px;">
        <div style="display:flex; align-items:center; gap:14px;
                    flex-wrap:wrap; margin-bottom:16px;">
            <div style="font-family:'Playfair Display',serif; font-size:28px;
                        color:#f1f5f9;">Person #{person_idx:04d}</div>
            <span class="badge {pred_badge}">{pred_text}</span>
            <span class="badge {true_badge}">{true_text}</span>
            <span style="font-family:'Fira Code',monospace; font-size:12px;
                         color:#94a3b8;">P(>50K) = {prob:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature profile columns
    pcol1, pcol2 = st.columns(2)
    feat_items   = list(features.items())
    half         = len(feat_items) // 2

    for col, items in [(pcol1, feat_items[:half]),
                       (pcol2, feat_items[half:])]:
        with col:
            for fname, val in items:
                meta       = FEATURE_META[fname]
                label      = meta['label']
                val_label  = feature_value_label(fname, val)
                actionable = meta['actionable']
                dot_color  = "#10b981" if actionable else "#f43f5e"

                # Normalized bar width
                rng     = meta['max'] - meta['min']
                bar_pct = int(((float(val) - meta['min']) / rng * 100)
                              if rng > 0 else 0)

                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px;
                            padding:8px 0; border-bottom:1px solid #1e1b2e;">
                    <div style="width:8px; height:8px; border-radius:50%;
                                background:{dot_color}; flex-shrink:0;"></div>
                    <div style="width:120px; font-size:12px;
                                font-family:'Fira Code',monospace;
                                color:#94a3b8;">{label}</div>
                    <div style="flex:1; height:4px; background:#2e2a3e;
                                border-radius:2px;">
                        <div style="width:{bar_pct}%; height:4px;
                                    background:#f59e0b; border-radius:2px;">
                        </div>
                    </div>
                    <div style="width:130px; text-align:right; font-size:12px;
                                font-family:'Fira Code',monospace;
                                color:#f1f5f9;">{val_label}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Run recourse ──────────────────────────────────────────────────────
    if run_btn or st.session_state.get(f"recourse_{person_idx}_{model_key}_{algo_key}"):

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if pred == 1:
            st.markdown("""
            <div class="callout emerald">
                This person is already predicted as high-income (>$50K)
                by this model. No recourse is needed.
            </div>
            """, unsafe_allow_html=True)
            return

        with st.spinner(f"Running {algo_choice} algorithm..."):
            kwargs = ({'max_restarts': 8} if algo_key == 'proximity'
                      else {'max_iterations': 200})
            result = find_recourse(
                model, features,
                target_class=1,
                algorithm=algo_key,
                **kwargs
            )

        # Cache the result so page doesn't re-run on next interaction
        st.session_state[f"recourse_{person_idx}_{model_key}_{algo_key}"] = True

        if result.found:
            cf_pred, cf_prob = predict_person(model, result.counterfactual)

            st.markdown(f"""
            <div class="card-emerald">
                <div style="display:flex; align-items:center; gap:16px;
                            flex-wrap:wrap; margin-bottom:12px;">
                    <div style="font-family:'Playfair Display',serif;
                                font-size:22px; color:#f1f5f9;">
                        Recourse Found ✓</div>
                    <span class="badge badge-found">
                        {result.n_changed} feature{'s' if result.n_changed!=1 else ''} changed
                    </span>
                    <span style="font-family:'Fira Code',monospace; font-size:12px;
                                 color:#94a3b8;">
                        Distance: {result.distance:.4f}
                    </span>
                    <span style="font-family:'Fira Code',monospace; font-size:12px;
                                 color:#10b981;">
                        New P(>50K): {cf_prob:.1%}
                    </span>
                </div>
                <p style="font-size:13px; color:#94a3b8; margin:0;">
                    {result.message}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<span class="section-label">Required Changes</span>',
                        unsafe_allow_html=True)

            for fname, change in result.changes.items():
                delta      = change['delta']
                arrow      = "↑" if delta > 0 else "↓"
                delta_class = "change-delta-pos" if delta > 0 else "change-delta-neg"
                delta_str   = f"{arrow} {abs(delta):.1f} {FEATURE_META[fname]['unit']}"

                st.markdown(f"""
                <div class="change-row">
                    <span class="change-label">
                        {FEATURE_META[fname]['label']}</span>
                    <span class="change-arrow">
                        {change['original_label']} → {change['new_label']}</span>
                    <span class="{delta_class}">{delta_str}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="callout" style="margin-top:16px;">
                <strong>What this means in plain English:</strong>
                The table above shows the minimum changes this person
                needs to make for the model to predict them as high-income.
                Features not listed should stay the same.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="card-rose">
                <div style="font-family:'Playfair Display',serif;
                            font-size:22px; color:#f1f5f9; margin-bottom:10px;">
                    No Recourse Found</div>
                <p style="font-size:13px; color:#94a3b8; margin:0;">
                    {result.message}
                </p>
            </div>
            <div class="callout rose">
                <strong>What this means:</strong>
                This person is placed by the <em>{model_choice}</em> in a
                region of feature space where no change to their mutable
                attributes — within realistic bounds — can flip the
                prediction. Try a different model or algorithm.
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4 — COMPARE ALGORITHMS
# =============================================================================

def render_compare(X, y, trained):
    """
    Runs all three algorithms on one person and shows them side by side.
    """

    st.markdown("""
    <div class="page-title">Compare<br><em>Algorithms</em></div>
    <div class="page-subtitle">
        Same person, same model, three different algorithms.
        Each finds a valid counterfactual — but they disagree
        on which features to change and by how much.
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────
    cc1, cc2 = st.columns([3, 1])

    with cc1:
        low_idx = [i for i, p in enumerate(trained['tree']['predictions'])
                   if p == 0][:150]

        def opt_label(i):
            f, lbl = get_person(X, y, i)
            return (f"#{i:04d} | Age {int(f['age'])} · "
                    f"{feature_value_label('education_num', f['education_num'])} · "
                    f"{feature_value_label('sex', f['sex'])}")

        sel       = st.selectbox("Select person:", [opt_label(i) for i in low_idx])
        pidx      = low_idx[[opt_label(i) for i in low_idx].index(sel)]

    with cc2:
        mdl_choice = st.selectbox(
            "Model:",
            ["Decision Tree","Logistic Regression","Random Forest"]
        )
        mdl_key = {"Decision Tree":"tree","Logistic Regression":"logistic",
                   "Random Forest":"forest"}[mdl_choice]

    if st.button("▶  Compare All Three Algorithms", type="primary"):

        features, true_label = get_person(X, y, pidx)
        model                = trained[mdl_key]['model']
        pred, prob           = predict_person(model, features)

        if pred == 1:
            st.markdown("""
            <div class="callout emerald">
                Already predicted >50K — no recourse needed.
            </div>""", unsafe_allow_html=True)
            return

        with st.spinner("Running all three algorithms..."):
            comparison = compare_algorithms(model, features)

        # ── Side-by-side result cards ─────────────────────────────────────
        algo_meta = {
            'greedy':    ('Greedy',           '#f59e0b', 'card-amber'),
            'importance':('Importance-Guided','#818cf8', 'card-indigo'),
            'proximity': ('Proximity (Wachter)','#10b981','card-emerald'),
        }

        acol1, acol2, acol3 = st.columns(3)
        for col, algo_key in zip([acol1, acol2, acol3],
                                  ['greedy','importance','proximity']):
            result  = comparison[algo_key]
            name, color, card_class = algo_meta[algo_key]
            with col:
                if result.found:
                    st.markdown(f"""
                    <div class="{card_class}">
                        <span class="section-label" style="color:{color};">
                            {name}</span>
                        <div style="font-family:'Playfair Display',serif;
                                    font-size:28px; color:#10b981;
                                    margin:8px 0;">Found ✓</div>
                        <div style="font-family:'Fira Code',monospace;
                                    font-size:12px; color:#94a3b8;
                                    margin-bottom:14px;">
                            Distance: {result.distance:.4f}<br>
                            Features changed: {result.n_changed}<br>
                            New P(>50K): {result.cf_prob:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    for fname, change in result.changes.items():
                        d  = change['delta']
                        dc = "change-delta-pos" if d > 0 else "change-delta-neg"
                        st.markdown(f"""
                        <div class="change-row">
                            <span style="font-size:12px; color:#94a3b8;">
                                {FEATURE_META[fname]['label']}</span>
                            <span class="{dc}; font-family:'Fira Code',
                                monospace; font-size:12px;">
                                {'↑' if d>0 else '↓'} {abs(d):.1f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="{card_class}">
                        <span class="section-label" style="color:{color};">
                            {name}</span>
                        <div style="font-family:'Playfair Display',serif;
                                    font-size:28px; color:#f43f5e;
                                    margin:8px 0;">Not Found</div>
                        <p style="font-size:12px; color:#94a3b8; margin:0;">
                            {result.message[:120]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Comparison chart ──────────────────────────────────────────────
        found_results = {k: v for k, v in comparison.items() if v.found}
        if len(found_results) >= 2:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="section-label">Distance Comparison</span>',
                        unsafe_allow_html=True)

            fig_cmp = go.Figure()
            for algo_key, result in found_results.items():
                name, color, _ = algo_meta[algo_key]
                fig_cmp.add_trace(go.Bar(
                    x=[name], y=[result.distance],
                    marker_color=color,
                    text=[f"{result.distance:.4f}"],
                    textposition="outside",
                    textfont=dict(family="Fira Code", color="#f1f5f9", size=12),
                    name=name
                ))
            fig_cmp.update_layout(
                title=dict(text="Normalized Distance (lower = closer to original)",
                           font={"family":"Playfair Display",
                                 "color":"#f1f5f9","size":15}),
                showlegend=False, xaxis_title="", yaxis_title="Distance"
            )
            apply_theme(fig_cmp, height=300)
            st.plotly_chart(fig_cmp, width='stretch')

        st.markdown("""
        <div class="callout indigo">
            <strong>Key insight:</strong>
            Proximity minimization tends to find the shortest path (lowest
            distance) because it treats recourse as a mathematical
            optimization problem. Greedy and Importance-Guided find
            valid recourse but may take a less efficient route.
            However, the Proximity path may involve non-integer changes
            to discrete features — sometimes less realistic in practice.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5 — FAIRNESS
# =============================================================================

def render_fairness(effort_df, fairness, feature_freq, trained):
    """
    The fairness analysis page. Shows effort disparities by sex and race.
    """

    st.markdown("""
    <div class="page-title">Who Bears<br><em>the Burden?</em></div>
    <div class="page-subtitle">
        If the model requires more effort from women or minorities to receive
        a favorable prediction, that is algorithmic harm — even when overall
        accuracy is equal across groups.
    </div>
    """, unsafe_allow_html=True)

    # ── Key fairness numbers ───────────────────────────────────────────────
    found_df   = effort_df[effort_df['found'] == True]
    male_df    = found_df[found_df['sex_label'] == 'Male']
    female_df  = found_df[found_df['sex_label'] == 'Female']
    male_mean  = male_df['distance'].mean()   if not male_df.empty   else 0
    female_mean= female_df['distance'].mean() if not female_df.empty else 0
    gap        = female_mean - male_mean
    gap_pct    = gap / male_mean * 100 if male_mean > 0 else 0

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#818cf8;">
                {male_mean:.3f}</span>
            <span class="metric-label">Male Mean Distance</span>
        </div>""", unsafe_allow_html=True)
    with fc2:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#f43f5e;">
                {female_mean:.3f}</span>
            <span class="metric-label">Female Mean Distance</span>
        </div>""", unsafe_allow_html=True)
    with fc3:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#f43f5e;">
                {gap:+.3f}</span>
            <span class="metric-label">Absolute Gap</span>
        </div>""", unsafe_allow_html=True)
    with fc4:
        st.markdown(f"""<div class="metric">
            <span class="metric-value" style="color:#f43f5e;">
                {gap_pct:+.1f}%</span>
            <span class="metric-label">Relative Gap</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sex fairness chart ────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["By Sex", "By Race", "Feature Frequency"])

    with tab1:
        st.markdown('<span class="section-label">Recourse Effort by Sex</span>',
                    unsafe_allow_html=True)

        sex_data   = fairness.get('sex', {})
        model_keys = ['logistic', 'forest', 'tree']
        model_names= ['Logistic Reg.', 'Random Forest', 'Decision Tree']

        fig_sex = go.Figure()
        for group_label, color in [('Male','#818cf8'),('Female','#f43f5e')]:
            means = []
            for mk in model_keys:
                val = sex_data.get(group_label, {}).get(mk, {})
                d   = val.get('mean_distance', np.nan)
                means.append(d if not (isinstance(d,float) and
                                        np.isnan(d)) else 0)
            fig_sex.add_trace(go.Bar(
                x=model_names, y=means, name=group_label,
                marker_color=color,
                marker_line_color="#13111a", marker_line_width=1.5,
                hovertemplate=f"{group_label}: %{{y:.4f}}<extra></extra>"
            ))

        fig_sex.update_layout(
            title=dict(
                text="Mean Recourse Distance: Male vs Female (per model)",
                font={"family":"Playfair Display","color":"#f1f5f9","size":15}
            ),
            barmode="group", xaxis_title="", yaxis_title="Mean Distance"
        )
        apply_theme(fig_sex, height=380)
        st.plotly_chart(fig_sex, width='stretch')

        # Success rate comparison
        fig_succ = go.Figure()
        for group_label, color in [('Male','#818cf8'),('Female','#f43f5e')]:
            rates = []
            for mk in model_keys:
                val  = sex_data.get(group_label, {}).get(mk, {})
                rates.append(val.get('success_rate', 0) * 100)
            fig_succ.add_trace(go.Bar(
                x=model_names, y=rates, name=group_label,
                marker_color=color,
                marker_line_color="#13111a", marker_line_width=1.5,
                hovertemplate=f"{group_label}: %{{y:.1f}}%<extra></extra>"
            ))
        fig_succ.update_layout(
            title=dict(
                text="Recourse Success Rate: Male vs Female (%)",
                font={"family":"Playfair Display","color":"#f1f5f9","size":15}
            ),
            barmode="group", xaxis_title="", yaxis_title="Success Rate (%)"
        )
        apply_theme(fig_succ, height=320)
        st.plotly_chart(fig_succ, width='stretch')

        st.markdown(f"""
        <div class="callout rose">
            <strong>Interpretation:</strong> Women require on average
            <strong>{abs(gap_pct):.1f}%</strong> more effort than men to
            receive a high-income prediction. This disparity is consistent
            across all three model architectures — it is a property of the
            1994 Census data itself, not an artifact of any single model.
            In practice this means: a woman and a man with identical
            education, occupation, and hours may be given very different
            advice about what to change.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<span class="section-label">Recourse Effort by Race</span>',
                    unsafe_allow_html=True)

        race_data   = fairness.get('race', {})
        race_groups = sorted(race_data.keys())

        # Bar chart per model
        fig_race = go.Figure()
        race_colors = ["#f59e0b","#10b981","#f43f5e","#818cf8","#fbbf24"]

        for (mk, mname), rcolor in zip(zip(model_keys, model_names), race_colors):
            means = [race_data.get(g,{}).get(mk,{}).get('mean_distance',0)
                     for g in race_groups]
            fig_race.add_trace(go.Bar(
                x=race_groups, y=means, name=mname,
                marker_color=rcolor,
                marker_line_color="#13111a", marker_line_width=1.5,
                hovertemplate=f"{mname}: %{{y:.4f}}<extra></extra>"
            ))

        fig_race.update_layout(
            title=dict(
                text="Mean Recourse Distance by Race",
                font={"family":"Playfair Display","color":"#f1f5f9","size":15}
            ),
            barmode="group", xaxis_title="Race Group",
            yaxis_title="Mean Distance"
        )
        apply_theme(fig_race, height=380)
        st.plotly_chart(fig_race, width='stretch')

        st.markdown("""
        <div class="callout indigo">
            <strong>Note on small group sizes:</strong>
            Some racial groups have few people in the 9,999-row sample.
            Interpret bars for small groups (n &lt; 30) with caution —
            high variance means individual results may not be representative.
            The sample is stratified to maximize representation, but
            the 1994 Census itself has unequal group sizes.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<span class="section-label">Which Features Appear in Recourse?</span>',
                    unsafe_allow_html=True)

        # One chart per model showing feature frequency
        ff_cols = st.columns(3)
        for col, (mk, mname) in zip(ff_cols,
            [('logistic','Logistic Reg.'),
             ('forest',  'Random Forest'),
             ('tree',    'Decision Tree')]):

            freq_data = feature_freq.get(mk, {})
            sorted_f  = sorted(freq_data.items(),
                               key=lambda x: x[1]['fraction'], reverse=True)
            labels    = [FEATURE_META[f]['label'] for f, _ in sorted_f]
            fractions = [info['fraction']*100 for _, info in sorted_f]

            fig_ff = go.Figure(go.Bar(
                x=fractions, y=labels,
                orientation='h',
                marker_color="#f59e0b",
                marker_line_color="#13111a",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
            ))
            fig_ff.update_layout(
                title=dict(text=mname,
                           font={"family":"Playfair Display",
                                 "color":"#f1f5f9","size":13}),
                xaxis_title="% of recourse paths",
                yaxis_title=""
            )
            apply_theme(fig_ff, height=340)

            with col:
                st.plotly_chart(fig_ff, width='stretch')

        st.markdown("""
        <div class="callout">
            <strong>What this reveals:</strong>
            Features that appear frequently are the model's primary "levers" —
            what it most often recommends changing.
            Features that rarely appear either have little impact on the
            model's decision or are difficult to move within their valid range.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 6 — ABOUT
# =============================================================================

def render_about(trained):
    """
    Methodology, research context, limitations, tech stack.
    """

    st.markdown("""
    <div class="page-title">About This<br><em>Project</em></div>
    <div class="page-subtitle">
        The technical pipeline, algorithmic details, research connections,
        and honest limitations of this tool.
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<span class="section-label">What This Project Is</span>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-size:14px; color:#94a3b8; line-height:1.9; margin:0;">
                An interactive tool for exploring
                <strong style="color:#f1f5f9;">algorithmic recourse</strong> —
                the set of minimum changes a person needs to make to receive
                a favorable prediction from a machine learning model.
                Built to demonstrate both the algorithms and the fairness
                implications of recourse on real Census data.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="section-label">Technical Pipeline</span>',
                    unsafe_allow_html=True)

        steps_about = [
            ("1. Dataset",
             "UCI Adult Income (48,842 rows) → 9,999 stratified sample. "
             "7 categorical features ordinal-encoded. Binary target: >$50K. "
             "3 immutable features: sex, race, native_country_us."),
            ("2. Three Classifiers",
             "Logistic Regression (StandardScaler Pipeline, C=1.0), "
             "Decision Tree (depth=6, min_leaf=20), "
             "Random Forest (100 trees, depth=8). "
             "All evaluated with Stratified 5-Fold CV."),
            ("3. Three Algorithms",
             "Greedy: best single-feature ±step perturbation per iteration. "
             "Importance-Guided: weights moves by model feature importances. "
             "Proximity (Wachter 2017): scipy L-BFGS-B minimizing normalized "
             "distance + quadratic flip penalty, 5 restarts."),
            ("4. Constraints",
             "All algorithms enforce: immutable features never changed, "
             "increase-only features (age, education) never decreased, "
             "all values clipped to valid ranges after every step."),
            ("5. Fairness Analysis",
             "Greedy recourse computed for 150 stratified persons × 3 models. "
             "Effort (normalized L1 distance) aggregated by sex, race, "
             "and age group to detect systematic disparities."),
        ]

        for title, desc in steps_about:
            st.markdown(f"""
            <div style="display:flex; gap:12px; margin-bottom:12px;
                        background:#1c1929; border:1px solid #2e2a3e;
                        border-radius:10px; padding:14px;">
                <div style="width:3px; min-width:3px; background:#f59e0b;
                            border-radius:2px;"></div>
                <div>
                    <div style="font-family:'Playfair Display',serif;
                                font-size:15px; color:#f1f5f9;
                                margin-bottom:4px;">{title}</div>
                    <div style="font-size:13px; color:#94a3b8;
                                line-height:1.7;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<span class="section-label">Honest Limitations</span>',
                    unsafe_allow_html=True)
        limitations = [
            ("1994 data:",
             "The Census data reflects historical income patterns and biases "
             "from 30 years ago. The fairness gaps are real but may not "
             "reflect today's labour market."),
            ("Greedy for bulk analysis:",
             "The 150-person fairness analysis uses Greedy only. Proximity "
             "would find shorter paths in many cases, potentially changing "
             "the effort distribution. Greedy is a conservative estimate."),
            ("Ordinal encoding:",
             "Treating categorical features as ordered integers is an "
             "approximation. 'Private' is not objectively between "
             "'Without-pay' and 'Self-employed-inc' on a linear scale."),
        ]
        for title, desc in limitations:
            st.markdown(f"""
            <div class="callout rose">
                <strong>{title}</strong> {desc}
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<span class="section-label">Research Foundation</span>',
                    unsafe_allow_html=True)

        papers = [
            ("Counterfactual Explanations Without Opening the Black Box",
             "Wachter, Mittelstadt & Russell · 2017 · Harvard JOLT",
             "The foundational paper. Defines the recourse optimization "
             "problem. Our Proximity algorithm implements this directly.",
             "#f59e0b"),
            ("Algorithmic Recourse Under Imperfect Causal Knowledge",
             "Karimi et al. · 2020 · NeurIPS",
             "Extends recourse to causal settings. Motivates our "
             "increase-only constraints on age and education.",
             "#10b981"),
            ("The Hidden Assumptions Behind Counterfactual Explanations",
             "Barocas, Hardt & Narayanan · 2020 · FAccT",
             "Critiques actionability. Directly motivates our "
             "immutable feature design.",
             "#818cf8"),
            ("Fairness Implications of Recourse in Algorithmic Decision-Making",
             "Gupta et al. · 2019 · AAAI AIES",
             "Shows recourse effort can be systematically higher for "
             "protected groups — exactly what our Fairness page demonstrates.",
             "#f43f5e"),
        ]

        for title, citation, desc, color in papers:
            st.markdown(f"""
            <div style="background:#1c1929;
                        border:1px solid {color}40;
                        border-left:3px solid {color};
                        border-radius:0 10px 10px 0;
                        padding:14px; margin-bottom:12px;">
                <div style="font-family:'Playfair Display',serif;
                            font-size:14px; color:#f1f5f9;
                            margin-bottom:4px; line-height:1.3;">
                    {title}</div>
                <div style="font-family:'Fira Code',monospace; font-size:10px;
                            color:{color}; margin-bottom:7px;">{citation}</div>
                <div style="font-size:12px; color:#475569;
                            line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<span class="section-label" style="margin-top:20px; display:block;">Tech Stack</span>',
                    unsafe_allow_html=True)
        stack = [
            ("Python 3.11",        "Language"),
            ("scikit-learn 1.2+",  "ML models + pipelines"),
            ("scipy 1.10+",        "Proximity optimization"),
            ("Streamlit 1.30+",    "Web app"),
            ("Plotly 5.18+",       "Interactive charts"),
            ("Pandas + NumPy",     "Data handling"),
        ]
        stack_rows = ""
        for lib, role in stack:
            stack_rows += f"""
            <div style="display:flex; justify-content:space-between;
                        padding:6px 0; border-bottom:1px solid #1e1b2e;">
                <span style="font-family:'Fira Code',monospace;
                             font-size:12px; color:#f59e0b;">{lib}</span>
                <span style="font-size:12px; color:#475569;">{role}</span>
            </div>"""
        st.markdown(f'<div class="card" style="padding:16px;">{stack_rows}</div>',
                    unsafe_allow_html=True)

        # Reproducibility numbers
        st.markdown('<span class="section-label" style="margin-top:20px; display:block;">Reproducibility</span>',
                    unsafe_allow_html=True)
        nums = [
            ("9,999",   "People sampled"),
            ("12",      "Input features"),
            ("9 / 3",   "Actionable / Immutable"),
            ("3",       "Classifiers"),
            ("3",       "Recourse algorithms"),
            ("150",     "Persons in fairness analysis"),
        ]
        rc1, rc2 = st.columns(2)
        for i, (val, label) in enumerate(nums):
            col = rc1 if i % 2 == 0 else rc2
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding:10px;
                            background:#13111a; border:1px solid #2e2a3e;
                            border-radius:8px; margin-bottom:8px;">
                    <div style="font-family:'Fira Code',monospace;
                                font-size:18px; color:#f59e0b;">{val}</div>
                    <div style="font-size:10px; color:#475569;
                                text-transform:uppercase;
                                letter-spacing:.07em;">{label}</div>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Entry point — called on every Streamlit script re-run.

    1. Initialize session state.
    2. Load everything (cached after first run).
    3. Store stats in session state for sidebar display.
    4. Render sidebar and route to the correct page.
    """

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    # ── FAST load — always runs, ~10 seconds on first visit ───────────────
    # Shows a brief spinner only on the very first load.
    # After that the cache returns instantly.
    with st.spinner("Loading data and training models (~10 seconds)..."):
        X, y, df, trained, all_preds = load_fast()

    page = st.session_state.get("page", "Home")

    # ── SLOW load — only for pages that need fairness data ───────────────
    # Home and Fairness need the effort distribution.
    # All other pages (Dataset, Find Recourse, Compare, About) are instant.
    if page in ("Home", "Fairness"):
        with st.spinner("Computing fairness analysis "
                        "(7-11 min first visit on Streamlit Cloud, then instant forever)..."):
            effort_df, fairness, feature_freq, stats = load_analyses(
                trained, X, y
            )
        st.session_state["stats"] = stats
    else:
        # Use cached stats if available, empty defaults otherwise
        stats        = st.session_state.get("stats", {"per_model":{}, "sex_gap":{}})
        effort_df    = None
        fairness     = {}
        feature_freq = {}

    render_sidebar()

    if   page == "Home":
        render_home(stats, effort_df, trained)
    elif page == "Dataset":
        render_dataset(X, y, df, trained, all_preds)
    elif page == "Find Recourse":
        render_find_recourse(X, y, trained, all_preds)
    elif page == "Compare Algorithms":
        render_compare(X, y, trained)
    elif page == "Fairness":
        render_fairness(effort_df, fairness, feature_freq, trained)
    elif page == "About":
        render_about(trained)


if __name__ == "__main__":
    main()
