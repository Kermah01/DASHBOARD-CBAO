"""Thème « Minuit & Or » — identité visuelle du dashboard.

Centralise tout le design : palette validée (daltonisme + contraste, via le
validateur dataviz, surface #101a30), mise en forme Plotly, nuage de mots et
CSS global (glassmorphism, fond animé, typographie Google Fonts).

Ambiance : élégance bancaire premium — bleu nuit profond rehaussé d'or/ambre.
"""

from __future__ import annotations

import streamlit as st
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Couleurs de l'interface (chrome)
# ---------------------------------------------------------------------------
NUIT_FOND = "#070d1c"        # plan de page
NUIT_SURFACE = "#101a30"     # surface des cartes / graphiques (verre sur nuit)
ENCRE_PRIMAIRE = "#eef2fa"
ENCRE_SECONDAIRE = "#a9b4cc"
OR_VIF = "#e8b54d"
OR_PROFOND = "#c98500"
OR_CLAIR = "#f2cc8f"
BLEU_CLAIR = "#9ec5f4"

# ---------------------------------------------------------------------------
# Palette catégorielle — 8 crans validés sur surface #101a30 :
# bande de luminance OK, chroma OK, ΔE CVD adjacent >= 8,4, ΔE vision
# normale >= 19,3, contraste >= 3:1 (script validate_palette.js, mode dark).
# Ordre fixe = mécanisme de sécurité daltonisme : ne pas réordonner.
# ---------------------------------------------------------------------------
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# Échelle divergente pour les notes (mauvais -> neutre -> bon) :
# deux pôles chaud/froid + point médian neutre (règle dataviz).
ECHELLE_NOTES = [[0.0, "#e66767"], [0.5, "#4a5370"], [1.0, "#3987e5"]]

# Rampe séquentielle (magnitude) pour le treemap : un seul bleu, resserré pour
# garder un texte blanc lisible sur chaque cellule (3,5:1 minimum).
ECHELLE_TREEMAP = [[0.0, "#163f78"], [1.0, "#3987e5"]]

# Nuage de mots : teintes claires du thème, toutes lisibles sur bleu nuit.
WC_CMAP = ListedColormap(["#86b6ef", "#e8b54d", "#9ec5f4", "#f2cc8f", "#54c49c", "#c9d6ee"])
WC_FOND = NUIT_SURFACE

# ---------------------------------------------------------------------------
# Mise en forme Plotly commune
# ---------------------------------------------------------------------------
POLICE_GRAPH = "Inter, system-ui, sans-serif"
GRILLE = "rgba(158,197,244,0.14)"
AXE = "rgba(158,197,244,0.30)"

LAYOUT_COMMUN = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family=POLICE_GRAPH, color="#dbe4f5", size=13),
    title=dict(font=dict(family=POLICE_GRAPH, size=16, color="#f1e9d6"), x=0.0),
    margin=dict(l=10, r=10, t=56, b=10),
    hoverlabel=dict(
        bgcolor="#0d1830",
        bordercolor="rgba(232,181,77,0.45)",
        font=dict(family=POLICE_GRAPH, color=ENCRE_PRIMAIRE, size=13),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def style_fig(fig, titre: str | None = None):
    """Applique le thème « Minuit & Or » à une figure Plotly."""
    fig.update_layout(**LAYOUT_COMMUN)
    if titre:
        fig.update_layout(title_text=titre)
    fig.update_xaxes(gridcolor=GRILLE, linecolor=AXE, zerolinecolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, linecolor=AXE, zerolinecolor=GRILLE)
    return fig


# ---------------------------------------------------------------------------
# CSS — base commune (typographie, verre, boutons, sidebar)
# ---------------------------------------------------------------------------
_CSS_BASE = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #f1e9d6 !important;
    letter-spacing: 0.01em;
}
[data-testid="stHeader"] { background: transparent; }

/* ------- Fond de page : nuit profonde + halos fixes ------- */
.stApp {
    background:
        radial-gradient(55rem 38rem at 12% -8%, rgba(57,135,229,0.13), transparent 60%),
        radial-gradient(48rem 34rem at 90% 108%, rgba(232,181,77,0.10), transparent 62%),
        linear-gradient(165deg, #060b18 0%, #0a1426 45%, #0d1b36 100%);
    background-attachment: fixed;
}

/* ------- Cartes KPI : glassmorphism ------- */
[data-testid="stMetric"] {
    background: linear-gradient(150deg, rgba(255,255,255,0.065), rgba(255,255,255,0.022));
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(232,181,77,0.22) !important;
    border-radius: 18px;
    padding: 1rem 1.15rem;
    box-shadow: 0 10px 30px rgba(3,8,20,0.45);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(232,181,77,0.45) !important;
    box-shadow: 0 16px 42px rgba(3,8,20,0.60), inset 0 0 0 1px rgba(232,181,77,0.18);
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #f6e7c6;
}
[data-testid="stMetricLabel"] { color: #a9b4cc; }

/* ------- Conteneurs de graphiques : verre léger ------- */
[data-testid="stPlotlyChart"], [data-testid="stImage"] {
    background: linear-gradient(160deg, rgba(255,255,255,0.040), rgba(255,255,255,0.015));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 0.65rem;
    box-shadow: 0 8px 26px rgba(3,8,20,0.38);
    transition: border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stPlotlyChart"]:hover, [data-testid="stImage"]:hover {
    border-color: rgba(158,197,244,0.28);
    box-shadow: 0 12px 34px rgba(3,8,20,0.55);
}

/* ------- Tableaux ------- */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(3,8,20,0.35);
}

/* ------- Bandeau d'avertissement (données fictives) ------- */
[data-testid="stAlert"] {
    background: rgba(232,181,77,0.09);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(232,181,77,0.32);
    border-radius: 14px;
}

/* ------- Expanders ------- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
}

/* ------- Boutons ------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(232,181,77,0.35);
    background: linear-gradient(135deg, rgba(232,181,77,0.16), rgba(232,181,77,0.05));
    color: #f2cc8f;
    font-weight: 600;
    transition: all .25s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #e8b54d;
    color: #ffe9bd;
    transform: translateY(-1px);
    box-shadow: 0 0 20px rgba(232,181,77,0.35);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f2cc8f 0%, #e8b54d 45%, #c98500 100%);
    color: #221703;
    border: none;
    font-weight: 700;
}
.stButton > button[kind="primary"]:hover {
    color: #140d02;
    box-shadow: 0 6px 26px rgba(232,181,77,0.55);
}

/* ------- Barre latérale : verre assorti ------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13,22,44,0.94), rgba(8,13,27,0.96));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid rgba(232,181,77,0.15);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f2cc8f !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(232,181,77,0.18); }

/* ------- Séparateurs de sections : filet doré ------- */
[data-testid="stHeaderActionElements"] + hr, h2 + hr { border: none; }

/* ------- Bannière d'en-tête du dashboard ------- */
.dash-hero {
    position: relative;
    text-align: center;
    padding: 1.6rem 1.4rem 1.5rem;
    border-radius: 20px;
    background: linear-gradient(150deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(232,181,77,0.25);
    box-shadow: 0 14px 40px rgba(3,8,20,0.5);
    overflow: hidden;
}
.dash-hero::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #e8b54d, transparent);
}
.dash-hero h1 {
    margin: 0;
    font-size: clamp(1.6rem, 3.4vw, 2.4rem);
    background: linear-gradient(95deg, #f6e3b4 10%, #e8b54d 45%, #9ec5f4 90%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent !important;
}
.dash-hero p { margin: 0.45rem 0 0; color: #a9b4cc; font-size: 0.98rem; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(26px); }
    to   { opacity: 1; transform: none; }
}
"""

# ---------------------------------------------------------------------------
# CSS — page d'accueil immersive (fond animé pur CSS, héros, cartes)
# ---------------------------------------------------------------------------
_CSS_ACCUEIL = """
/* Plein écran : pas de barre latérale sur l'accueil */
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2.5rem;
    max-width: 1100px;
}

/* ------- Fond animé : aurore + blobs lumineux (aucune image) ------- */
.stApp {
    background: linear-gradient(-45deg, #060b18, #0b1630, #15224a, #0d1b36, #060b18);
    background-size: 400% 400%;
    background-attachment: fixed;
    animation: aurora 20s ease infinite;
}
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    z-index: 0;
    pointer-events: none;
}
.stApp::before {
    width: 48vw; height: 48vw;
    left: -12vw; top: -14vh;
    background: radial-gradient(circle, rgba(57,135,229,0.34), transparent 65%);
    animation: blobA 16s ease-in-out infinite alternate;
}
.stApp::after {
    width: 42vw; height: 42vw;
    right: -10vw; bottom: -16vh;
    background: radial-gradient(circle, rgba(232,181,77,0.26), transparent 65%);
    animation: blobB 21s ease-in-out infinite alternate;
}
@keyframes aurora {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes blobA {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(9vw, 7vh) scale(1.18); }
}
@keyframes blobB {
    from { transform: translate(0, 0) scale(1.05); }
    to   { transform: translate(-8vw, -9vh) scale(0.9); }
}

/* ------- Héros ------- */
.hero-wrap { position: relative; z-index: 1; text-align: center; padding: 7vh 0.5rem 0.5rem; }
.hero-eyebrow {
    letter-spacing: 0.45em;
    text-transform: uppercase;
    color: #9ec5f4;
    font-size: 0.82rem;
    font-weight: 600;
    animation: fadeUp 0.9s ease 0.10s both;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 700;
    line-height: 1.12;
    margin: 1.1rem auto 0.9rem;
    max-width: 22ch;
    background: linear-gradient(100deg, #f6e3b4 0%, #e8b54d 28%, #9ec5f4 62%, #f6e3b4 100%);
    background-size: 250% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    animation: fadeUp 1.1s ease 0.25s both, textShine 9s linear 1.4s infinite;
}
@keyframes textShine { to { background-position: 250% center; } }
.hero-sub {
    color: #c3cde3;
    font-size: clamp(1rem, 1.8vw, 1.2rem);
    max-width: 58ch;
    margin: 0 auto;
    line-height: 1.6;
    animation: fadeUp 1s ease 0.45s both;
}
.hero-rule {
    width: 140px; height: 2px;
    margin: 1.6rem auto 1.2rem;
    background: linear-gradient(90deg, transparent, #e8b54d, transparent);
    animation: fadeUp 1s ease 0.55s both;
}
.hero-badge {
    display: inline-block;
    padding: 0.45rem 1.1rem;
    border-radius: 999px;
    background: rgba(232,181,77,0.10);
    border: 1px solid rgba(232,181,77,0.35);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: #f2cc8f;
    font-size: 0.86rem;
    animation: fadeUp 1s ease 0.65s both;
}

/* ------- Cartes de fonctionnalités (verre) ------- */
.hero-grid {
    display: flex;
    gap: 1.1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 2.3rem auto 1.4rem;
    max-width: 980px;
    position: relative;
    z-index: 1;
    animation: fadeUp 1s ease 0.8s both;
}
.hero-card {
    width: 290px;
    text-align: left;
    padding: 1.35rem 1.25rem;
    border-radius: 18px;
    background: linear-gradient(150deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 10px 32px rgba(3,8,20,0.45);
    transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease;
}
.hero-card:hover {
    transform: translateY(-6px);
    border-color: rgba(232,181,77,0.45);
    box-shadow: 0 18px 46px rgba(3,8,20,0.65), 0 0 24px rgba(232,181,77,0.12);
}
.hero-card .ic { font-size: 1.7rem; }
.hero-card h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.02rem;
    color: #f2e7cf !important;
    margin: 0.55rem 0 0.3rem;
}
.hero-card p { color: #a9b4cc; font-size: 0.88rem; line-height: 1.5; margin: 0; }

/* ------- Bouton d'entrée (CTA) ------- */
div[data-testid="stButton"] { position: relative; z-index: 1; }
.stButton > button[kind="primary"] {
    width: 100%;
    padding: 0.9rem 2.4rem;
    font-size: 1.1rem;
    border-radius: 999px;
    letter-spacing: 0.03em;
    animation: fadeUp 1s ease 0.95s both, pulseGlow 3.2s ease 2.2s infinite;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-2px) scale(1.02); }
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 8px 28px rgba(232,181,77,0.30); }
    50%      { box-shadow: 0 10px 46px rgba(232,181,77,0.55); }
}
.hero-foot {
    text-align: center;
    color: #7c8aa5;
    font-size: 0.82rem;
    margin-top: 2.2rem;
    position: relative;
    z-index: 1;
    animation: fadeUp 1s ease 1.1s both;
}
"""


def inject_css(vue: str = "dashboard") -> None:
    """Injecte le CSS du thème. ``vue`` : « dashboard » ou « accueil »."""
    css = _CSS_BASE + (_CSS_ACCUEIL if vue == "accueil" else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
