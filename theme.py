"""Thème « Minuit & Or » — identité visuelle du dashboard.

Centralise tout le design : palette validée (daltonisme + contraste, via le
validateur dataviz, surface #101a30), mise en forme Plotly, nuage de mots et
CSS global (image héros dorée en fond, glassmorphism, typographie).

Ambiance : élégance bancaire premium — vagues de lumière dorées sur bleu nuit.
L'image ``assets/hero_bg.webp`` est encodée en base64 et sert de fond aux deux
vues : quasi pure sur l'accueil (overlay léger), affleurante sur le dashboard
(overlay très opaque pour préserver la lisibilité des graphiques).
"""

from __future__ import annotations

import base64
from pathlib import Path

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
# Image héros (vagues dorées sur bleu nuit) encodée une seule fois par session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _hero_b64() -> str:
    """Encode ``assets/hero_bg.webp`` en base64 (mise en cache ressource)."""
    chemin = Path(__file__).resolve().parent / "assets" / "hero_bg.webp"
    return base64.b64encode(chemin.read_bytes()).decode("ascii")


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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
/* Streamlit enveloppe le texte des titres dans un span : cibler aussi les
   descendants, sinon la règle universelle Inter ci-dessus l'emporte. */
h1, h2, h3, h1 span, h2 span, h3 span {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #f1e9d6 !important;
    letter-spacing: 0.01em;
}
[data-testid="stHeader"] { background: transparent; }

/* Contenu au-dessus des couches de fond fixes */
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; }

/* ------- Cartes KPI : verre bordé d'or ------- */
[data-testid="stMetric"] {
    background: linear-gradient(150deg, rgba(20,30,54,0.72), rgba(10,16,32,0.60));
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(232,181,77,0.26) !important;
    border-radius: 18px;
    padding: 1.05rem 1.2rem;
    box-shadow: 0 10px 30px rgba(3,8,20,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 4%, rgba(232,181,77,0.75), transparent 96%);
    opacity: 0.8;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(232,181,77,0.55) !important;
    box-shadow: 0 18px 46px rgba(3,8,20,0.62), 0 0 26px rgba(232,181,77,0.14),
                inset 0 0 0 1px rgba(232,181,77,0.16);
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #f6e7c6;
    font-size: 2.1rem;
}
[data-testid="stMetricLabel"] { color: #c0cadf; font-weight: 600; letter-spacing: 0.015em; }

/* ------- Conteneurs de graphiques : verre léger ------- */
[data-testid="stPlotlyChart"], [data-testid="stImage"] {
    background: linear-gradient(160deg, rgba(18,28,52,0.62), rgba(9,15,30,0.52));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 0.65rem;
    box-shadow: 0 8px 26px rgba(3,8,20,0.38);
    transition: border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stPlotlyChart"]:hover, [data-testid="stImage"]:hover {
    border-color: rgba(232,181,77,0.30);
    box-shadow: 0 12px 34px rgba(3,8,20,0.55), 0 0 22px rgba(232,181,77,0.08);
}

/* ------- Tableaux ------- */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(232,181,77,0.16);
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

/* ------- Barre latérale : verre assorti, filet doré ------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13,22,44,0.94), rgba(8,13,27,0.97));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid rgba(232,181,77,0.16);
    box-shadow: 8px 0 28px rgba(3,8,20,0.35);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f2cc8f !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(232,181,77,0.18); }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(26px); }
    to   { opacity: 1; transform: none; }
}
"""

# ---------------------------------------------------------------------------
# CSS — vue dashboard : l'image dorée affleure sous un voile très opaque
# ---------------------------------------------------------------------------
_CSS_DASHBOARD = """
/* ------- Fond : image héros sous un voile nuit quasi opaque ------- */
.stApp {
    background:
        linear-gradient(165deg, rgba(6,10,22,0.92) 0%, rgba(6,10,22,0.95) 55%, rgba(6,10,22,0.97) 100%),
        url("__HERO__") center / cover no-repeat fixed #060b18;
}

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* ------- Fil d'ariane / retour accueil ------- */
.st-key-btn_retour_haut button {
    border-radius: 999px;
    padding: 0.3rem 1.05rem;
    font-size: 0.85rem;
    background: rgba(16,26,48,0.55);
    border: 1px solid rgba(232,181,77,0.30);
    color: #e9d9ae;
}
.dash-crumb {
    letter-spacing: 0.4em;
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 600;
    color: #9ec5f4;
    margin-bottom: 0.55rem;
}
.dash-crumb .sep { color: rgba(232,181,77,0.8); margin: 0 0.5rem; }

/* ------- Bannière d'en-tête du dashboard ------- */
.dash-hero {
    position: relative;
    text-align: center;
    padding: 1.8rem 1.4rem 1.7rem;
    border-radius: 22px;
    background:
        radial-gradient(60rem 16rem at 50% 120%, rgba(232,181,77,0.10), transparent 70%),
        linear-gradient(150deg, rgba(22,33,60,0.70), rgba(10,16,32,0.55));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(232,181,77,0.28);
    box-shadow: 0 14px 40px rgba(3,8,20,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    overflow: hidden;
    animation: fadeUp 0.7s ease both;
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
    font-size: clamp(1.7rem, 3.6vw, 2.5rem);
    background: linear-gradient(95deg, #f6e3b4 10%, #e8b54d 45%, #9ec5f4 90%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent !important;
    filter: drop-shadow(0 2px 14px rgba(4,7,16,0.55));
}
.dash-hero p { margin: 0.5rem 0 0; color: #b7c2d9; font-size: 0.98rem; }
.dash-hero .filet {
    width: 120px; height: 2px;
    margin: 0.9rem auto 0;
    background: linear-gradient(90deg, transparent, rgba(232,181,77,0.9), transparent);
}

/* ------- Titres de sections : filets dorés ------- */
[data-testid="stHeading"] h2 {
    font-size: 1.45rem;
    margin-top: 0.4rem;
}
[data-testid="stHeading"] hr,
[data-testid="stHeadingDivider"] {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, rgba(232,181,77,0.85), rgba(232,181,77,0.25) 45%, transparent 85%) !important;
}
[data-testid="stHeading"] h3 { font-size: 1.12rem; color: #e9dec0 !important; }

/* ------- Pied de page ------- */
.dash-foot {
    text-align: center;
    color: #7c8aa5;
    font-size: 0.82rem;
    margin-top: 2.6rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(232,181,77,0.14);
}
"""

# ---------------------------------------------------------------------------
# CSS — page d'accueil cinématographique (image héros plein écran)
# ---------------------------------------------------------------------------
_CSS_ACCUEIL = """
/* Plein écran : pas de barre latérale sur l'accueil */
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2.2rem;
    max-width: 1100px;
}

/* ------- Fond : image héros plein écran + lente respiration (Ken Burns) ------- */
.stApp { background: #04070f; }
.stApp::before {
    content: "";
    position: fixed;
    inset: -5vmax;
    z-index: 0;
    background: url("__HERO__") center / cover no-repeat;
    animation: kenburns 42s ease-in-out infinite alternate;
    pointer-events: none;
}
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(130% 90% at 50% 8%, transparent 42%, rgba(4,7,16,0.42) 100%),
        linear-gradient(180deg, rgba(4,7,16,0.35) 0%, rgba(4,7,16,0.40) 55%, rgba(4,7,16,0.55) 100%);
    pointer-events: none;
}
@keyframes kenburns {
    from { transform: scale(1) translate(0, 0); }
    to   { transform: scale(1.09) translate(1.2vw, -1vh); }
}

/* Rideau d'ouverture : fondu depuis le noir (signature cinéma) */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 50;
    background: #04070f;
    animation: curtain 1.5s ease 0.05s both;
    pointer-events: none;
}
@keyframes curtain { from { opacity: 1; } to { opacity: 0; } }

/* ------- Héros ------- */
.hero-wrap { position: relative; z-index: 1; text-align: center; padding: 6.5vh 0.5rem 0.5rem; }
.hero-eyebrow {
    letter-spacing: 0.45em;
    text-transform: uppercase;
    color: #b9d4f7;
    font-size: 0.82rem;
    font-weight: 600;
    text-shadow: 0 2px 16px rgba(4,7,16,0.9);
    animation: fadeUp 0.9s ease 0.35s both;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2.7rem, 6.4vw, 4.9rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 1.1rem auto 0.9rem !important;
    max-width: 21ch;
    background: linear-gradient(100deg, #fdf3dc 0%, #f2cc8f 30%, #e8b54d 52%, #fdf3dc 76%, #e8b54d 100%);
    background-size: 220% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    filter: drop-shadow(0 3px 22px rgba(4,7,16,0.75));
    animation: fadeUp 1.1s ease 0.55s both, textShine 8s linear 2s infinite;
}
@keyframes textShine { to { background-position: 220% center; } }
.hero-sub {
    color: #dde5f3;
    font-size: clamp(1rem, 1.8vw, 1.22rem);
    max-width: 58ch;
    margin: 0 auto !important;
    line-height: 1.65;
    text-shadow: 0 2px 18px rgba(4,7,16,0.95), 0 0 42px rgba(4,7,16,0.8);
    animation: fadeUp 1s ease 0.8s both;
}
.hero-rule {
    width: 150px; height: 2px;
    margin: 1.7rem auto 1.25rem;
    background: linear-gradient(90deg, transparent, #e8b54d, transparent);
    box-shadow: 0 0 14px rgba(232,181,77,0.55);
    animation: fadeUp 1s ease 0.95s both;
}
.hero-badge {
    display: inline-block;
    padding: 0.45rem 1.15rem;
    border-radius: 999px;
    background: rgba(10,16,32,0.55);
    border: 1px solid rgba(232,181,77,0.40);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: #f2cc8f;
    font-size: 0.86rem;
    text-shadow: 0 1px 10px rgba(4,7,16,0.8);
    animation: fadeUp 1s ease 1.1s both;
}

/* ------- Cartes de fonctionnalités (verre sur image) ------- */
.hero-grid {
    display: flex;
    gap: 1.1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 2.4rem auto 1.5rem;
    max-width: 980px;
    position: relative;
    z-index: 1;
}
.hero-card {
    width: 290px;
    text-align: left;
    padding: 1.4rem 1.3rem;
    border-radius: 18px;
    background: linear-gradient(155deg, rgba(16,26,48,0.58), rgba(7,12,24,0.46));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(232,181,77,0.20);
    box-shadow: 0 12px 36px rgba(3,8,20,0.55), inset 0 1px 0 rgba(255,255,255,0.07);
    transition: transform .3s ease, border-color .3s ease, box-shadow .3s ease;
    animation: fadeUp 0.95s ease both;
}
.hero-card:nth-child(1) { animation-delay: 1.25s; }
.hero-card:nth-child(2) { animation-delay: 1.4s; }
.hero-card:nth-child(3) { animation-delay: 1.55s; }
.hero-card:hover {
    transform: translateY(-7px);
    border-color: rgba(232,181,77,0.55);
    box-shadow: 0 20px 52px rgba(3,8,20,0.7), 0 0 30px rgba(232,181,77,0.16);
}
.hero-card .ic { font-size: 1.7rem; filter: drop-shadow(0 2px 8px rgba(4,7,16,0.6)); }
.hero-card h3, .hero-card h3 span {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.02rem;
    color: #f6e9cd !important;
    margin: 0.55rem 0 0.3rem;
}
.hero-card p { color: #c3cde0; font-size: 0.88rem; line-height: 1.5; margin: 0; }

/* ------- Bouton d'entrée (CTA) : or au halo pulsant ------- */
div[data-testid="stButton"] { position: relative; z-index: 1; }
.stButton > button[kind="primary"] {
    width: 100%;
    padding: 0.95rem 2.4rem;
    font-size: 1.12rem;
    border-radius: 999px;
    letter-spacing: 0.03em;
    animation: fadeUp 1s ease 1.75s both, pulseGlow 3.2s ease 3s infinite;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-2px) scale(1.02); }
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 8px 30px rgba(232,181,77,0.35); }
    50%      { box-shadow: 0 10px 52px rgba(232,181,77,0.62), 0 0 70px rgba(232,181,77,0.25); }
}
.hero-foot {
    text-align: center;
    color: #93a1bb;
    font-size: 0.82rem;
    margin-top: 2.3rem;
    position: relative;
    z-index: 1;
    text-shadow: 0 1px 10px rgba(4,7,16,0.9);
    animation: fadeUp 1s ease 1.95s both;
}
"""


def inject_css(vue: str = "dashboard") -> None:
    """Injecte le CSS du thème. ``vue`` : « dashboard » ou « accueil »."""
    css = _CSS_BASE + (_CSS_ACCUEIL if vue == "accueil" else _CSS_DASHBOARD)
    css = css.replace("__HERO__", f"data:image/webp;base64,{_hero_b64()}")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
