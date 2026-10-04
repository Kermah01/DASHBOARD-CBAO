"""Dashboard Bancaire — Démo.

Point d'entrée de l'application Streamlit. Les données affichées sont 100 %
fictives : elles sont produites par ``generate_synthetic_data.py`` (graine
fixe) et ne proviennent d'aucun établissement bancaire réel.

L'application s'ouvre sur une page d'accueil immersive (fond animé pur CSS),
puis bascule vers le dashboard via ``st.session_state``.
"""

import os

import pandas as pd
import streamlit as st

from dashboard import render_dashboard
from theme import inject_css

CHEMIN_DONNEES = os.path.join(os.path.dirname(__file__), "data", "donnees_synthetiques.csv")

st.set_page_config(
    page_title="Dashboard Bancaire — Démo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "vue" not in st.session_state:
    st.session_state.vue = "accueil"


def _aller_au_dashboard() -> None:
    st.session_state.vue = "dashboard"


def _retour_accueil() -> None:
    st.session_state.vue = "accueil"


# ---------------------------------------------------------------------------
# Page d'accueil immersive
# ---------------------------------------------------------------------------
def render_accueil() -> None:
    inject_css("accueil")
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow">Expérience client · Réseau d'agences</div>
            <h1 class="hero-title">La satisfaction client, révélée en pleine lumière</h1>
            <p class="hero-sub">
                Plongez dans les retours de la boîte à idées digitale d'un réseau
                d'agences bancaires : indicateurs clés, classements, analyses
                croisées et exploration des suggestions en texte libre.
            </p>
            <div class="hero-rule"></div>
            <div class="hero-badge">⚠️ Démonstration — données 100 % fictives, générées aléatoirement</div>
        </div>
        <div class="hero-grid">
            <div class="hero-card">
                <div class="ic">📊</div>
                <h3>KPI &amp; classements</h3>
                <p>Volumes de questionnaires, notes moyennes d'accueil et de prise
                en charge, palmarès des zones et des agences.</p>
            </div>
            <div class="hero-card">
                <div class="ic">🔭</div>
                <h3>Analyses croisées</h3>
                <p>Répartitions, histogrammes croisés et évolution mensuelle,
                filtrables par année, période et périmètre.</p>
            </div>
            <div class="hero-card">
                <div class="ic">💬</div>
                <h3>Voix du client</h3>
                <p>Nuage de mots, bigrammes et treemap pour faire parler les
                suggestions laissées en texte libre.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    gauche, centre, droite = st.columns([1.2, 1, 1.2])
    with centre:
        st.button(
            "✦ Explorer le dashboard",
            type="primary",
            key="cta_explorer",
            on_click=_aller_au_dashboard,
            use_container_width=True,
        )
    st.markdown(
        '<div class="hero-foot">Streamlit · Plotly · pandas — projet de démonstration</div>',
        unsafe_allow_html=True,
    )


if st.session_state.vue == "accueil":
    render_accueil()
    st.stop()

# ---------------------------------------------------------------------------
# Vue dashboard
# ---------------------------------------------------------------------------
inject_css("dashboard")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données de démonstration…")
def charger_donnees_demo(chemin: str) -> pd.DataFrame:
    return pd.read_csv(chemin)


def lire_fichier_charge(fichier) -> pd.DataFrame:
    if fichier.name.lower().endswith(".csv"):
        return pd.read_csv(fichier)
    return pd.read_excel(fichier, engine="openpyxl")


# ---------------------------------------------------------------------------
# En-tête + disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dash-hero">
        <h1>Dashboard Bancaire — Démo</h1>
        <p>
            Analyse de la satisfaction client d'un réseau d'agences bancaires
            (questionnaires « boîte à idées digitale »)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "**Données 100 % fictives.** Ce tableau de bord est une démonstration : "
    "toutes les données (zones, agences, notes, commentaires) sont générées "
    "aléatoirement par `generate_synthetic_data.py` et ne correspondent à "
    "aucune banque ni à aucun client réel.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Barre latérale : source de données + périmètre d'analyse
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 Dashboard Bancaire")
st.sidebar.caption("Démo — données synthétiques")
st.sidebar.button("← Retour à l'accueil", key="btn_accueil", on_click=_retour_accueil)

st.sidebar.subheader("Source des données")
fichier_charge = st.sidebar.file_uploader(
    "Analyser votre propre fichier (optionnel)",
    type=["csv", "xlsx"],
    help="Le fichier doit contenir les mêmes colonnes que le jeu de démonstration.",
)

if fichier_charge is not None:
    try:
        df = lire_fichier_charge(fichier_charge)
        st.sidebar.success(f"Fichier chargé : {len(df)} lignes")
    except Exception as exc:  # fichier illisible -> on retombe sur la démo
        st.sidebar.error(f"Fichier illisible ({exc}). Données de démo utilisées.")
        df = charger_donnees_demo(CHEMIN_DONNEES)
else:
    df = charger_donnees_demo(CHEMIN_DONNEES)

colonnes_requises = {
    "Horodateur",
    "Zone",
    "Agence",
    "Point de contact",
    "Note de l'accueil",
    "Note de la prise en charge",
    "Suggestions",
}
manquantes = colonnes_requises - set(df.columns)
if manquantes:
    st.error(f"Colonnes manquantes dans le fichier : {', '.join(sorted(manquantes))}")
    st.stop()

# Périmètre : remplace l'ancien système de comptes par zone (démo publique).
st.sidebar.subheader("Périmètre d'analyse")
zones = sorted(df["Zone"].dropna().unique().tolist())
perimetre = st.sidebar.selectbox(
    "Vue",
    ["Direction (toutes les zones)"] + zones,
    help="La vue « Direction » couvre tout le réseau ; une zone restreint "
    "l'analyse aux agences de cette zone.",
)
if perimetre != "Direction (toutes les zones)":
    df = df[df["Zone"] == perimetre]

render_dashboard(df)

st.sidebar.divider()
st.sidebar.caption(
    "Projet de démonstration — Streamlit · Plotly · pandas. "
    "Données fictives générées avec une graine fixe."
)
