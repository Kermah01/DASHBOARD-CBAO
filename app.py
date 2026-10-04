"""Dashboard Bancaire — Démo.

Point d'entrée de l'application Streamlit. Les données affichées sont 100 %
fictives : elles sont produites par ``generate_synthetic_data.py`` (graine
fixe) et ne proviennent d'aucun établissement bancaire réel.
"""

import os

import pandas as pd
import streamlit as st

from dashboard import render_dashboard

CHEMIN_DONNEES = os.path.join(os.path.dirname(__file__), "data", "donnees_synthetiques.csv")

st.set_page_config(
    page_title="Dashboard Bancaire — Démo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    <div style="text-align:center;padding:1rem 1.25rem;border-radius:10px;
                background:#fcfcfb;border:1px solid rgba(11,11,11,0.10);">
        <h1 style="margin:0;color:#0b0b0b;">Dashboard Bancaire — Démo</h1>
        <p style="margin:0.35rem 0 0;color:#52514e;">
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
st.sidebar.title("📊 Dashboard Bancaire")
st.sidebar.caption("Démo — données synthétiques")

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
