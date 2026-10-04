"""Cœur du dashboard : KPI, classements, analyses graphiques et textuelles.

La logique métier reprend celle du tableau de bord d'origine (enquête de
satisfaction d'un réseau d'agences bancaires), modernisée pour les versions
récentes de Streamlit / pandas et alimentée par des données synthétiques.
"""

from __future__ import annotations

import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype
from unidecode import unidecode
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Thème « Minuit & Or » : palette validée (daltonisme + contraste) et mise en
# forme Plotly centralisées dans theme.py.
# ---------------------------------------------------------------------------
from theme import ECHELLE_NOTES, ECHELLE_TREEMAP, PALETTE, WC_CMAP, WC_FOND, style_fig


# ---------------------------------------------------------------------------
# Préparation temporelle du dataframe
# ---------------------------------------------------------------------------
ORDRE_MOIS = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]
DIC_MOIS = {i + 1: m for i, m in enumerate(ORDRE_MOIS)}
ORDRE_JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
DIC_JOURS = {i: j for i, j in enumerate(ORDRE_JOURS)}


def ordre_mois_annee(df: pd.DataFrame) -> list[str]:
    """Liste ordonnée « Mois Année » couvrant toute la période du jeu de données."""
    debut, fin = int(df["Année"].min()), int(df["Année"].max())
    return [f"{mois} {annee}" for annee in range(debut, fin + 1) for mois in ORDRE_MOIS]


@st.cache_data(show_spinner=False)
def transf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Décompose l'horodateur en heure / jour / mois / année (catégories ordonnées)."""
    df = df.copy()
    df["Horodateur"] = pd.to_datetime(df["Horodateur"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["Horodateur"])
    df["Mois"] = df["Horodateur"].dt.month.map(DIC_MOIS)
    df["Jour"] = df["Horodateur"].dt.day_of_week.map(DIC_JOURS)
    df["heure"] = df["Horodateur"].dt.hour
    df["Année"] = df["Horodateur"].dt.year
    df["Mois*"] = pd.Categorical(df["Mois"], categories=ORDRE_MOIS, ordered=True)
    df["Jour*"] = pd.Categorical(df["Jour"], categories=ORDRE_JOURS, ordered=True)
    df["Mois de l'année"] = df["Mois"].astype(str) + " " + df["Année"].astype(str)
    df["Mois de l'année"] = pd.Categorical(
        df["Mois de l'année"], categories=ordre_mois_annee(df), ordered=True
    )
    return df


# ---------------------------------------------------------------------------
# Nettoyage de texte (suggestions / motifs) — analyse lexicale
# ---------------------------------------------------------------------------
STOPWORDS_FR = [
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et",
    "eux", "il", "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "meme", "mes",
    "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "plus", "pour",
    "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi", "ton",
    "tu", "un", "une", "vos", "votre", "vous", "c", "d", "j", "l", "m", "n", "s", "t", "y",
    "est", "sont", "etre", "avoir", "fait", "faire", "nan", "none", "non", "rien", "ras",
]


def sans_stop_words(texte: str, stop_words: list[str]) -> str:
    return " ".join(mot for mot in texte.lower().split(" ") if mot and mot not in stop_words)


@st.cache_data(show_spinner=False)
def nettoyer_texte(serie: pd.Series, stop_words: tuple[str, ...]) -> pd.Series:
    """Minuscule, sans accents, sans caractères spéciaux, sans stop words."""
    serie = serie.fillna("").astype(str).str.lower()
    serie = serie.apply(unidecode)
    serie = serie.apply(lambda x: re.sub(r"[^a-z]+", " ", x))
    return serie.apply(lambda x: sans_stop_words(x, list(stop_words)))


def bigrammes(commentaires: pd.Series) -> Counter:
    mots = " ".join(commentaires).split()
    return Counter(zip(mots, mots[1:]))


# ---------------------------------------------------------------------------
# Classements (palmarès) par zone / agence
# ---------------------------------------------------------------------------
def palmares(df: pd.DataFrame, df_annee: pd.DataFrame, feat: str, debut, fin, critere: str) -> pd.DataFrame:
    periode = df[(df["Mois de l'année"] >= debut) & (df["Mois de l'année"] <= fin)]
    total_periode = periode.groupby(feat, observed=True)["Mois"].count().reset_index(name="Total sur la période")
    total_annee = df_annee.groupby(feat, observed=True)[feat].count().reset_index(name="Total sur l'année")
    moy_acc = periode.groupby(feat, observed=True)["Note de l'accueil"].mean().round(2).reset_index(name="Moy. de l'accueil")
    moy_pec = periode.groupby(feat, observed=True)["Note de la prise en charge"].mean().round(2).reset_index(name="Moy. de la prise en charge")

    if critere == "Total des questionnaires sur l'année":
        d = pd.merge(total_periode, total_annee, on=feat, how="right")
        d.sort_values(by="Total sur l'année", inplace=True, ascending=False)
    elif critere == "Total des questionnaires sur la période":
        d = pd.merge(total_periode, total_annee, on=feat, how="right")
        d.sort_values(by="Total sur la période", inplace=True, ascending=False)
    elif critere == "Moy. de l'accueil":
        d = pd.merge(total_periode, moy_acc, on=feat, how="right")
        d.sort_values(by=critere, inplace=True, ascending=False)
    else:
        d = pd.merge(total_periode, moy_pec, on=feat, how="right")
        d.sort_values(by=critere, inplace=True, ascending=False)
    return d


def format_rang(df: pd.DataFrame, col: str = "Position") -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df[col] = [f"{i + 1}er" if i == 0 else f"{i + 1}ème" for i in range(len(df))]
    return df.set_index(col)


# ---------------------------------------------------------------------------
# Filtre interactif de dataframe
# ---------------------------------------------------------------------------
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une interface de filtrage colonne par colonne au-dessus d'un dataframe."""
    modify = st.checkbox("Ajouter un filtre")
    if not modify:
        return df

    df = df.copy()
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    with st.container():
        colonnes = st.multiselect("Variables à utiliser comme filtre", df.columns)
        for column in colonnes:
            left, right = st.columns((1, 20))
            left.write("↳")
            if isinstance(df[column].dtype, pd.CategoricalDtype) or (
                not is_numeric_dtype(df[column])
                and not is_datetime64_any_dtype(df[column])
                and df[column].nunique() < 100
            ):
                valeurs = df[column].dropna().unique()
                choix = right.multiselect(f"Valeurs de {column}", valeurs, default=list(valeurs))
                df = df[df[column].isin(choix)]
            elif is_numeric_dtype(df[column]):
                _min, _max = int(df[column].min()), int(df[column].max())
                bornes = right.slider(f"Valeurs de {column}", _min, _max, (_min, _max))
                df = df[df[column].between(*bornes)]
            elif is_datetime64_any_dtype(df[column]):
                dates = right.date_input(f"Période pour {column}", value=(df[column].min(), df[column].max()))
                if len(dates) == 2:
                    debut, fin = map(pd.to_datetime, dates)
                    df = df.loc[df[column].between(debut, fin)]
            else:
                texte = right.text_input(f"Texte ou regex dans {column}")
                if texte:
                    df = df[df[column].astype(str).str.contains(texte, na=False)]
    return df


# ---------------------------------------------------------------------------
# Nuage de mots
# ---------------------------------------------------------------------------
def nuage_de_mots(commentaires: pd.Series):
    texte = " ".join(commentaires.values)
    if not texte.strip():
        st.info("Pas assez de texte pour générer un nuage de mots.")
        return
    fig_wc, ax = plt.subplots(figsize=(12, 8))
    fig_wc.patch.set_alpha(0)  # fond transparent : le verre du thème transparaît
    wc = WordCloud(
        background_color=WC_FOND,
        colormap=WC_CMAP,
        collocations=True,
        width=1200,
        height=750,
        stopwords=set(STOPWORDS_FR),
    ).generate(texte)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nuage de mots des suggestions", fontsize=22, color="#f1e9d6", pad=14)
    st.pyplot(fig_wc)
    plt.close(fig_wc)


# ===========================================================================
# PAGE PRINCIPALE
# ===========================================================================
def render_dashboard(df_brut: pd.DataFrame) -> None:
    df = transf_df(df_brut)
    if df.empty:
        st.error("Aucune ligne exploitable (vérifiez le format de la colonne Horodateur : jj/mm/aaaa hh:mm).")
        return

    df_complet = df.copy()

    # ------------------------------------------------------------------ KPI annuels
    st.header("KPI annuels", divider="orange")
    annee = st.selectbox(
        "Année d'analyse",
        np.sort(df["Année"].unique())[::-1],
        index=0,
    )
    df_annee = df[df["Année"] == annee]

    mois_disponibles = df_annee["Mois*"].unique().sort_values(ascending=True)
    mois_reference = mois_disponibles[-2] if len(mois_disponibles) > 1 else mois_disponibles[-1]
    dec_temp = dec_temp1 = dec_temp2 = mois_reference
    acc_selected = pec_selected = ""
    new_norm = new_norm2 = 4

    # Personnalisation des deltas des KPI (barre latérale)
    st.sidebar.subheader("Variations des KPI")
    if st.sidebar.checkbox("Nombre total de questionnaires"):
        dec_temp = st.sidebar.select_slider(
            "Décalage temporel (en mois)", options=mois_disponibles, value=mois_reference
        )
    if st.sidebar.checkbox("Note moyenne de l'accueil"):
        acc_selected = st.sidebar.radio(
            "Accueil : personnaliser", ["Modifier le décalage temporel", "Modifier la norme"]
        )
        if acc_selected == "Modifier le décalage temporel":
            dec_temp1 = st.sidebar.select_slider(
                "Décalage temporel (accueil)", options=mois_disponibles, value=mois_reference
            )
        else:
            new_norm = st.sidebar.number_input("Norme (accueil)", min_value=1, max_value=5, value=4)
    if st.sidebar.checkbox("Note moyenne de la prise en charge"):
        pec_selected = st.sidebar.radio(
            "Prise en charge : personnaliser", ["Modifier le décalage temporel ", "Modifier la norme "]
        )
        if pec_selected == "Modifier le décalage temporel ":
            dec_temp2 = st.sidebar.select_slider(
                "Décalage temporel (prise en charge)", options=mois_disponibles, value=mois_reference
            )
        else:
            new_norm2 = st.sidebar.number_input("Norme (prise en charge)", min_value=1, max_value=5, value=4)

    col1, col2, col3 = st.columns(3)
    delta_quest = df_annee.shape[0] - df_annee[df_annee["Mois*"] <= dec_temp].shape[0]
    col1.metric(
        f"Questionnaires soumis en {annee}",
        f"{df_annee.shape[0]:,}".replace(",", " "),
        f"{delta_quest} depuis fin {dec_temp}",
        help="Cochez « Nombre total de questionnaires » dans la barre latérale pour personnaliser le delta.",
        border=True,
    )

    moy_accueil = np.round(df_annee["Note de l'accueil"].mean(), 2)
    if acc_selected == "Modifier le décalage temporel":
        delta_acc = np.round(moy_accueil - df_annee[df_annee["Mois*"] <= dec_temp1]["Note de l'accueil"].mean(), 2)
        col2.metric(f"Note moyenne de l'accueil en {annee}", f"{moy_accueil} / 5",
                    f"{delta_acc} vs cumul à fin {dec_temp1}", border=True)
    elif acc_selected == "Modifier la norme":
        col2.metric(f"Note moyenne de l'accueil en {annee}", f"{moy_accueil} / 5",
                    f"{np.round(moy_accueil - new_norm, 2)} (norme de {new_norm})", border=True)
    else:
        col2.metric(f"Note moyenne de l'accueil en {annee}", f"{moy_accueil} / 5",
                    f"{np.round(moy_accueil - 4, 2)} (norme de 4)", border=True)

    moy_pec = np.round(df_annee["Note de la prise en charge"].mean(), 2)
    if pec_selected == "Modifier le décalage temporel ":
        delta_pec = np.round(moy_pec - df_annee[df_annee["Mois*"] <= dec_temp2]["Note de la prise en charge"].mean(), 2)
        col3.metric(f"Note moyenne de la prise en charge en {annee}", f"{moy_pec} / 5",
                    f"{delta_pec} vs cumul à fin {dec_temp2}", border=True)
    elif pec_selected == "Modifier la norme ":
        col3.metric(f"Note moyenne de la prise en charge en {annee}", f"{moy_pec} / 5",
                    f"{np.round(moy_pec - new_norm2, 2)} (norme de {new_norm2})", border=True)
    else:
        col3.metric(f"Note moyenne de la prise en charge en {annee}", f"{moy_pec} / 5",
                    f"{np.round(moy_pec - 4, 2)} (norme de 4)", border=True)

    # --------------------------------------------------------------- KPI mensuels
    st.header("KPI mensuels et classements", divider="orange")
    df = df_complet
    mois_annee = df["Mois de l'année"].unique().sort_values(ascending=True)
    defaut = mois_annee[-1]
    start_val, last_val = st.select_slider(
        "Intervalle d'analyse", options=mois_annee, value=[defaut, defaut]
    )
    titre_periode = (
        f"Analyse des performances en {start_val}"
        if start_val == last_val
        else f"Analyse des performances entre {start_val} et {last_val}"
    )
    st.subheader(titre_periode)

    masque_periode = (df["Mois de l'année"] >= start_val) & (df["Mois de l'année"] <= last_val)

    actu, palm, top = st.columns(3)
    with actu:
        st.markdown("**KPI de la période**")
        nb_periode = int(masque_periode.sum())
        delta_mois = int(
            df[df["Mois de l'année"] == last_val].shape[0] - df[df["Mois de l'année"] == start_val].shape[0]
        )
        st.metric("Questionnaires sur la période", nb_periode,
                  f"{delta_mois} entre {start_val} et {last_val}", border=True)

        def nb_avis(masque) -> int:
            return int(
                (df.loc[masque, "Motifs de la note de l'accueil"].count()
                 + df.loc[masque, "Motifs de la note de la prise en charge"].count()) / 2
            )

        delta_avis = nb_avis(df["Mois de l'année"] == last_val) - nb_avis(df["Mois de l'année"] == start_val)
        st.metric("Avis (motifs) sur la période", nb_avis(masque_periode),
                  f"{delta_avis} entre {start_val} et {last_val}", border=True)

        nb_sugg = int(df.loc[masque_periode, "Suggestions"].count())
        delta_sugg = int(
            df.loc[df["Mois de l'année"] == last_val, "Suggestions"].count()
            - df.loc[df["Mois de l'année"] == start_val, "Suggestions"].count()
        )
        st.metric("Suggestions sur la période", nb_sugg,
                  f"{delta_sugg} entre {start_val} et {last_val}", border=True)

    criteres = [
        "Total des questionnaires sur l'année",
        "Total des questionnaires sur la période",
        "Moy. de l'accueil",
        "Moy. de la prise en charge",
    ]
    with palm:
        st.markdown("**Classement par zone**")
        critere = st.radio("Critère de classement (zones)", criteres, key="critere_zone")
        st.dataframe(format_rang(palmares(df, df_annee, "Zone", start_val, last_val, critere)))
    with top:
        st.markdown("**Classement par agence**")
        critere1 = st.radio("Critère de classement (agences)", criteres, key="critere_agence")
        st.dataframe(format_rang(palmares(df, df_annee, "Agence", start_val, last_val, critere1)))

    # ------------------------------------------------- Base de données personnalisée
    st.header("Base de données personnalisée", divider="orange")
    df_perso = filter_dataframe(df)
    st.dataframe(df_perso, height=320)

    st.sidebar.subheader("Base de données des graphiques")
    df_selected = st.sidebar.radio(
        "Base utilisée pour les graphiques",
        ["Année sélectionnée", "Base personnalisée", "Base complète"],
    )
    if df_selected == "Base personnalisée":
        df = df_perso
    elif df_selected == "Année sélectionnée":
        df = df_annee
    # sinon : base complète

    if df.empty:
        st.warning("La base sélectionnée est vide : ajustez vos filtres.")
        return

    # ------------------------------------------------------------ Analyses graphiques
    st.header("Analyses graphiques", divider="orange")
    variables_cat = ["Agence", "Point de contact", "Note de l'accueil",
                     "Note de la prise en charge", "Jour", "Mois", "Zone"]

    st.subheader("Analyse univariée")
    cam, hist = st.columns(2, gap="medium")
    with cam:
        var_pie = st.selectbox("Variable du diagramme circulaire", variables_cat, index=1)
        counts = df[var_pie].value_counts()
        fig_pie = px.pie(
            names=counts.index.astype(str),
            values=counts.values,
            color_discrete_sequence=PALETTE,
            hole=0.45,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(style_fig(fig_pie, f"Répartition — {var_pie}"), width="stretch")
    with hist:
        var_hist = st.selectbox("Variable de l'histogramme", variables_cat, index=6)
        fig_hist = px.histogram(df, x=var_hist, color=var_hist, color_discrete_sequence=PALETTE)
        if var_hist == "Mois":
            fig_hist.update_xaxes(categoryorder="array", categoryarray=ORDRE_MOIS)
        elif var_hist == "Jour":
            fig_hist.update_xaxes(categoryorder="array", categoryarray=ORDRE_JOURS)
        fig_hist.update_layout(showlegend=False, yaxis_title="Nombre de questionnaires")
        st.plotly_chart(style_fig(fig_hist, f"Histogramme — {var_hist}"), width="stretch")

    st.subheader("Analyses croisées")
    quant, qual = st.columns(2, gap="medium")
    with quant:
        num_cols = df.select_dtypes(include=["int", "float"]).columns
        num_cols = [c for c in num_cols if c not in ("Année",)]
        var3 = st.selectbox("Variable numérique 1", num_cols, index=0)
        var4 = st.selectbox("Variable numérique 2", num_cols, index=min(1, len(num_cols) - 1))
        occ = df.groupby([var3, var4], observed=True).size().reset_index(name="Effectif")
        occ["Moyenne des deux variables"] = (occ[var3] + occ[var4]) / 2
        fig_scatter = px.scatter(
            occ, x=var3, y=var4, size="Effectif", size_max=60,
            color="Moyenne des deux variables", color_continuous_scale=ECHELLE_NOTES,
        )
        st.plotly_chart(style_fig(fig_scatter, f"Nuage de points — {var3} vs {var4}"), width="stretch")
    with qual:
        var1 = st.selectbox("Variable catégorielle 1", ["Agence", "Point de contact", "Jour", "Mois", "Zone"], index=4)
        var2 = st.selectbox(
            "Variable catégorielle 2",
            ["Agence", "Point de contact", "Note de la prise en charge", "Note de l'accueil", "Jour", "Mois", "Zone"],
            index=1,
        )
        type_graph = st.sidebar.radio("Type d'histogramme croisé", ["Empilé", "Groupé"])
        barmode = "relative" if type_graph == "Empilé" else "group"
        if var2 in ("Note de l'accueil", "Note de la prise en charge"):
            moyennes = df.groupby(var1, observed=True)[var2].mean().reset_index()
            fig_croise = px.bar(
                moyennes, x=var1, y=var2, color=var2,
                color_continuous_scale=ECHELLE_NOTES, range_color=[1, 5],
            )
            fig_croise.update_layout(yaxis_title=f"{var2} (moyenne / 5)")
        else:
            fig_croise = px.bar(df, x=var1, color=var2, barmode=barmode, color_discrete_sequence=PALETTE)
        if var1 == "Mois":
            fig_croise.update_xaxes(categoryorder="array", categoryarray=ORDRE_MOIS)
        elif var1 == "Jour":
            fig_croise.update_xaxes(categoryorder="array", categoryarray=ORDRE_JOURS)
        st.plotly_chart(style_fig(fig_croise, f"{var1} vs {var2}"), width="stretch")

    # Évolution mensuelle (volume de questionnaires par mois et jour de semaine)
    evolution = (
        df.groupby(["Mois de l'année", "Jour*"], observed=True)
        .size()
        .reset_index(name="Questionnaires")
    )
    fig_evo = px.area(
        evolution, x="Mois de l'année", y="Questionnaires", color="Jour*",
        color_discrete_sequence=PALETTE,
        category_orders={"Jour*": ORDRE_JOURS},
    )
    fig_evo.update_xaxes(categoryorder="array", categoryarray=ordre_mois_annee(df_complet))
    fig_evo.update_layout(height=450, legend_title_text="Jour de la semaine")
    st.plotly_chart(
        style_fig(fig_evo, "Évolution mensuelle des questionnaires (par jour de la semaine)"),
        width="stretch",
    )

    # ------------------------------------------------------------ Analyse textuelle
    st.header("Analyse des suggestions (texte libre)", divider="orange")
    commentaires = nettoyer_texte(df["Suggestions"], tuple(STOPWORDS_FR))
    compteur = bigrammes(commentaires)
    top_bigrams = dict(sorted(compteur.items(), key=lambda kv: kv[1], reverse=True)[:20])

    wc_col, big_col = st.columns(2, gap="medium")
    with wc_col:
        nuage_de_mots(commentaires)
    with big_col:
        if top_bigrams:
            df_big = pd.DataFrame(
                {
                    "Bigramme": [" ".join(b) for b in top_bigrams],
                    "Fréquence": list(top_bigrams.values()),
                }
            ).sort_values("Fréquence")
            fig_big = px.bar(
                df_big, x="Fréquence", y="Bigramme", orientation="h",
                color_discrete_sequence=[PALETTE[3]],
            )
            fig_big.update_layout(height=520)
            st.plotly_chart(style_fig(fig_big, "Bigrammes les plus fréquents"), width="stretch")

    if top_bigrams:
        labels = [" ".join(b) for b in top_bigrams]
        valeurs = list(top_bigrams.values())
        fig_tree = go.Figure(
            go.Treemap(
                labels=labels,
                parents=[""] * len(labels),
                values=valeurs,
                text=[f"Fréquence : {v}" for v in valeurs],
                hoverinfo="label+text",
                marker=dict(colors=valeurs, colorscale=ECHELLE_TREEMAP),
                textfont=dict(color="#ffffff"),
            )
        )
        fig_tree.update_layout(height=450)
        st.plotly_chart(
            style_fig(fig_tree, "Treemap — bigrammes les plus fréquents dans les suggestions"),
            width="stretch",
        )
