"""Générateur de données bancaires synthétiques.

Produit un jeu de données 100 % fictif reproduisant la structure attendue par
le dashboard (enquête de satisfaction type « boîte à idées digitale ») :

    Horodateur | Zone | Agence | Point de contact | Note de l'accueil |
    Note de la prise en charge | Motifs de la note de l'accueil |
    Motifs de la note de la prise en charge | Suggestions

Usage :
    python generate_synthetic_data.py

Le fichier est écrit dans ``data/donnees_synthetiques.csv``.
La graine aléatoire est fixe : la génération est entièrement reproductible.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 42
N_LIGNES = 6000
SORTIE = os.path.join("data", "donnees_synthetiques.csv")

# ---------------------------------------------------------------------------
# Référentiel fictif : zones et agences (aucun lien avec une banque réelle)
# ---------------------------------------------------------------------------
ZONES_AGENCES = {
    "Zone Centre": ["Agence du Parc", "Agence de la Gare", "Agence du Marché"],
    "Zone Nord": ["Agence des Palmiers", "Agence du Littoral", "Agence des Dunes"],
    "Zone Sud": ["Agence de la Rivière", "Agence des Collines"],
    "Zone Est": ["Agence de l'Horloge", "Agence des Acacias", "Agence du Stade"],
    "Zone Ouest": ["Agence de la Corniche", "Agence du Phare"],
}

POINTS_DE_CONTACT = [
    "Guichet",
    "Conseiller clientèle",
    "GAB",
    "Accueil",
    "Application mobile",
    "Centre d'appels",
]
P_POINTS = [0.28, 0.22, 0.16, 0.14, 0.12, 0.08]

MOTIFS_POSITIFS = [
    "Personnel accueillant et souriant",
    "Service rapide et efficace",
    "Conseiller à l'écoute du client",
    "Agence propre et bien organisée",
    "Prise en charge rapide au guichet",
    "Très bonne orientation dès l'entrée",
]

MOTIFS_NEGATIFS = [
    "Temps d'attente trop long au guichet",
    "File d'attente mal organisée",
    "Manque de personnel aux heures de pointe",
    "Guichet automatique souvent en panne",
    "Peu d'écoute face à ma réclamation",
    "Horaires d'ouverture peu pratiques",
]

SUGGESTIONS = [
    "Réduire le temps d'attente au guichet",
    "Améliorer l'application mobile de la banque",
    "Ajouter un guichet automatique dans le quartier",
    "Augmenter le personnel aux heures de pointe",
    "Améliorer le temps d'attente en agence",
    "Installer plus de sièges dans la salle d'attente",
    "Former le personnel à l'accueil des clients",
    "Mettre en place un système de tickets plus clair",
    "Proposer plus de services sur l'application mobile",
    "Climatiser la salle d'attente de l'agence",
    "Ouvrir l'agence le samedi matin",
    "Réduire les frais de tenue de compte",
]

JOURS_OUVRES = [0, 1, 2, 3, 4, 5]  # lundi -> samedi
P_JOURS = [0.19, 0.18, 0.18, 0.17, 0.18, 0.10]


def generer(n: int = N_LIGNES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Zones / agences, avec un « niveau de qualité » propre à chaque agence
    zones = list(ZONES_AGENCES)
    agences = [(z, a) for z, liste in ZONES_AGENCES.items() for a in liste]
    poids_agences = rng.dirichlet(np.ones(len(agences)) * 8)
    biais_agences = {a: rng.normal(0.0, 0.45) for _, a in agences}

    idx = rng.choice(len(agences), size=n, p=poids_agences)
    zone_col = np.array([agences[i][0] for i in idx])
    agence_col = np.array([agences[i][1] for i in idx])

    # --- Horodateur : du 01/01/2023 au 30/09/2025, jours ouvrés, 8 h - 17 h,
    #     avec une légère croissance du volume au fil du temps.
    jours = pd.date_range("2023-01-01", "2025-09-30", freq="D")
    jours = jours[jours.dayofweek.isin(JOURS_OUVRES)]
    tendance = np.linspace(1.0, 1.9, len(jours))
    poids_jour_semaine = np.array([P_JOURS[d] for d in jours.dayofweek])
    p_jours = tendance * poids_jour_semaine
    p_jours = p_jours / p_jours.sum()
    dates = rng.choice(jours, size=n, p=p_jours)
    heures = rng.choice(range(8, 18), size=n, p=[0.10, 0.14, 0.15, 0.13, 0.08, 0.06, 0.09, 0.11, 0.09, 0.05])
    minutes = rng.integers(0, 60, size=n)
    horodateur = pd.to_datetime(dates) + pd.to_timedelta(heures, unit="h") + pd.to_timedelta(minutes, unit="m")

    # --- Notes (1 à 5), corrélées entre elles et au biais de l'agence
    base = np.array([biais_agences[a] for a in agence_col])

    def note(biais_local: np.ndarray) -> np.ndarray:
        brut = 4.05 + biais_local + rng.normal(0.0, 0.95, size=n)
        return np.clip(np.rint(brut), 1, 5).astype(int)

    note_accueil = note(base)
    note_pec = note(base + rng.normal(0.0, 0.25, size=n) - 0.12)

    # --- Motifs : renseignés plus souvent quand la note est extrême
    def motifs(notes: np.ndarray) -> list:
        txt = []
        for v in notes:
            if v <= 2 and rng.random() < 0.80:
                txt.append(str(rng.choice(MOTIFS_NEGATIFS)))
            elif v >= 4 and rng.random() < 0.55:
                txt.append(str(rng.choice(MOTIFS_POSITIFS)))
            elif rng.random() < 0.25:
                txt.append(str(rng.choice(MOTIFS_POSITIFS + MOTIFS_NEGATIFS)))
            else:
                txt.append(None)
        return txt

    # --- Suggestions : ~55 % des répondants en laissent une
    suggestions = [
        str(rng.choice(SUGGESTIONS)) if rng.random() < 0.55 else None for _ in range(n)
    ]

    df = pd.DataFrame(
        {
            "Horodateur": pd.Series(horodateur).dt.strftime("%d/%m/%Y %H:%M"),
            "Zone": zone_col,
            "Agence": agence_col,
            "Point de contact": rng.choice(POINTS_DE_CONTACT, size=n, p=P_POINTS),
            "Note de l'accueil": note_accueil,
            "Note de la prise en charge": note_pec,
            "Motifs de la note de l'accueil": motifs(note_accueil),
            "Motifs de la note de la prise en charge": motifs(note_pec),
            "Suggestions": suggestions,
        }
    )

    # Tri chronologique pour un fichier lisible
    ordre = pd.to_datetime(df["Horodateur"], format="%d/%m/%Y %H:%M").sort_values().index
    return df.loc[ordre].reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    df = generer()
    df.to_csv(SORTIE, index=False)
    print(f"{len(df)} lignes écrites dans {SORTIE}")
    print(df.head())
