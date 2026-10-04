# 📊 Dashboard Bancaire — Démo

Tableau de bord interactif d'analyse de la **satisfaction client d'un réseau
d'agences bancaires**, construit avec [Streamlit](https://streamlit.io) et
[Plotly](https://plotly.com/python/).

> ⚠️ **Données 100 % fictives.** Ce projet est une démonstration à vocation de
> portfolio : toutes les données (zones, agences, notes, commentaires) sont
> générées aléatoirement par `generate_synthetic_data.py` avec une graine fixe.
> Aucune donnée réelle de banque ou de client n'est utilisée.

## Contexte

Le dashboard simule l'exploitation d'une « boîte à idées digitale » : des
questionnaires de satisfaction remplis par les clients après un passage en
agence (note de l'accueil, note de la prise en charge, motifs, suggestions
libres). Il permet au pilotage qualité de suivre les performances du réseau,
de comparer zones et agences, et d'analyser les verbatims clients.

## Fonctionnalités

- **KPI annuels** : volume de questionnaires, notes moyennes d'accueil et de
  prise en charge, avec deltas personnalisables (décalage temporel ou norme
  cible) depuis la barre latérale.
- **KPI mensuels** : sélection d'une période glissante (mois à mois), volumes
  de questionnaires, d'avis et de suggestions avec variations.
- **Classements dynamiques** : palmarès des zones et des agences selon quatre
  critères (volumes ou notes moyennes).
- **Exploration de données** : filtre interactif multi-colonnes (numérique,
  catégoriel, dates, texte/regex) pour construire une base personnalisée.
- **Analyses graphiques** : diagramme circulaire, histogrammes, analyses
  croisées (numérique × numérique, catégoriel × catégoriel), évolution
  mensuelle en aires empilées — palette harmonisée et accessible (vérifiée
  pour le daltonisme).
- **Analyse textuelle des suggestions** : nettoyage du texte (minuscules,
  accents, stop words), nuage de mots, bigrammes les plus fréquents
  (histogramme + treemap).
- **Vues par périmètre** : vue « Direction » (réseau complet) ou restreinte à
  une zone.
- **Import optionnel** : possibilité d'analyser son propre fichier CSV/XLSX au
  même format.

## Stack technique

| Outil | Usage |
|---|---|
| Python 3.11+ | langage |
| Streamlit | interface web interactive (`st.metric`, cache, thème) |
| pandas / NumPy | manipulation et agrégation des données |
| Plotly | graphiques interactifs |
| WordCloud + Matplotlib | nuage de mots |
| Unidecode | normalisation du texte français |

## Installation et lancement

```bash
# 1. Cloner le dépôt puis installer les dépendances
pip install -r requirements.txt

# 2. (Optionnel) Regénérer les données synthétiques — graine fixe, reproductible
python generate_synthetic_data.py

# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501` et charge automatiquement le
jeu de données de démonstration `data/donnees_synthetiques.csv` (6 000
questionnaires fictifs répartis sur 2023–2025, 5 zones, 13 agences).

## Structure du projet

```
├── app.py                      # Point d'entrée : config, disclaimer, chargement des données
├── dashboard.py                # Logique du dashboard : KPI, classements, graphiques, NLP
├── generate_synthetic_data.py  # Générateur de données fictives (seed fixe)
├── data/
│   └── donnees_synthetiques.csv
├── .streamlit/
│   └── config.toml             # Thème clair sobre
└── requirements.txt
```

## Licence / usage

Projet de démonstration pour portfolio data science. Les données étant
synthétiques, le dépôt peut être réutilisé librement comme base d'un dashboard
d'enquête de satisfaction.

## Déploiement (Streamlit Community Cloud)

1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io) et connectez-vous avec votre compte GitHub.
2. Cliquez sur **New app**, puis choisissez ce dépôt, la branche à déployer et le fichier principal `app.py`.
3. Cliquez sur **Deploy** : l'application est construite puis mise en ligne sur une URL du type `https://<nom-de-l-appli>.streamlit.app`.

> **Version Python** : dans **Advanced settings** (avant le déploiement), choisissez une
> version récente de Python (3.11 ou 3.12), compatible avec les versions minimales
> listées dans `requirements.txt`.

### Éviter l'hibernation

Streamlit Community Cloud met l'application en veille après environ 12 heures sans
trafic (un visiteur tombe alors sur un écran « l'appli se réveille » pendant
plusieurs dizaines de secondes). Pour l'éviter, ce dépôt contient le workflow
GitHub Actions [`.github/workflows/keep-alive.yml`](.github/workflows/keep-alive.yml)
qui envoie un ping HTTP à l'application toutes les 4 heures (cron `33 */4 * * *`).

Après le déploiement, renseignez l'URL de l'appli dans une variable de dépôt :

1. Sur GitHub : **Settings → Secrets and variables → Actions → Variables → New repository variable**.
2. Name : `APP_URL` — Value : l'URL publique de l'appli (ex. `https://<nom-de-l-appli>.streamlit.app`).

Tant que `APP_URL` n'est pas définie, le workflow se termine sans rien faire (et sans
échouer). À noter : GitHub désactive les workflows planifiés après 60 jours sans activité
sur le dépôt ; il suffit alors de le relancer une fois manuellement via l'onglet
**Actions → Keep-alive Streamlit → Run workflow**.

Alternative sans GitHub Actions : créer un moniteur HTTP(S) gratuit sur
[UptimeRobot](https://uptimerobot.com) qui interroge l'URL de l'appli toutes les 5 minutes.
