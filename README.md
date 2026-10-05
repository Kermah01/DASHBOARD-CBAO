# 📊 Tableau de bord interactif de la boîte à idées digitale — Démo

Tableau de bord interactif d'analyse de la **satisfaction client d'un réseau
d'agences bancaires** (qualité de service), construit avec
[Streamlit](https://streamlit.io) et [Plotly](https://plotly.com/python/).

> ⚠️ **Données 100 % fictives.** Ce dépôt est la version de démonstration
> (portfolio) d'un dashboard de pilotage qualité : le design d'origine est
> conservé (page de connexion animée, fond néon, barre latérale noire et or),
> mais toutes les données (zones, agences, notes, commentaires) sont générées
> aléatoirement par `generate_synthetic_data.py` avec une graine fixe. Aucune
> donnée réelle de banque ou de client n'est utilisée.

**Démo en ligne :** https://dashboard-cbao.streamlit.app

## Comptes de démonstration

Cliquez sur **LOGIN** sur la page d'accueil, puis connectez-vous avec :

| Identifiant | Mot de passe | Périmètre |
|---|---|---|
| `direction` | `demo` | toutes les zones |
| `agence` | `demo` | Zone Centre uniquement |

## Contexte

Le dashboard exploite une « boîte à idées digitale » : des questionnaires de
satisfaction remplis par les clients après un passage en agence (note de
l'accueil, note de la prise en charge, motifs, suggestions libres). Il permet
au pilotage qualité de suivre les performances du réseau, de comparer zones et
agences, et d'analyser les verbatims clients.

## Fonctionnalités

- **Page de connexion** avec profils : la direction voit tout le réseau, une
  agence ne voit que sa zone.
- **KPI annuels** : volume de questionnaires, notes moyennes d'accueil et de
  prise en charge, avec deltas personnalisables (décalage temporel ou norme)
  depuis la barre latérale.
- **KPI mensuels** : sélection d'une période (mois à mois), volumes de
  questionnaires, d'avis et de suggestions.
- **Classements** des zones et des agences selon quatre critères.
- **Base de données personnalisée** : filtre interactif multi-colonnes.
- **Analyses graphiques** : camembert, histogramme, nuage de points,
  histogramme croisé (empilé / étalé), évolution mensuelle en aires.
- **Analyse textuelle des suggestions** : nettoyage du texte, nuage de mots,
  bigrammes les plus fréquents (histogramme + treemap).

## Stack technique

| Outil | Usage |
|---|---|
| Python 3.12 / 3.13 / 3.14 | langage |
| Streamlit 1.44 | interface web interactive |
| pandas / NumPy | manipulation et agrégation des données |
| Plotly | graphiques interactifs |
| WordCloud + Matplotlib | nuage de mots |
| NLTK + Unidecode | bigrammes et normalisation du texte français |
| openpyxl | lecture de fichiers Excel téléversés |

## Installation et lancement

```bash
# 1. Cloner le dépôt puis installer les dépendances (versions épinglées)
pip install -r requirements.txt

# 2. (Optionnel) Regénérer les données synthétiques — graine fixe, reproductible
python generate_synthetic_data.py

# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501` et charge le jeu de données
de démonstration `data/donnees_synthetiques.csv` (6 000 questionnaires fictifs
répartis sur 2023–2025, 5 zones, 13 agences). Les images de fond sont chargées
depuis leurs URL d'origine (une connexion Internet est nécessaire pour les
afficher).

## Structure du projet

```
├── app.py                      # Point d'entrée : page de connexion (fond GIF animé)
├── dashboard.py                # Dashboard : KPI, classements, graphiques, analyse textuelle
├── generate_synthetic_data.py  # Générateur de données fictives (seed fixe)
├── data/
│   └── donnees_synthetiques.csv
├── .streamlit/
│   └── config.toml             # Thème sombre d'origine
└── requirements.txt            # Versions épinglées (==)
```

## Licence / usage

Projet de démonstration pour portfolio data science. Les données étant
synthétiques, le dépôt peut être réutilisé librement comme base d'un dashboard
d'enquête de satisfaction.

## Déploiement (Streamlit Community Cloud)

1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io) et connectez-vous avec votre compte GitHub.
2. Cliquez sur **New app**, puis choisissez ce dépôt, la branche à déployer et le fichier principal `app.py`.
3. Cliquez sur **Deploy** : l'application est construite puis mise en ligne sur une URL du type `https://<nom-de-l-appli>.streamlit.app`.

> **Version Python** : Streamlit Cloud utilise Python 3.14 par défaut (modifiable dans
> **Advanced settings**). Les versions de `requirements.txt` sont épinglées (`==`) et ont
> été testées sous Python 3.12, 3.13 et 3.14 : le rendu en ligne est identique au rendu local.

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
