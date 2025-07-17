# Dashboard de Crédit Scoring & Veille Technique NLP

## 1. Dashboard interactif de crédit scoring

### Objectif

Développer un dashboard interactif connecté à une API de prédiction pour :

- Visualiser le score de crédit et sa probabilité (proximité du seuil de décision)  
- Interpréter les résultats avec SHAP  
- Afficher les principales informations clients  
- Comparer un client à des groupes similaires  
- Respecter les critères d’accessibilité (WCAG)  
- Déployer en environnement cloud (Docker-ready)  

### Structure technique

```
├── api/
│   └── main.py                      # API FastAPI exposant le modèle
├── dashboard/
│   ├── dashboard.py                   # Application Streamlit
│   ├── api_data.pkl                   # Données d’entrée simulées pour l’API
│   ├── feature_dictionary.csv         # Dictionnaire des variables
│   ├── model_features_config.json     # Configuration des variables utilisées
│   ├── mon_pipeline_lgbm_opti.joblib  # Modèle de scoring optimisé
│   ├── variables_importantes.png      # Graphique des SHAP values
│   ├── requirements.txt               # Dépendances Python
│   ├── Dockerfile                     # Conteneurisation du dashboard
│   └── .dockerignore

```

### Accéder au dashboard en ligne
Le dashboard est accessible à l'adresse suivante :
👉 [https://mon-dashboard.azurewebsites.net](https://projet8-dashboard-h9ancgetd4cmbjdr.westeurope-01.azurewebsites.net/) (actuellement désactivé)

## 2. Veille technique – NLP pour classification de produits

### Objectif

Tester une technique récente de NLP (moins de 5 ans) pour automatiser la classification de produits à partir de descriptions textuelles.

### Données et modèles

- Jeu de données : descriptions produits e-commerce (Place de marché)  
- Modèles comparés :  
  - DistilBERT (Hugging Face, 2019)  
  - DeBERTa (Microsoft, 2021)  

### Fichiers associés

- `Sole_Johanna_2_notebook_veille_052025.ipynb` : notebook d’expérimentation  
- `Sole_Johanna_3_note_méthodologique_052025.pdf` : note méthodologique  

## Résultats clés

- Dashboard ergonomique facilitant l’interprétation des décisions de crédit  
- API fonctionnelle exposant un modèle de scoring  
- Preuve de concept NLP comparant DistilBERT et DeBERTa pour classification automatique :  
  - DistilBERT présente de bonnes performances et une bonne généralisation  
  - DeBERTa a des performances plus faibles dans cette configuration, mais l’analyse de ses embeddings suggère des pistes d’amélioration

## 📄 Auteurs

Projet réalisé par **Johanna Sole**, Data Scientist
Pour la formation Data Scientist - OpenClassrooms
