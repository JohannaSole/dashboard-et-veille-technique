## 📊 Dashboard de Crédit Scoring & Veille Technique NLP

Bienvenue dans ce dépôt contenant deux livrables complémentaires réalisés dans le cadre du **Projet 8 - OpenClassrooms : Réalisez un dashboard et assurez une veille technique**.

---

### Contexte

Vous êtes **Data Scientist chez "Prêt à dépenser"**, une entreprise de crédits à la consommation pour des personnes avec peu ou pas d'historique de prêt.

Votre mission comporte **deux volets** :

1. **Développer un dashboard de crédit scoring** pour aider les chargés de relation client à expliquer les décisions d’octroi de crédit.
2. **Effectuer une veille technique** sur un modèle récent de traitement de données textuelles.

---

##  1. Dashboard interactif de crédit scoring

### 🎯 Objectif

Créer un dashboard interactif connecté à une **API de prédiction** afin de :

* Visualiser le score et sa probabilité (proximité du seuil de décision),
* Interpréter les résultats de manière intelligible (SHAP),
* Afficher les principales informations d’un client,
* Comparer ce client à des groupes similaires,
* Respecter des critères d’**accessibilité (WCAG)**,
* Être déployé sur le Cloud (Docker-ready).

###  Structure technique

```
├── api/
│   └── main.py                      # API FastAPI exposant le modèle
├── dashboard/
│   ├── dashboard.py                   # Application Streamlit
│   ├── main.py                        # API FastAPI exposant le modèle
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

##  2. Veille technique – NLP pour classification de produits

### 🎯 Objectif

Tester une **technique récente de NLP (moins de 5 ans)** pour automatiser la classification de produits à partir de leur description textuelle.

### Détails

* **Jeu de données** : Descriptions de produits e-commerce (Place de marché)
* **Objectif** : Automatiser la catégorisation des produits
* **Modèles comparés** :

  * DistilBERT (Hugging Face, 2019)
  * DeBERTa (Microsoft, 2021)

###  Fichiers associés

* `Sole_Johanna_2_notebook_veille_052025.ipynb` : Notebook de veille et expérimentation
* `Sole_Johanna_3_note_méthodologique_052025.pdf` : Note méthodologique accompagnant la veille

---

## ✅ Résultats clés

* Un **dashboard ergonomique** permettant une interprétation simple des décisions d’octroi de crédit.  
* Une **API fonctionnelle** exposant un modèle de scoring.  
* Une preuve de concept NLP comparant deux modèles pré-entraînés, DistilBERT et DeBERTa, pour la classification automatique de descriptions produits :  

  - DistilBERT offre de bonnes performances et une capacité satisfaisante de généralisation.  
  - DeBERTa, dans la configuration actuelle, présente des performances plus faibles mais une analyse fine de ses embeddings ouvre des pistes pour des améliorations futures.

## 📄 Auteurs

Projet réalisé par **Johanna Sole**, Data Scientist
Pour la formation Data Scientist - OpenClassrooms
