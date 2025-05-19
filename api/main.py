# Importer les bibliothèques nécessaires
import os  # Pour interagir avec le système d'exploitation, comme les fichiers et répertoires
import joblib  # Pour charger et sauvegarder les modèles de machine learning
import pandas as pd  # Pour manipuler les données sous forme de DataFrame (structure de données utilisée pour le traitement de données)
from fastapi import FastAPI  # FastAPI est utilisé pour créer l'API Web
import datetime  # Pour travailler avec les dates et heures
import json  # Pour lire et écrire des fichiers JSON

# Charger la configuration du modèle depuis un fichier JSON
with open("model_features_config.json", "r") as f:
    model_config = json.load(f)

# Définition des chemins des fichiers et dossiers contenant le modèle et le seuil optimal
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "mon_pipeline_lgbm_opti.joblib")
optimal_threshold_path = os.path.join(os.path.dirname(__file__), "..", "optimal_threshold.joblib")

# Charger le pipeline et le seuil optimal
try:
    # Charger le modèle de machine learning pré-entraîné
    pipeline = joblib.load(pipeline_path)
    print("Pipeline chargé avec succès!")  # Affichage d'un message pour indiquer le succès du chargement
    
    # Charger le seuil optimal permettant de prendre une décision finale
    optimal_threshold = joblib.load(optimal_threshold_path)
    print("Seuil optimal chargé avec succès!")  # Affichage d'un message pour indiquer le succès du chargement
except FileNotFoundError as e:
    print(f"Fichier introuvable : {e}")  # Erreur si l'un des fichiers n'est pas trouvé
except Exception as e:
    print(f"Une erreur est survenue : {e}")  # Gestion d'autres types d'erreurs inattendues

# Charger les données des clients
try:
    # Définir le chemin des données clients stockées sous forme de fichier pickle
    data_path = os.path.join(os.path.dirname(__file__), "..", "api_data.pkl")
    
    # Charger les données des clients
    clients_data = pd.read_pickle(data_path)
    print("Données clients chargées avec succès!")  # Message indiquant le succès du chargement
    print(f"Chemin utilisé : {data_path}")  # Afficher le chemin du fichier utilisé (utile pour le débogage)
except Exception as e:
    print(f"ERREUR CRITIQUE : {str(e)}")  # Affichage d'un message d'erreur critique
    clients_data = None  # Mettre à None pour signaler que les données ne sont pas disponibles
    raise  # Lever l'exception pour interrompre l'exécution en cas d'erreur critique

# Initialiser l'application FastAPI
app = FastAPI()

# Définition d'une route pour la page d'accueil de l'API
@app.get("/")  # Route GET à l'adresse "/"
def root():
    """Retourne un message d'accueil lorsqu'on accède à la racine de l'API."""
    return {"message": "Bienvenue sur l'API FastAPI avec modèle prédictif !"}

# Définition d'une route pour obtenir des informations sur le modèle
@app.get("/model-info")  # Route GET à l'adresse "/model-info"
def get_model_info():
    """Retourne des informations sur l'état du modèle, comme s'il est chargé et à quelle heure."""
    info = {
        "model_loaded": pipeline is not None,  # Indique si le modèle est chargé ou non
        "loaded_at": datetime.datetime.now().isoformat() if pipeline else "N/A"  # Affiche l'heure actuelle si le modèle est chargé
    }
    return info

# Définition d'une route pour effectuer une prédiction
@app.post("/predict/")
def predict(client_id: int):
    """
    Effectue une prédiction pour un client donné en fonction de son ID.
    
    Paramètres :
        - client_id (int) : Identifiant unique du client

    Retour :
        - Un dictionnaire contenant le pourcentage de risque et la classe prédite
    """
    try:
        if clients_data is None:
            return {"error": "Données clients non disponibles"}  # Vérifie si les données clients ont été chargées
        
        # Extraire les données du client correspondant à l'ID donné
        client_data = clients_data[clients_data['SK_ID_CURR'] == client_id]
        
        if client_data.empty:
            return {"error": f"Client {client_id} introuvable"}  # Vérifie si le client existe bien dans les données
        
        # Sélectionner uniquement les colonnes attendues par le modèle
        features = client_data[model_config["expected_features"]]
        
        # Vérification pour s'assurer que les colonnes correspondent bien aux attentes du modèle
        assert features.columns.tolist() == model_config["expected_features"], \
            "Les colonnes ne correspondent pas aux attentes du modèle"
        
        # Faire une prédiction de probabilité avec le modèle
        proba = pipeline.predict_proba(features)[0][1]  # Extraire la probabilité de la classe positive (risque)
        
        # Appliquer le seuil optimal pour classifier le client
        classe_predite = 1 if proba >= optimal_threshold else 0
        
        # Convertir la probabilité en pourcentage
        risque_pourcentage = proba * 100
        
        # Construire le résultat sous forme d'un dictionnaire
        resultat = {
            "Client numéro": client_id,
            "risque_pourcentage": f"Il y a {risque_pourcentage:.1f}% de risques que le client ait des difficultés de paiement",
            "classe": f"Classe après application du seuil optimisé ({optimal_threshold}): {classe_predite}"
        }
        return resultat  # Retourner le résultat sous forme de JSON
        
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}  # Gérer les erreurs et retourner un message d'erreur
