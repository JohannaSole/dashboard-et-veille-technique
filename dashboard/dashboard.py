import joblib
import streamlit as st
import requests
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard de scoring crédit", # Titre affiché dans l'onglet du navigateur
    layout="wide", # Largeur de la page étendue pour une meilleure lisibilité du dashboard
    page_icon="📊" # Icône de l'onglet
)

@st.cache_resource
def load_model():
    """
    Charge et met en cache le modèle LightGBM optimisé depuis un fichier joblib.
    Retourne le pipeline de prédiction.
    """
    pipeline = joblib.load("mon_pipeline_lgbm_opti.joblib")
    return pipeline

@st.cache_data
def load_data():
    """
    Charge et met en cache les données des clients à partir d’un fichier Pickle.
    Retourne un DataFrame contenant les données clients.
    """
    data = pd.read_pickle("api_data.pkl")
    return data

@st.cache_data
def load_config():
    """
    Charge et met en cache la configuration des features du modèle depuis un fichier JSON.
    Retourne un dictionnaire de configuration.
    """
    with open("model_features_config.json", "r") as f:
        config = json.load(f)
    return config

# Chargement du modèle, des données clients et de la configuration des variables :
pipeline = load_model()        # Modèle LightGBM optimisé chargé en mémoire
clients_data = load_data()     # Données clients utilisées pour l’analyse
model_config = load_config()   # Dictionnaire de configuration des features du modèle

# Chargement du dictionnaire depuis le CSV
dico_df = pd.read_csv("feature_dictionary.csv", encoding="utf-8", encoding_errors="ignore")
feature_dictionary = {}
for _, row in dico_df.iterrows():
    feature_dictionary[row["variable_name"]] = {
        "label": row["label"],
        "explication": row["explication"],
        "sens_risque": row["sens_risque"],
        "impact": row["impact"]
    }

def plot_gauge(pred_prob, threshold=0.51, title="Probabilité de défaut du client"):
    """
    Affiche une jauge colorée illustrant la probabilité de défaut prédite pour un client.

    Paramètres :
    - pred_prob (float) : Probabilité de défaut du client (entre 0 et 1).
    - threshold (float) : Seuil de décision à afficher sur la jauge (défaut = 0.51).
    - title (str) : Titre du graphique.

    La jauge est colorée du vert (faible risque) au rouge (fort risque),
    avec une ligne noire pour la probabilité du client et une ligne bleue pointillée pour le seuil.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cmap = plt.get_cmap('RdYlGn_r')
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, -0.5, 0.5])
    ax.plot([pred_prob, pred_prob], [-0.5, 0.5], color='black', lw=6)
    ax.plot([threshold, threshold], [-0.5, 0.5], color='#004080', lw=2, linestyle='--')
    ax.text(pred_prob, 0.55, f'{pred_prob:.3f}', ha='center', fontsize=16, color='black', fontweight='bold')
    plt.figtext(0.5, 0.03, f'Seuil ({threshold})', ha='center', fontsize=14, color='#004080', fontweight='bold')
    for i in np.linspace(0, 1, 11):
        ax.text(i, -0.6, f'{i:.1f}', ha='center', fontsize=12, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=16, fontweight='bold', pad=40)
    plt.box(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    st.pyplot(fig)

def plot_histogram_comparatif(df, client_row, feature, target_col='TARGET'):
    """
    Affiche deux histogrammes comparant la distribution d'une variable numérique 
    entre les clients solvables et non solvables, avec la valeur du client sélectionné.

    Paramètres :
    - df (pd.DataFrame) : Jeu de données complet contenant la variable à analyser.
    - client_row (pd.DataFrame) : Ligne correspondant au client sélectionné (une seule ligne).
    - feature (str) : Nom de la variable numérique à analyser.
    - target_col (str) : Colonne cible indiquant le défaut de paiement (0 = bon, 1 = défaut).

    L'histogramme affiche :
    - en vert : la distribution des clients solvables (TARGET=0),
    - en orange : la distribution des clients non solvables (TARGET=1),
    - une ligne rouge pointillée : la position de la valeur du client.
    """
    if not np.issubdtype(df[feature].dtype, np.number):
        st.warning(f"La variable {feature} n'est pas numérique, pas d'histogramme possible.")
        return
    if df[feature].nunique() < 2:
        st.warning(f"La variable {feature} est constante, pas d'histogramme possible.")
        return
    data_good = df[df[target_col] == 0][feature].dropna()
    data_bad = df[df[target_col] == 1][feature].dropna()
    client_value = client_row[feature].values[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    sns.histplot(data_good, bins=20, color="#228B22", ax=ax1, kde=True)
    ax1.axvline(client_value, color="#C70039", linestyle="--", label="Client")
    ax1.set_title("Clients solvables", fontsize=14, color="#222222")
    ax1.legend()
    sns.histplot(data_bad, bins=20, color="#FF8C00", ax=ax2, kde=True)
    ax2.axvline(client_value, color="#C70039", linestyle="--", label="Client")
    ax2.set_title("Clients non solvables", fontsize=14, color="#222222")
    ax2.legend()
    plt.suptitle(f"Distribution de {feature}", fontsize=16, color="#222222")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "<span style='font-size:16px;'>"
        "La <span style='color:#C70039; font-weight:bold;'>ligne rouge</span> indique la valeur du client sélectionné."
        "</span>",
        unsafe_allow_html=True
    )

# Affiche un titre centré en haut de l'application Streamlit avec un style HTML personnalisé.
st.markdown(
    """
    <h1 style='text-align: center;'>Dashboard de scoring crédit</h1>
    """,
    unsafe_allow_html=True
)

# Sélecteur dans la barre latérale pour choisir la taille du texte affiché dans l'application.
taille = st.sidebar.selectbox("Taille du texte :", ["Petit", "Normal", "Grand"])
taille_map = {"Petit": "14px", "Normal": "18px", "Grand": "22px"}

# Applique dynamiquement la taille de police sélectionnée à tous les éléments HTML de l'application.
st.markdown(
    f"<style>div, span, p {{ font-size: {taille_map[taille]} !important; }}</style>",
    unsafe_allow_html=True
)

# --- Gestion de l'état avec session_state ---
if "client_id" not in st.session_state:
    st.session_state.client_id = None

# Champ pour saisir l'identifiant du client, avec un minimum de 0 et des incréments de 1.
client_id_input = st.number_input("Entrez l'identifiant du client :", min_value=0, step=1, format="%d")

# Lorsque le bouton est cliqué, on enregistre l'ID du client dans la session.
if st.button("Afficher la décision et les explications"):
    st.session_state.client_id = int(client_id_input)

# Si un ID client est stocké en session, on envoie une requête API.
if st.session_state.client_id is not None:
    client_id = st.session_state.client_id
    API_URL = "https://projet8-plan-dghphvd9d9g4etbf.westeurope-01.azurewebsites.net/predict/"
    response = requests.post(API_URL, params={"client_id": int(client_id)})
    result = response.json()

    # Si l'API retourne une erreur, on l'affiche.
    if "error" in result:
        st.error(result["error"])
    else:
        st.success(f"Client numéro : {result['Client numéro']}")
        proba = float(re.findall(r"([\d\.]+)", result["risque_pourcentage"])[0]) / 100
        seuil = float(re.findall(r"\(([\d\.]+)\)", result["classe"])[0])
        
        # --- ENCADRÉ INFOS CLIENT ENRICHI ---
        client_data = clients_data[clients_data['SK_ID_CURR'] == client_id]

        if client_data.empty:
            st.warning("Client non trouvé dans les données locales.")
        else:
            try:
                # Infos de base
                genre = client_data["CODE_GENDER"].values[0]
                genre_str = "Femme" if genre == 1 else "Homme"
                age = int(abs(client_data["DAYS_BIRTH"].values[0]) / 365.25)
                enfants = int(client_data["CNT_CHILDREN"].values[0])
                emploi = int(abs(client_data["DAYS_EMPLOYED"].values[0]) / 365.25)
                revenu = int(client_data["AMT_INCOME_TOTAL"].values[0])
                credit = int(client_data["AMT_CREDIT"].values[0])
                annuite = int(client_data["AMT_ANNUITY"].values[0])

                # Situation familiale principale
                family_vars = [col for col in client_data.columns if col.startswith("NAME_FAMILY_STATUS_")]
                family_status = None
                for col in family_vars:
                    if client_data[col].values[0] == 1:
                        family_status = col.replace("NAME_FAMILY_STATUS_", "")
                        break

                # Type de logement principal
                housing_vars = [col for col in client_data.columns if col.startswith("NAME_HOUSING_TYPE_")]
                housing_type = None
                for col in housing_vars:
                    if client_data[col].values[0] == 1:
                        housing_type = col.replace("NAME_HOUSING_TYPE_", "")
                        break

                # Niveau d'études principal
                education_vars = [col for col in client_data.columns if col.startswith("NAME_EDUCATION_TYPE_")]
                education_type = None
                for col in education_vars:
                    if client_data[col].values[0] == 1:
                        education_type = col.replace("NAME_EDUCATION_TYPE_", "")
                        break

                # ----------- Affichage dans la sidebar -----------
                with st.sidebar:
                    st.markdown("### Informations du client")
                    st.write(f"**Sexe :** {genre_str}")
                    st.write(f"**Âge :** {age} ans")
                    st.write(f"**Enfants :** {enfants}")
                    st.write(f"**Ancienneté emploi :** {emploi} ans")
                    st.write(f"**Revenu annuel :** {revenu:,} €")
                    st.write(f"**Montant crédit :** {credit:,} €")
                    st.write(f"**Annuité :** {annuite:,} €")
                    st.write(f"**Situation familiale :** {family_status if family_status else 'Non renseigné'}")
                    st.write(f"**Type de logement :** {housing_type if housing_type else 'Non renseigné'}")
                    st.write(f"**Niveau d'études :** {education_type if education_type else 'Non renseigné'}")
                # ----------- Fin affichage sidebar -----------

            except Exception as e:
                st.error(f"Erreur lors de l'affichage des infos client : {e}")

            # Création des onglets dans la zone principale
            tab1, tab2 = st.tabs(["Vue client", "Analyses graphiques"])

            with tab1:
                st.header("Vue client")
                # --- SCORE ET PHRASE EXPLICATIVE ---
                plot_gauge(proba, threshold=round(seuil, 2), title="Probabilité de défaut du client")

                # Calcul de la distance au seuil
                if proba > seuil:
                    decision = "REFUSÉ"
                    couleur = "#C70039"
                    phrase = (
                        f"<b>Le crédit est <span style='color:{couleur}'>refusé</span>.</b> "
                        f"La probabilité de défaut ({proba:.3f}) dépasse le seuil ({seuil:.2f})."
                    )
                else:
                    decision = "ACCORDÉ"
                    couleur = "#228B22"
                    phrase = (
                        f"<b>Le crédit est <span style='color:{couleur}'>accordé</span>.</b> "
                        f"La probabilité de défaut ({proba:.3f}) est en-dessous du seuil ({seuil:.2f})."
                    )

                st.markdown(
                    f"<div style='font-size:18px; margin-bottom:10px;'>{phrase}</div>"
                    "<div style='font-size:16px; color:#FAFAFA;'>"
                    "Ce score représente la probabilité que le client ne rembourse pas son crédit. "
                    "Plus il est proche de 1, plus le risque est élevé. Le seuil est fixé par la politique de la banque."
                    "</div>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace

                # --- EXPLICATION SHAP ---
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<h2 style='text-align:center; margin-top:0px;'>"
                        "Variables importantes pour le modèle</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace

                    st.markdown(
                        "<span style='font-size:16px;'>"
                        "Le graphique ci-dessous montre les variables qui influencent le plus le modèle pour ses décisions.<br>"
                        "</span>",
                        unsafe_allow_html=True
                    )

                    st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace

                    st.image("variables_importantes.png", use_container_width=True)

                with col2:
                    features = client_data[model_config["expected_features"]]
                    features_preprocessed = pipeline[:-1].transform(features)
                    lgbm_model = pipeline.named_steps['model']
                    explainer = shap.TreeExplainer(lgbm_model)
                    shap_values = explainer.shap_values(features_preprocessed)
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        values = shap_values[1][0]
                        base_value = explainer.expected_value[1]
                    else:
                        values = shap_values[0]
                        base_value = explainer.expected_value
                
                    st.markdown(
                        f"<h2 style='text-align:center; margin-top:0px;'>"
                        "Pourquoi ce score&nbsp;?</h2>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "<span style='font-size:16px;'>"
                        "Le graphique ci-dessous montre les variables qui ont le plus influencé la décision pour ce client.<br>"
                        "<span style='color:#C70039;'>Barres rouges</span> : augmentent le risque<br>"
                        "<span style='color:#1E88E5;'>Barres bleues</span> : diminuent le risque"
                        "</span>",
                        unsafe_allow_html=True
                    )
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(
                        shap.Explanation(
                            values=values,
                            base_values=base_value,
                            feature_names=features.columns
                        ),
                        max_display=10, show=False
                    )
                    st.pyplot(fig)

                # --- TABLEAU EXPLICATIF ---
                shap_importance = pd.Series(np.abs(values), index=features.columns).sort_values(ascending=False)
                top10 = shap_importance.head(10).index.tolist()

                rows = []
                for var in top10:
                    val_client = client_data[var].values[0]
                    shap_val = values[features.columns.get_loc(var)]
                    effet = "🟢 Diminue le risque" if shap_val < 0 else "🔴 Augmente le risque"
                    label = feature_dictionary.get(var, {}).get("label", var)
                    explication = feature_dictionary.get(var, {}).get("explication", "")
                    sens_risque = feature_dictionary.get(var, {}).get("impact", "")

                    # Construction du tooltip HTML
                    tooltip = (
                        f"<span style='font-size:14px; font-weight:bold;'>Définition :</span> {explication}<br>"
                        f"<span style='font-size:14px; font-weight:bold;'>Impact :</span> {sens_risque}"
                    )
                    # Icône "?" avec info-bulle (tooltip natif navigateur)
                    var_with_tooltip = (
                        f"{var} "
                        f"<span style='cursor: pointer; color: #1E88E5;' "
                        f"title=\"Définition : {explication.replace('\"', '&quot;')} (Impact : {sens_risque})\">&#9432;</span>"
                    )

                    rows.append({
                        "Variable": var_with_tooltip,
                        "Variable en français": label,
                        "Valeur du client": round(val_client, 2) if isinstance(val_client, (int, float, np.integer, np.floating)) else val_client,
                        "Effet sur la prédiction": effet
                    })

                st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace
                df_explications = pd.DataFrame(rows)

                st.markdown(
                    "<h2 style='text-align:center; margin-top:0px;'>Résumé des principales variables et de leur impact</h2>",
                    unsafe_allow_html=True
                )

                # Affichage du tableau avec info-bulle (HTML dans la colonne Variable)
                st.write(
                    df_explications.to_html(escape=False, index=False),
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace

            with tab2:
                st.header("Analyses graphiques")

                # --- Spider plot ---
                # Exclure 'code_gender' de la liste des variables affichées sur le radar plot
                top10 = [var for var in top10 if var.lower() != "code_gender"]

                # Normalisation min-max pour chaque variable du radar
                norm_clients_data = clients_data.copy()
                for var in top10:
                    min_val = norm_clients_data[var].min()
                    max_val = norm_clients_data[var].max()
                    # Pour éviter la division par zéro si min = max
                    if min_val == max_val:
                        norm_clients_data[var] = 0.5
                    else:
                        norm_clients_data[var] = (norm_clients_data[var] - min_val) / (max_val - min_val)

                # Normaliser aussi les valeurs du client
                norm_client_data = client_data.copy()
                for var in top10:
                    min_val = clients_data[var].min()
                    max_val = clients_data[var].max()
                    if min_val == max_val:
                        norm_client_data[var] = 0.5
                    else:
                        norm_client_data[var] = (norm_client_data[var] - min_val) / (max_val - min_val)

                # Récupérer les valeurs du client pour les variables top10
                client_vals = [norm_client_data[var].values[0] for var in top10]

                # Calculer la moyenne population pour ces variables
                mean_vals = [norm_clients_data[var].mean() for var in top10]

                # Ajouter le premier point à la fin pour fermer le polygone
                client_vals += [client_vals[0]]
                mean_vals += [mean_vals[0]]
                radar_vars = top10 + [top10[0]]

                # Créer le radar Plotly
                fig = go.Figure(layout=dict(
                    width=1000,
                    height=800
                ))

                fig.add_trace(go.Scatterpolar(
                    r=client_vals,
                    theta=radar_vars,
                    fill='toself',
                    name='Client',
                    line_color='red'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=mean_vals,
                    theta=radar_vars,
                    fill='toself',
                    name='Moyenne population',
                    line_color='blue'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True)
                    ),
                    showlegend=True
                )

                # Affichage dans Streamlit
                st.markdown(
                    f"<h2 style='text-align:center; margin-top:0px;'>"
                    "Synthèse multi-variables du profil client</h2>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    "<span style='font-size:16px;'>"
                    "Ce graphique compare le profil du client (en rouge) à la moyenne de la population (en bleu) sur plusieurs critères importants.<br>"
                    "Plus la zone rouge s’éloigne de la zone bleue pour un critère, plus le client est différent de la moyenne sur ce point."
                    "</span>",
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace

                # --- HISTOGRAMME COMPARATIF ---
                st.markdown(
                    f"<h2 style='text-align:center; margin-top:0px;'>"
                    "Distribution des variables chez les clients solvables et non solvables</h2>",
                    unsafe_allow_html=True
                )

                numeric_top10 = [f for f in top10 if np.issubdtype(clients_data[f].dtype, np.number)]
                if numeric_top10:
                    if "selected_feature" not in st.session_state:
                        st.session_state.selected_feature = numeric_top10[0]
                    selected_feature = st.selectbox(
                        "Sélectionner une variable pour comparer",
                        numeric_top10,
                        key="feature_selectbox",
                        index=numeric_top10.index(st.session_state.selected_feature)
                    )
                    st.session_state.selected_feature = selected_feature
                    plot_histogram_comparatif(clients_data, client_data, selected_feature)
                    st.markdown(
                        "<span style='font-size:16px;'>"
                        "Chaque histogramme compare la valeur du client (ligne rouge) à la population des clients solvables et non solvables."
                        "</span>",
                        unsafe_allow_html=True
                    )    
                    st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace

                # ---- Scatter plot intéractif ---

                st.markdown(
                    f"<h2 style='text-align:center; margin-top:0px;'>"
                    "Comparaison bivariée sur les variables les plus importantes pour le client</h2>",
                    unsafe_allow_html=True
                )
                # Sélecteurs pour choisir les variables à afficher
                x_var = st.selectbox("Variable en abscisse", options=numeric_top10, key="scatter_x")
                y_var = st.selectbox("Variable en ordonnée", options=[v for v in numeric_top10 if v != x_var], key="scatter_y")

                # Convertir TARGET en chaîne de caractères avant de passer à Plotly 
                clients_data['TARGET_STR'] = clients_data['TARGET'].map({0: 'Solvable', 1: 'Insolvable'})

                # Construction du scatter plot
                fig = px.scatter(
                    clients_data,
                    x=x_var,
                    y=y_var,
                    color="TARGET_STR", 
                    color_discrete_map={
                        'Solvable': "#228B22",      # Vert
                        'Insolvable': "#C70039"     # Rouge
                    },
                    opacity=0.6,
                    title=f"Nuage de points : {x_var} vs {y_var}",
                    labels={x_var: x_var, y_var: y_var}
                )

                # Mettre en évidence le client sélectionné
                client_row = clients_data[clients_data["SK_ID_CURR"] == client_id]
                if not client_row.empty:
                    fig.add_scatter(
                        x=client_row[x_var],
                        y=client_row[y_var],
                        mode="markers",
                        marker=dict(size=16, color="red", line=dict(width=2, color="black")),
                        name="Client sélectionné"
                    )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    "<span style='font-size:16px;'>"
                    "<b>Légende :</b><br>"
                    "<span style='color:#228B22;'>●</span> Client solvable (TARGET = 0)<br>"
                    "<span style='color:#C70039;'>●</span> Client insolvable (TARGET = 1)<br>"
                    "<span style='color:red;'>●</span> Client analysé (point rouge vif)"
                    "</span>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)  # Ajoute de l'espace
                st.markdown(
                    "<span style='font-size:16px;'>"
                    "Ce nuage de points montre la relation entre les deux variables sélectionnées pour l’ensemble des clients.<br>"
                    "Chaque point correspond à un client, coloré selon son statut (solvable ou en défaut).<br>"
                    "Le point rouge indique la position du client analysé, permettant de visualiser où il se situe par rapport à la population."
                    "</span>",
                    unsafe_allow_html=True
                ) 