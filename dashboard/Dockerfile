# Utiliser l'image officielle Python 3.12
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y libgomp1

# Définit le dossier de travail dans le conteneur
WORKDIR /app

# Copie les fichiers requirements.txt et le code dans le conteneur
COPY requirements.txt ./
COPY . .

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port par défaut de Streamlit
EXPOSE 8501

# Commande de lancement de l'app Streamlit
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
