# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - SETUP
# Master 2 - SEP
# ===============================================

"""
Script de setup pour configurer l'environnement du projet.

Usage:
    python setup.py

Ce script va:
2. Télécharger les données Olist
4. Initialiser les logs
"""

import urllib.request
import pandas as pd
from pathlib import Path
from config import DataConfig, RAW_DATA_DIR


def download_olist_data():
    """Télécharge les données Olist depuis GitHub."""
    print("Téléchargement des données Olist...")

    for name, filename in DataConfig.DATASETS.items():
        url = DataConfig.OLIST_BASE_URL + filename
        local_path = RAW_DATA_DIR / filename

        if local_path.exists():
            print(f"{filename} existe déjà")
            continue

        try:
            print(f" Téléchargement {filename}...")
            urllib.request.urlretrieve(url, local_path)

            # Vérifier que le fichier est bien un CSV valide
            df = pd.read_csv(local_path, nrows=5)
            print(f" {filename} - {len(df.columns)} colonnes")

        except Exception as e:
            print(f" Erreur téléchargement {filename}: {e}")

    print(" Données téléchargées avec succès!\n")

def create_sample_env():
    """Crée un fichier .env d'exemple."""
    print("Création du fichier .env d'exemple...")

    env_content = """# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - ENVIRONMENT
# ===============================================

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True

# ML Model Settings
MODEL_VERSION=1.0.0
RETRAIN_THRESHOLD=0.1

# Logging
LOG_LEVEL=INFO

"""

    env_path = Path(".env.example")
    with open(env_path, "w", encoding='utf-8') as f:
        f.write(env_content)

    print(f"   ✅ Fichier {env_path} créé")
    print("   📝 Copiez-le vers .env et adaptez selon vos besoins\n")


def main():
    """Fonction principale de setup."""
    print("" + "=" * 50)
    print("OLIST RECOMMENDATION SYSTEM - SETUP")
    print("Master 2 - SEP")
    print("" + "=" * 50 + "\n")

    # Étapes de setup
    create_sample_env()

    # Télécharger les données
    download_olist_data()

    print("" + "=" * 50)
    print("SETUP TERMINÉ AVEC SUCCÈS!")
    print("" + "=" * 50 + "\n")

    print("PROCHAINES ÉTAPES:")
    print("   1. uv run python ml_pipeline/train_model.py")
    print("   2. uvicorn backend.app.main:app --reload")
    print("   3. streamlit run frontend/app.py")
    print("\n Happy coding! :)")


if __name__ == "__main__":
    main()
