"""
Script de setup : .env d'exemple, téléchargement des données Olist.
Usage: uv run python scripts/setup.py
"""

import urllib.request
from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR, DataConfig, ROOT_DIR


def download_olist_data():
    """Télécharge les données Olist depuis GitHub."""
    print("Téléchargement des données Olist...")
    for _name, filename in DataConfig.DATASETS.items():
        url = DataConfig.OLIST_BASE_URL + filename
        local_path = RAW_DATA_DIR / filename
        if local_path.exists():
            print(f"{filename} existe déjà")
            continue
        try:
            print(f"  Téléchargement {filename}...")
            urllib.request.urlretrieve(url, local_path)
            df = pd.read_csv(local_path, nrows=5)
            print(f"  {filename} - {len(df.columns)} colonnes")
        except Exception as e:
            print(f"  Erreur téléchargement {filename}: {e}")
    print("  Données téléchargées avec succès!\n")


def create_sample_env():
    """Crée .env.example à la racine du projet."""
    print("Création du fichier .env d'exemple...")
    env_content = """# OLIST RECOMMENDATION SYSTEM - ENVIRONMENT
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True
MODEL_VERSION=1.0.0
RETRAIN_THRESHOLD=0.1
LOG_LEVEL=INFO
"""
    env_path = ROOT_DIR / ".env.example"
    env_path.write_text(env_content, encoding="utf-8")
    print(f"  OK {env_path.relative_to(ROOT_DIR)} cree")
    print("  Copiez vers .env et adaptez.\n")


def main():
    print("=" * 50)
    print("OLIST RECOMMENDATION SYSTEM - SETUP")
    print("=" * 50 + "\n")
    create_sample_env()
    download_olist_data()
    print("=" * 50)
    print("SETUP TERMINÉ AVEC SUCCÈS!")
    print("=" * 50 + "\n")
    print("PROCHAINES ÉTAPES:")
    print("  1. uv run python scripts/train.py")
    print("  2. uv run uvicorn backend.app.main:app --reload")
    print("  3. uv run streamlit run frontend/app.py")
    print("\nHappy coding! :)")


if __name__ == "__main__":
    main()
