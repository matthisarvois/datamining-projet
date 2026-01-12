# ===============================================
# OLIST RECOMMENDATION SYSTEM - TRAINING
# Master 2 - SEP
# ===============================================

"""
Script d'entraînement du modèle de recommandation.

Ce script:
1. Effectue le feature engineering
2. Entraîne le modèle RandomForest
3. Évalue les performances
4. Sauvegarde le modèle pour l'API
"""

import argparse
import sys
from pathlib import Path

from ml_pipeline.models.recommendation_model import RecommendationPipeline

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR


def check_data_availability(data_dir: Path) -> bool:
    """
    Vérifie que toutes les données nécessaires sont disponibles.

    Args:
        data_dir: Répertoire des données

    Returns:
        True si toutes les données sont présentes
    """
    required_files = [
        'olist_customers_dataset.csv',
        'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_products_dataset.csv'
    ]

    missing_files = []
    for filename in required_files:
        if not (data_dir / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"Fichiers manquants: {missing_files}")
        return False

    print("Toutes les données requises sont présentes")
    return True



def train_recommendation_model(data_dir: Path) -> dict:
    """
    Entraîne le modèle de recommandation.

    Args:
        data_dir: Répertoire contenant les données

    Returns:
        Métriques de performance
    """
    print("Démarrage de l'entraînement du modèle...")

    # Initialiser le pipeline
    pipeline = RecommendationPipeline()

    # Entraîner
    metrics = pipeline.train_pipeline(data_dir)

    return metrics


def display_training_results(metrics: dict):
    """
    Affiche les résultats d'entraînement de manière lisible.

    Args:
        metrics: Métriques de performance
    """
    print("\n" + "=" * 50)
    print("RÉSULTATS D'ENTRAÎNEMENT")
    print("" + "=" * 50)

    print(f"Précision d'entraînement: {metrics['train_accuracy']:.3f}")
    print(f"Précision de test: {metrics['test_accuracy']:.3f}")
    print(f"Score AUC: {metrics['auc_score']:.3f}")
    print(f"Cross-validation: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

    # Interprétation des résultats
    auc = metrics['auc_score']
    if auc >= 0.9:
        performance = "Excellente"
    elif auc >= 0.8:
        performance = "Très bonne"
    elif auc >= 0.7:
        performance = "Bonne"
    elif auc >= 0.6:
        performance = "Correcte"
    else:
        performance = "À améliorer"

    print(f"Performance globale: {performance}")

    # Conseils pédagogiques
    print("\n" + "=" * 50)
    print("CONSEILS")
    print("" + "=" * 50)

    if auc < 0.7:
        print("🔧 Suggestions d'amélioration:")
        print("   • Ajouter plus de features (interactions temporelles, etc.)")
        print("   • Ajuster les hyperparamètres du RandomForest")
        print("   • Équilibrer davantage les données d'entraînement")
        print("   • Essayer d'autres algorithmes (XGBoost, LightGBM)")

    print("Points d'apprentissage:")
    print("   • Observer l'importance des features")
    print("   • Analyser la matrice de confusion")
    print("   • Tester avec différents seuils de probabilité")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle de recommandation Olist"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(RAW_DATA_DIR),
        help=f'Répertoire des données (défaut: {RAW_DATA_DIR})'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("" + "=" * 60)
    print("OLIST RECOMMENDATION SYSTEM - ENTRAÎNEMENT")
    print("Master 2 - SEP")
    print("" + "=" * 60)

    print(f"Répertoire de données: {data_dir}")

    # 1. Vérifier/générer les données
    if not check_data_availability(data_dir):
        print("\n❌ Données manquantes. Options:")
        print("   1. Télécharger les vraies données Olist")
        return 1

    # 2. Entraîner le modèle
    metrics = train_recommendation_model(data_dir)

    # 3. Afficher les résultats
    display_training_results(metrics)

    # 4. Instructions de suivi
    print("\n" + "=" * 50)
    print("PROCHAINES ÉTAPES")
    print("" + "=" * 50)
    print("1. Lancer l'API: uvicorn backend.app.main:app --reload")
    print("2. Lancer le frontend: streamlit run frontend/app.py")
    print("3. Tester les recommandations via l'interface web")
    print("4. Analyser les features importantes")
    print("5. Optimiser les hyperparamètres")

    print("\n✨ Entraînement terminé avec succès! ✨")




if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
