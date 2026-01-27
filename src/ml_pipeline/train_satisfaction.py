# ===============================================
# OLIST - ENTRAÎNEMENT MODÈLE DE SATISFACTION
# Master 2 - SEP
# ===============================================

"""
Script d'entraînement du modèle de satisfaction client.

1. Charge les données (customers, orders, order_items, reviews)
2. Construit les features (délai livraison, nb articles, montant, historique client)
3. Entraîne un RandomForest en régression (cible: review_score 1–5)
4. Sauvegarde dans data/models/satisfaction_model.joblib (même emplacement que recommendation_model)
"""

import argparse
import sys
from pathlib import Path

from src.config import RAW_DATA_DIR
from src.ml_pipeline.models.satisfaction_model import SatisfactionPipeline


def check_data_availability(data_dir: Path) -> bool:
    """Vérifie que les données nécessaires (dont reviews) sont présentes."""
    required = [
        "olist_customers_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_reviews_dataset.csv",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        print(f"Fichiers manquants: {missing}")
        return False
    print("Toutes les données requises pour la satisfaction sont présentes")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle de satisfaction client (review_score 1–5)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"Répertoire des données brutes (défaut: {RAW_DATA_DIR})",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("OLIST - ENTRAÎNEMENT MODÈLE DE SATISFACTION CLIENT")
    print("=" * 60)
    print(f"Répertoire de données: {data_dir}\n")

    if not check_data_availability(data_dir):
        print("\n❌ Données manquantes. Lancez d’abord: uv run python src/scripts/setup.py")
        return 1

    pipeline = SatisfactionPipeline()
    metrics = pipeline.train_pipeline(data_dir)

    print("\n" + "=" * 50)
    print("RÉSULTATS SATISFACTION")
    print("=" * 50)
    print(f"R² train:    {metrics.get('train_r2', 0):.3f}")
    print(f"R² test:     {metrics.get('test_r2', 0):.3f}")
    print(f"MAE test:    {metrics.get('test_mae', 0):.3f}")
    print(f"RMSE test:   {metrics.get('test_rmse', 0):.3f}")
    print(f"CV R²:       {metrics.get('cv_r2_mean', 0):.3f} ± {metrics.get('cv_r2_std', 0):.3f}")
    print("\nModèle sauvegardé dans data/models/satisfaction_model.joblib")
    print("Vous pouvez l’utiliser ensuite dans Streamlit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
