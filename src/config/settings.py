"""Paramètres et chemins du projet."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

for _d in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True, parents=True)


class MLConfig:
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    }
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    RECOMMENDATION_MODEL_FILE = MODELS_DIR / "recommendation_model.joblib"
    FEATURE_PIPELINE_FILE = MODELS_DIR / "feature_pipeline.joblib"
    CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "customer_features.csv"
    PRODUCT_FEATURES_FILE = RAW_DATA_DIR / "olist_products_dataset.csv"


class APIConfig:
    HOST = "127.0.0.1"
    PORT = 8000
    TITLE = "Olist Recommendation System API"
    VERSION = "1.0.0"
    DESCRIPTION = """
    🛒 **Olist Recommendation System API**

    Cette API fournit des recommandations personnalisées basées sur:
    - L'historique d'achat des clients
    - Les similarités entre produits
    - Les préférences par catégorie

    **Fonctionnalités:**
    - Recommandations personnalisées par client
    - Prédiction de probabilité d'achat
    - Métriques de performance du modèle
    """


class StreamlitConfig:
    PAGE_TITLE = "🛒 Olist Recommendation System"
    PAGE_ICON = "🛒"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"


class DataConfig:
    OLIST_BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"
    DATASETS = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
    }
    CUSTOMER_FEATURES = [
        "total_orders",
        "total_spent",
        "avg_order_value",
        "days_since_last_order",
        "favorite_category",
        "avg_review_score",
        "unique_products_bought",
    ]
    HIGH_VALUE_THRESHOLD = 500
    FREQUENT_BUYER_THRESHOLD = 5


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[{asctime}] {levelname:<8} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "olist_recommendation.log"),
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file"],
    },
}
