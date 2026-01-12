# ===============================================
# 🧪 OLIST RECOMMENDATION SYSTEM - TEST CONFIG
# Master 2 - SEP
# ===============================================

"""
Configuration commune pour tous les tests avec pytest.

Ce fichier définit des fixtures réutilisables à travers tous les tests :
- Données de test
- Configuration d'environnement
- Instances de modèles
- Client API de test

Usage:
    pytest                    # Tous les tests
    pytest tests/unit/        # Tests unitaires uniquement
    pytest tests/integration/ # Tests d'intégration uniquement
    pytest tests/e2e/         # Tests end-to-end uniquement
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import MagicMock

# Ajouter le répertoire racine au PYTHONPATH pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from config import MLConfig, DataConfig
from ml_pipeline.models.recommendation_model import OlistRecommendationModel
from ml_pipeline.preprocessing.feature_engineering import CustomerFeatureEngineer, ProductFeatureEngineer

@pytest.fixture(scope="session")
def test_data_dir():
    """
    Crée un répertoire temporaire pour les données de test.
    Nettoyé automatiquement à la fin des tests.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="olist_test_"))

    # Créer les sous-répertoires nécessaires
    (temp_dir / "raw").mkdir(exist_ok=True)
    (temp_dir / "processed").mkdir(exist_ok=True)
    (temp_dir / "models").mkdir(exist_ok=True)

    yield temp_dir

    # Cleanup après tous les tests
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_customers_data():
    """
    Fixture qui fournit des données clients de test réalistes.

    Returns:
        pd.DataFrame: Données clients avec toutes les colonnes nécessaires
    """
    np.random.seed(42)  # Pour la reproductibilité des tests

    data = {
        'customer_id': [f'test_customer_{i:03d}' for i in range(10)],
        'customer_unique_id': [f'unique_{i:03d}' for i in range(10)],
        'customer_zip_code_prefix': np.random.randint(10000, 99999, 10),
        'customer_city': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte'] * 3 + ['Curitiba'],
        'customer_state': ['SP', 'RJ', 'MG'] * 3 + ['PR']
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_orders_data():
    """
    Fixture qui fournit des données de commandes de test.
    """
    np.random.seed(42)

    base_date = datetime(2023, 1, 1)

    data = {
        'order_id': [f'test_order_{i:03d}' for i in range(15)],
        'customer_id': [f'test_customer_{i%10:03d}' for i in range(15)],  # Répartir sur 10 clients
        'order_status': np.random.choice(['delivered', 'shipped', 'processing'], 15, p=[0.8, 0.15, 0.05]),
        'order_purchase_timestamp': [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(15)],
        'order_approved_at': [base_date + timedelta(days=np.random.randint(0, 365), hours=24) for _ in range(15)],
        'order_delivered_customer_date': [base_date + timedelta(days=np.random.randint(7, 30)) for _ in range(15)]
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_products_data():
    """
    Fixture qui fournit des données produits de test.
    """
    np.random.seed(42)

    categories = ['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'informatica_acessorios', 'moveis_decoracao']

    data = {
        'product_id': [f'test_product_{i:03d}' for i in range(8)],
        'product_category_name': np.random.choice(categories, 8),
        'product_weight_g': np.random.randint(100, 5000, 8),
        'product_length_cm': np.random.randint(10, 50, 8),
        'product_height_cm': np.random.randint(5, 30, 8),
        'product_width_cm': np.random.randint(10, 40, 8)
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_order_items_data():
    """
    Fixture qui fournit des données d'items de commande.
    """
    np.random.seed(42)

    data = {
        'order_id': [f'test_order_{i%15:03d}' for i in range(20)],  # Répartir sur 15 commandes
        'order_item_id': list(range(1, 21)),
        'product_id': [f'test_product_{i%8:03d}' for i in range(20)],  # Répartir sur 8 produits
        'seller_id': [f'seller_{i%5:03d}' for i in range(20)],
        'price': np.random.uniform(20, 500, 20),
        'freight_value': np.random.uniform(5, 50, 20)
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_reviews_data():
    """
    Fixture qui fournit des données d'avis clients.
    """
    np.random.seed(42)

    data = {
        'review_id': [f'review_{i:03d}' for i in range(12)],
        'order_id': [f'test_order_{i:03d}' for i in range(12)],
        'review_score': np.random.choice([1, 2, 3, 4, 5], 12, p=[0.05, 0.05, 0.1, 0.3, 0.5]),
        'review_comment_title': ['Bon produit', 'Très satisfait', 'Moyen'] * 4,
        'review_creation_date': [datetime(2023, 1, 1) + timedelta(days=i*10) for i in range(12)]
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_customer_features():
    """
    Fixture qui fournit des features clients pré-calculées pour les tests.
    """
    np.random.seed(42)

    customer_ids = [f'test_customer_{i:03d}' for i in range(10)]

    data = {
        'total_orders': np.random.poisson(3, 10) + 1,
        'total_spent': np.random.exponential(200, 10) + 50,
        'days_since_last_order': np.random.exponential(30, 10) + 1,
        'avg_review_score': np.random.normal(4.0, 0.8, 10).clip(1, 5),
        'unique_products_bought': np.random.poisson(2, 10) + 1,
        'favorite_category': np.random.choice(['cama_mesa_banho', 'beleza_saude', 'esporte_lazer'], 10)
    }

    df = pd.DataFrame(data, index=customer_ids)
    df['avg_order_value'] = df['total_spent'] / df['total_orders']

    return df

@pytest.fixture
def sample_product_features():
    """
    Fixture qui fournit des features produits pour les tests.
    """
    np.random.seed(42)

    product_ids = [f'test_product_{i:03d}' for i in range(8)]

    data = {
        'category_encoded': np.random.randint(0, 5, 8),
        'weight': np.random.exponential(500, 8) + 100,
        'popularity_score': np.random.beta(2, 5, 8)
    }

    return pd.DataFrame(data, index=product_ids)

@pytest.fixture
def mock_trained_model():
    """
    Fixture qui fournit un modèle ML mocké pour les tests.
    Évite d'avoir à entraîner un vrai modèle à chaque test.
    """
    model = OlistRecommendationModel()
    model.is_trained = True
    model.feature_columns = ['total_orders', 'total_spent', 'avg_order_value', 'days_since_last_order']

    # Mock du pipeline sklearn
    model.pipeline = MagicMock()
    model.pipeline.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])

    # Mock des métriques de performance
    model.training_score_ = {
        'train_accuracy': 0.85,
        'test_accuracy': 0.78,
        'auc_score': 0.82,
        'cv_mean': 0.80,
        'cv_std': 0.03
    }

    # Mock de l'importance des features
    model.feature_importance_ = pd.DataFrame({
        'feature': ['total_spent', 'total_orders', 'days_since_last_order', 'avg_order_value'],
        'importance': [0.4, 0.3, 0.2, 0.1]
    })

    return model

@pytest.fixture
def api_client():
    """
    Fixture qui fournit un client FastAPI de test.
    Permet de tester les endpoints API sans démarrer un serveur.
    """
    from fastapi.testclient import TestClient
    from backend.app.main import app

    return TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, test_data_dir):
    """
    Fixture automatique qui configure l'environnement de test.

    - Redirige les chemins de fichiers vers le répertoire de test
    - Configure les variables d'environnement appropriées
    - S'assure que les tests sont isolés
    """
    # Rediriger les chemins de configuration vers le répertoire de test
    monkeypatch.setattr("config.RAW_DATA_DIR", test_data_dir / "raw")
    monkeypatch.setattr("config.PROCESSED_DATA_DIR", test_data_dir / "processed")
    monkeypatch.setattr("config.MODELS_DIR", test_data_dir / "models")

    # Configuration pour les tests
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")  # Réduire le bruit dans les logs de test

# Markers personnalisés pour organiser les tests
pytest_plugins = []

def pytest_configure(config):
    """Configuration personnalisée de pytest."""
    config.addinivalue_line(
        "markers", "unit: Tests unitaires - testent des composants individuels"
    )
    config.addinivalue_line(
        "markers", "integration: Tests d'intégration - testent l'interaction entre composants"
    )
    config.addinivalue_line(
        "markers", "e2e: Tests end-to-end - testent des workflows complets"
    )
    config.addinivalue_line(
        "markers", "slow: Tests lents - peuvent être exclus pour les tests rapides"
    )

# Helper functions pour les tests
def assert_dataframe_equals(df1, df2, check_dtype=True):
    """
    Helper function pour comparer des DataFrames dans les tests.
    Plus lisible que pd.testing.assert_frame_equal dans les assertions.
    """
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
        return True
    except AssertionError:
        return False

def assert_recommendations_valid(recommendations):
    """
    Helper function pour valider la structure des recommandations.
    """
    assert isinstance(recommendations, list), "Les recommandations doivent être une liste"

    for i, rec in enumerate(recommendations):
        assert isinstance(rec, dict), f"Recommandation {i} doit être un dictionnaire"

        required_fields = ['customer_id', 'product_id', 'purchase_probability', 'confidence', 'rank']
        for field in required_fields:
            assert field in rec, f"Champ '{field}' manquant dans la recommandation {i}"

        assert 0 <= rec['purchase_probability'] <= 1, f"Probabilité invalide: {rec['purchase_probability']}"
        assert rec['confidence'] in ['High', 'Medium', 'Low', 'Very Low'], f"Confiance invalide: {rec['confidence']}"
        assert rec['rank'] > 0, f"Rang invalide: {rec['rank']}"