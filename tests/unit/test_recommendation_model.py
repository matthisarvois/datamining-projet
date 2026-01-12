# ===============================================
# 🧪 TESTS UNITAIRES - MODÈLE DE RECOMMANDATION
# Master 2 - SEP
# ===============================================

"""
Tests unitaires pour le modèle de recommandation.

Ces tests vérifient que le modèle ML fonctionne correctement :
- Initialisation et configuration
- Préparation des données d'entraînement
- Processus d'entraînement
- Génération de prédictions
- Métriques et performances

Usage:
    pytest tests/unit/test_recommendation_model.py -v
    pytest tests/unit/test_recommendation_model.py::TestOlistRecommendationModel::test_initialization -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import joblib
import tempfile
from pathlib import Path
import sys

# Import du module à tester
sys.path.append(str(Path(__file__).parent.parent.parent))
from ml_pipeline.models.recommendation_model import (
    OlistRecommendationModel,
    RecommendationPipeline
)
from config import MLConfig

@pytest.mark.unit
class TestOlistRecommendationModel:
    """
    Tests unitaires pour OlistRecommendationModel.

    Teste toutes les fonctionnalités du modèle de manière isolée.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du modèle avec paramètres par défaut.
        """
        # ARRANGE & ACT
        model = OlistRecommendationModel()

        # ASSERT
        assert model.rf_params == MLConfig.RANDOM_FOREST_PARAMS
        assert model.pipeline is not None
        assert model.feature_columns == []
        assert not model.is_trained
        assert model.feature_importance_ is None
        assert model.training_score_ is None

    def test_initialization_with_custom_params(self):
        """
        Test: L'initialisation du modèle avec paramètres personnalisés.
        """
        # ARRANGE
        custom_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 123
        }

        # ACT
        model = OlistRecommendationModel(custom_params)

        # ASSERT
        assert model.rf_params == custom_params
        # Vérifier que les paramètres sont passés au RandomForest
        rf_classifier = model.pipeline.named_steps['classifier']
        assert rf_classifier.n_estimators == 50
        assert rf_classifier.max_depth == 5
        assert rf_classifier.random_state == 123

    def test_prepare_training_data(self, sample_customer_features, sample_product_features):
        """
        Test: La préparation des données d'entraînement fonctionne correctement.
        """
        # ARRANGE
        model = OlistRecommendationModel()

        # Créer des interactions factices
        interactions = pd.DataFrame({
            'customer_id': ['test_customer_000', 'test_customer_001', 'test_customer_002'],
            'product_id': ['test_product_000', 'test_product_001', 'test_product_002'],
            'purchased': [1, 1, 1]
        })

        # ACT
        X, y = model.prepare_training_data(sample_customer_features, sample_product_features, interactions)

        # ASSERT
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) > len(interactions)  # Doit inclure des échantillons négatifs
        assert len(X) == len(y)
        assert set(y.unique()).issubset({0, 1})  # y doit être binaire

        # Vérifier que les features sont correctement jointes
        assert not X.empty
        assert all(col in X.columns for col in X.select_dtypes(include=[np.number]).columns)

        # Vérifier la proportion d'exemples positifs/négatifs
        positive_ratio = y.sum() / len(y)
        assert 0.1 <= positive_ratio <= 0.9  # Ratio raisonnable

    def test_create_negative_samples(self, sample_customer_features, sample_product_features):
        """
        Test: La création d'échantillons négatifs est correcte.
        """
        # ARRANGE
        model = OlistRecommendationModel()

        positive_interactions = pd.DataFrame({
            'customer_id': ['test_customer_000', 'test_customer_001'],
            'product_id': ['test_product_000', 'test_product_001'],
            'purchased': [1, 1]
        })

        customer_ids = sample_customer_features.index.tolist()
        product_ids = sample_product_features.index.tolist()

        # ACT
        negative_samples = model._create_negative_samples(
            positive_interactions, customer_ids, product_ids, negative_ratio=2.0
        )

        # ASSERT
        assert isinstance(negative_samples, pd.DataFrame)
        assert len(negative_samples) == len(positive_interactions) * 2  # Ratio de 2.0
        assert all(negative_samples['purchased'] == 0)

        # Vérifier qu'aucun échantillon négatif ne correspond à une interaction positive
        positive_pairs = set(zip(positive_interactions['customer_id'], positive_interactions['product_id']))
        negative_pairs = set(zip(negative_samples['customer_id'], negative_samples['product_id']))
        assert len(positive_pairs.intersection(negative_pairs)) == 0

    def test_predict_proba(self, mock_trained_model, sample_product_features):
        """
        Test: La prédiction de probabilité fonctionne correctement.
        """
        # ARRANGE
        customer_features = {
            'total_orders': 5,
            'total_spent': 250.0,
            'avg_order_value': 50.0,
            'days_since_last_order': 15
        }

        # Limiter les produits pour la simplicité du test
        limited_products = sample_product_features.iloc[:3]

        # ACT
        predictions = mock_trained_model.predict_proba(customer_features, limited_products)

        # ASSERT
        assert isinstance(predictions, pd.DataFrame)
        assert 'product_id' in predictions.columns
        assert 'purchase_probability' in predictions.columns
        assert len(predictions) == len(limited_products)

        # Vérifier que les probabilités sont valides
        probabilities = predictions['purchase_probability']
        assert all(0 <= prob <= 1 for prob in probabilities)

        # Vérifier que les résultats sont triés par probabilité décroissante
        assert predictions['purchase_probability'].is_monotonic_decreasing

    def test_calculate_confidence(self, mock_trained_model):
        """
        Test: Le calcul du niveau de confiance est correct.
        """
        # ARRANGE & ACT & ASSERT
        assert mock_trained_model._calculate_confidence(0.9) == 'High'
        assert mock_trained_model._calculate_confidence(0.75) == 'Medium'
        assert mock_trained_model._calculate_confidence(0.5) == 'Low'
        assert mock_trained_model._calculate_confidence(0.2) == 'Very Low'

        # Test des limites
        assert mock_trained_model._calculate_confidence(0.8) == 'High'
        assert mock_trained_model._calculate_confidence(0.6) == 'Medium'
        assert mock_trained_model._calculate_confidence(0.4) == 'Low'

    def test_get_model_performance(self, mock_trained_model):
        """
        Test: Récupération des métriques de performance.
        """
        # ACT
        performance = mock_trained_model.get_model_performance()

        # ASSERT
        assert isinstance(performance, dict)
        expected_metrics = ['train_accuracy', 'test_accuracy', 'auc_score', 'cv_mean', 'cv_std']
        for metric in expected_metrics:
            assert metric in performance
            assert isinstance(performance[metric], (int, float))

    def test_get_feature_importance(self, mock_trained_model):
        """
        Test: Récupération de l'importance des features.
        """
        # ACT
        importance = mock_trained_model.get_feature_importance(top_n=3)

        # ASSERT
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 3  # top_n = 3
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

        # Vérifier que c'est trié par importance décroissante
        assert importance['importance'].is_monotonic_decreasing

    def test_model_not_trained_errors(self):
        """
        Test: Les erreurs quand le modèle n'est pas entraîné.
        """
        # ARRANGE
        model = OlistRecommendationModel()
        customer_features = {'total_orders': 1}
        product_features = pd.DataFrame({'feature1': [1, 2]})

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Le modèle n'est pas encore entraîné"):
            model.predict_proba(customer_features, product_features)

        with pytest.raises(ValueError, match="Impossible de sauvegarder un modèle non entraîné"):
            model.save_model()

@pytest.mark.unit
class TestRecommendationPipeline:
    """
    Tests unitaires pour RecommendationPipeline.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du pipeline complet.
        """
        # ARRANGE & ACT
        pipeline = RecommendationPipeline()

        # ASSERT
        assert pipeline.model is not None
        assert isinstance(pipeline.model, OlistRecommendationModel)
        assert pipeline.feature_engine is None

# Tests d'intégration légers (mais toujours unitaires)

@pytest.mark.unit
def test_edge_cases_and_error_handling():
    """
    Test: Gestion des cas limites et erreurs.
    """
    model = OlistRecommendationModel()

    # Test avec données vides
    empty_df = pd.DataFrame()
    with pytest.raises((ValueError, KeyError)):
        model.prepare_training_data(empty_df, empty_df, empty_df)

    # Test avec customer_features manquantes
    customer_features = {}
    product_features = pd.DataFrame({'feature': [1]}, index=['prod1'])

    # Doit gérer gracieusement les features manquantes
    model.is_trained = True
    model.feature_columns = ['missing_feature']
    model.pipeline = Mock()
    model.pipeline.predict_proba.return_value = np.array([[0.3, 0.7]])

    predictions = model.predict_proba(customer_features, product_features)
    assert len(predictions) == 1