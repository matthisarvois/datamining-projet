# ===============================================
# 🧪 TESTS UNITAIRES - FEATURE ENGINEERING
# Master 2 - SEP
# ===============================================

"""
Tests unitaires pour le module de feature engineering.

Ces tests vérifient que chaque fonction et classe de feature engineering
fonctionne correctement de manière isolée.

Concepts testés:
- Transformation des données clients
- Calcul des métriques RFM
- Segmentation des clients
- Validation des outputs

Usage:
    pytest tests/unit/test_feature_engineering.py -v
    pytest tests/unit/test_feature_engineering.py::test_customer_feature_engineer_basic -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Import du module à tester
sys.path.append(str(Path(__file__).parent.parent.parent))
from ml_pipeline.preprocessing.feature_engineering import (
    CustomerFeatureEngineer,
    ProductFeatureEngineer,
    RecommendationFeatureEngine
)

@pytest.mark.unit
class TestCustomerFeatureEngineer:
    """
    Tests unitaires pour CustomerFeatureEngineer.

    Teste la transformation des données clients en features ML.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du CustomerFeatureEngineer fonctionne correctement.
        """
        # ARRANGE & ACT
        engineer = CustomerFeatureEngineer()

        # ASSERT
        assert engineer.reference_date is None
        assert engineer.category_encoder is not None
        assert hasattr(engineer, 'fit')
        assert hasattr(engineer, 'transform')

    def test_fit_basic(self, sample_customer_features):
        """
        Test: La méthode fit apprend correctement les paramètres.
        """
        # ARRANGE
        engineer = CustomerFeatureEngineer()

        # ACT
        result = engineer.fit(sample_customer_features)

        # ASSERT
        assert result == engineer  # fit doit retourner self
        assert engineer.reference_date is not None
        assert hasattr(engineer.category_encoder, 'classes_')

    def test_transform_single_customer(self):
        """
        Test: La transformation d'un client unique fonctionne correctement.
        """
        # ARRANGE
        engineer = CustomerFeatureEngineer()
        engineer.reference_date = datetime(2023, 12, 31)

        customer_data = {
            'total_orders': 5,
            'total_spent': 250.0,
            'last_order_date': datetime(2023, 11, 15),
            'favorite_category': 'cama_mesa_banho',
            'avg_review_score': 4.2,
            'unique_products_bought': 3
        }

        # Fit avec des données factices pour l'encodeur
        dummy_df = pd.DataFrame({
            'favorite_category': ['cama_mesa_banho', 'beleza_saude']
        })
        engineer.fit(dummy_df)

        # ACT
        features = engineer.transform(customer_data)

        # ASSERT
        assert isinstance(features, dict)
        assert 'total_orders' in features
        assert 'total_spent' in features
        assert 'avg_order_value' in features
        assert 'days_since_last_order' in features

        # Vérifications des calculs
        expected_avg_order = 250.0 / 5
        assert features['avg_order_value'] == expected_avg_order

        expected_days_since = (engineer.reference_date - customer_data['last_order_date']).days
        assert features['days_since_last_order'] == expected_days_since





@pytest.mark.unit
class TestProductFeatureEngineer:
    """
    Tests unitaires pour ProductFeatureEngineer.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du ProductFeatureEngineer fonctionne.
        """
        # ARRANGE & ACT
        engineer = ProductFeatureEngineer()

        # ASSERT
        assert engineer.scaler is not None
        assert engineer.category_encoder is not None

    def test_fit_and_transform(self, sample_products_data):
        """
        Test: Fit et transform fonctionnent ensemble correctement.
        """
        # ARRANGE
        engineer = ProductFeatureEngineer()

        # ACT
        engineer.fit(sample_products_data)
        result = engineer.transform(sample_products_data)

        # ASSERT
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_products_data)
        assert 'category_encoded' in result.columns

        # Vérifier que les dimensions ont été normalisées
        numeric_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        available_cols = [col for col in numeric_cols if col in result.columns]

        if available_cols:
            # Les colonnes normalisées ne doivent pas avoir exactement les mêmes valeurs qu'avant
            original_col = available_cols[0]
            original_values = sample_products_data[original_col].values
            normalized_values = result[original_col].values
            assert not np.array_equal(original_values, normalized_values)


@pytest.mark.unit
class TestRecommendationFeatureEngine:
    """
    Tests unitaires pour RecommendationFeatureEngine.
    """

    def test_initialization(self):
        """
        Test: L'initialisation du pipeline complet fonctionne.
        """
        # ARRANGE & ACT
        engine = RecommendationFeatureEngine()

        # ASSERT
        assert engine.customer_engineer is not None
        assert engine.product_engineer is not None
        assert not engine.is_fitted



# Tests d'aide et utiitaires
@pytest.mark.unit
def test_feature_validation_helpers():
    """
    Test: Les fonctions helper de validation des features.
    """
    from tests.conftest import assert_dataframe_equals

    # ARRANGE
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df3 = pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]})

    # ACT & ASSERT
    assert assert_dataframe_equals(df1, df2) is True
    assert assert_dataframe_equals(df1, df3) is False

# Tests de régression
@pytest.mark.unit
def test_feature_engineering_reproducibility(sample_customer_features):
    """
    Test: Les transformations sont reproductibles avec les mêmes données.
    """
    # ARRANGE
    engineer1 = CustomerFeatureEngineer()
    engineer2 = CustomerFeatureEngineer()

    # ACT
    result1 = engineer1.fit_transform(sample_customer_features)
    result2 = engineer2.fit_transform(sample_customer_features)

    # ASSERT
    pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.unit
def test_feature_engineering_with_single_row(sample_customer_features):
    """
    Test: Le feature engineering fonctionne avec une seule ligne de données.
    """
    # ARRANGE
    engineer = CustomerFeatureEngineer()
    single_row = sample_customer_features.iloc[:1].copy()

    # ACT
    result = engineer.fit_transform(single_row)

    # ASSERT
    assert len(result) == 1
    assert isinstance(result, pd.DataFrame)
    # Avec une seule ligne, les quartiles peuvent ne pas fonctionner normalement
    # mais le processus ne doit pas crash