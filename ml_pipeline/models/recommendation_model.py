# ===============================================
# OLIST RECOMMENDATION SYSTEM - MODEL
# Master 2 - SEP
# ===============================================

"""
Modèle de recommandation basé sur RandomForest.

Ce module implémente un système de recommandation hybride qui combine:
- Collaborative Filtering (similarités entre clients)
- Content-Based Filtering (caractéristiques des produits)
- Features comportementales (RFM, historique)

Classes:
    - OlistRecommendationModel: Modèle principal de recommandation
    - RecommendationPipeline: Pipeline complet d'entraînement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Dict, Optional
import joblib
import sys
from pathlib import Path

from ml_pipeline.preprocessing.feature_engineering import load_and_prepare_data, RecommendationFeatureEngine

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MLConfig, MODELS_DIR

class OlistRecommendationModel:
    """
    Modèle de recommandation Olist basé sur RandomForest.

    Le modèle prédit la probabilité qu'un client achète un produit donné
    en se basant sur:
    - Les features du client (RFM, préférences, historique)
    - Les features du produit (catégorie, prix, caractéristiques)
    - Les interactions passées
    """

    def __init__(self, random_forest_params: Dict = None):
        """
        Initialise le modèle de recommandation.

        Args:
            random_forest_params: Paramètres pour RandomForest
        """
        self.rf_params = random_forest_params or MLConfig.RANDOM_FOREST_PARAMS

        # Pipeline de preprocessing + modèle
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(**self.rf_params))
        ])

        self.feature_columns = []
        self.is_trained = False
        self.feature_importance_ = None
        self.training_score_ = None

    def prepare_training_data(self, customer_features: pd.DataFrame,
                            product_features: pd.DataFrame,
                            interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les données d'entraînement en combinant les features.

        Args:
            customer_features: Features des clients
            product_features: Features des produits
            interactions: Matrice d'interaction client-produit

        Returns:
            X: Features d'entraînement
            y: Variable cible (achat ou non)
        """
        print("Préparation des données d'entraînement...")

        # Créer des échantillons négatifs (produits non achetés)
        positive_interactions = interactions[['customer_id', 'product_id', 'purchased']].copy()
        negative_interactions = self._create_negative_samples(
            positive_interactions, customer_features.index, product_features.index
        )

        # Combiner interactions positives et négatives
        all_interactions = pd.concat([positive_interactions, negative_interactions], ignore_index=True)

        # Joindre les features clients et produits
        training_data = all_interactions.merge(
            customer_features, left_on='customer_id', right_index=True, how='left'
        ).merge(
            product_features, left_on='product_id', right_index=True, how='left'
        )

        # Supprimer les colonnes non numériques
        feature_cols = training_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols.remove('purchased')  # Retirer la variable cible

        X = training_data[feature_cols].fillna(0)
        y = training_data['purchased']

        self.feature_columns = feature_cols

        print(f"   {len(X)} échantillons préparés ({y.sum()} positifs, {len(y) - y.sum()} négatifs)")
        print(f"   {len(feature_cols)} features utilisées")

        return X, y

    def _create_negative_samples(self, positive_interactions: pd.DataFrame,
                               customer_ids: List, product_ids: List,
                               negative_ratio: float = 2.0) -> pd.DataFrame:
        """
        Crée des échantillons négatifs (client-produit non achetés).

        Args:
            positive_interactions: Interactions positives existantes
            customer_ids: Liste des IDs clients
            product_ids: Liste des IDs produits
            negative_ratio: Ratio négatif/positif

        Returns:
            DataFrame des interactions négatives
        """
        # Ensemble des paires client-produit positives
        positive_pairs = set(
            zip(positive_interactions['customer_id'], positive_interactions['product_id'])
        )

        # Générer des paires aléatoires
        n_negative = int(len(positive_interactions) * negative_ratio)
        negative_pairs = []

        # Échantillonnage stratifié pour assurer la diversité
        customers_sample = np.random.choice(customer_ids, size=n_negative, replace=True)
        products_sample = np.random.choice(product_ids, size=n_negative, replace=True)

        for customer_id, product_id in zip(customers_sample, products_sample):
            if (customer_id, product_id) not in positive_pairs:
                negative_pairs.append({
                    'customer_id': customer_id,
                    'product_id': product_id,
                    'purchased': 0
                })

        return pd.DataFrame(negative_pairs[:n_negative])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'OlistRecommendationModel':
        """
        Entraîne le modèle de recommandation.

        Args:
            X: Features d'entraînement
            y: Variable cible

        Returns:
            self
        """
        print("Entraînement du modèle de recommandation...")

        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MLConfig.TEST_SIZE, random_state=MLConfig.RANDOM_STATE,
            stratify=y
        )

        # Entraînement
        self.pipeline.fit(X_train, y_train)

        # Évaluation
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)

        # Prédictions pour métriques avancées
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train,
            cv=MLConfig.CV_FOLDS, scoring='roc_auc'
        )

        self.training_score_ = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        # Importance des features
        rf_model = self.pipeline.named_steps['classifier']
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.is_trained = True

        print(f"   Modèle entraîné - Test AUC: {auc_score:.3f}")
        print(f"   Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        return self

    def predict_proba(self, customer_features: Dict, product_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit la probabilité d'achat pour un client et plusieurs produits.

        Args:
            customer_features: Features du client
            product_features: Features des produits

        Returns:
            DataFrame avec product_id et probabilité d'achat
        """
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas encore entraîné. Utilisez fit() d'abord.")

        # Préparer les features pour la prédiction
        customer_df = pd.DataFrame([customer_features] * len(product_features))
        customer_df.index = product_features.index

        # Combiner client et produits
        prediction_data = pd.concat([customer_df, product_features], axis=1)

        # S'assurer que toutes les colonnes d'entraînement sont présentes
        for col in self.feature_columns:
            if col not in prediction_data.columns:
                prediction_data[col] = 0

        X_pred = prediction_data[self.feature_columns].fillna(0)

        # Prédiction
        probas = self.pipeline.predict_proba(X_pred)[:, 1]

        # Résultats
        results = pd.DataFrame({
            'product_id': product_features.index,
            'purchase_probability': probas
        }).sort_values('purchase_probability', ascending=False)

        return results

    def get_recommendations(self, customer_id: str, customer_features: Dict,
                          product_features: pd.DataFrame, n_recommendations: int = 10) -> List[Dict]:
        """
        Obtient les recommandations personnalisées pour un client.

        Args:
            customer_id: ID du client
            customer_features: Features du client
            product_features: Features de tous les produits
            n_recommendations: Nombre de recommandations

        Returns:
            Liste des recommandations avec probabilité et features
        """
        # Prédire les probabilités
        predictions = self.predict_proba(customer_features, product_features)

        # Top N recommandations
        top_recommendations = predictions.head(n_recommendations)

        # Enrichir avec les détails produits
        recommendations = []
        for _, row in top_recommendations.iterrows():
            product_id = row['product_id']
            probability = row['purchase_probability']

            # Récupérer les détails du produit
            product_info = product_features.loc[product_id].to_dict()

            recommendation = {
                'customer_id': customer_id,
                'product_id': product_id,
                'purchase_probability': float(probability),
                'confidence': self._calculate_confidence(probability),
                'product_info': product_info
            }
            recommendations.append(recommendation)

        return recommendations

    def _calculate_confidence(self, probability: float) -> str:
        """Calcule le niveau de confiance basé sur la probabilité."""
        if probability >= 0.8:
            return 'High'
        elif probability >= 0.6:
            return 'Medium'
        elif probability >= 0.4:
            return 'Low'
        else:
            return 'Very Low'

    def get_model_performance(self) -> Dict:
        """Retourne les métriques de performance du modèle."""
        if not self.is_trained:
            return {"error": "Modèle non entraîné"}

        return self.training_score_

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Retourne l'importance des features."""
        if self.feature_importance_ is None:
            return pd.DataFrame()

        return self.feature_importance_.head(top_n)

    def save_model(self, filepath: Optional[Path] = None):
        """Sauvegarde le modèle entraîné."""
        if not self.is_trained:
            raise ValueError("Impossible de sauvegarder un modèle non entraîné")

        filepath = filepath or MLConfig.RECOMMENDATION_MODEL_FILE
        joblib.dump(self, filepath)
        print(f"   Modèle sauvegardé: {filepath}")

    @classmethod
    def load_model(cls, filepath: Optional[Path] = None) -> 'OlistRecommendationModel':
        """Charge un modèle pré-entraîné."""
        filepath = filepath or MLConfig.RECOMMENDATION_MODEL_FILE

        if not filepath.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {filepath}")

        model = joblib.load(filepath)
        print(f"   Modèle chargé: {filepath}")
        return model

class RecommendationPipeline:
    """
    Pipeline complet d'entraînement du système de recommandation.
    """

    def __init__(self):
        self.model = OlistRecommendationModel()
        self.feature_engine = None

    def train_pipeline(self, raw_data_dir: Path) -> Dict:
        """
        Entraîne le pipeline complet.

        Args:
            raw_data_dir: Répertoire contenant les données brutes

        Returns:
            Métriques d'entraînement
        """

        print("" + "="*50)
        print("ENTRAÎNEMENT PIPELINE DE RECOMMANDATION")
        print("" + "="*50)

        # 1. Charger les données
        customers, orders, order_items, products, reviews = load_and_prepare_data(raw_data_dir)

        # 2. Feature engineering
        self.feature_engine = RecommendationFeatureEngine()
        self.feature_engine.fit(customers, products)

        # Créer les features clients
        customer_features = self.feature_engine.create_customer_features(
            orders, reviews, order_items, products
        )

        # Features produits (simplifiées pour la démo)
        product_features = products[['product_id']].set_index('product_id')
        product_features['category_encoded'] = pd.Categorical(
            products['product_category_name']
        ).codes
        product_features['weight'] = products.get('product_weight_g', 100)

        # Créer les interactions
        interactions = self.feature_engine.create_interaction_matrix(orders, order_items)

        # 3. Entraîner le modèle
        X, y = self.model.prepare_training_data(customer_features, product_features, interactions)
        self.model.fit(X, y)

        # 4. Sauvegarder
        self.model.save_model()

        # Sauvegarder les features clients pour l'API
        customer_features_file = MLConfig.CUSTOMER_FEATURES_FILE
        customer_features.to_csv(customer_features_file)
        print(f"Features clients sauvegardées: {customer_features_file}")

        print("\nPipeline entraîné avec succès!")

        return self.model.get_model_performance()