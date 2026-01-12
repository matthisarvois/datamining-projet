# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - SERVICE
# Master 2 - SEP
# ===============================================

"""
Service de recommandation pour l'API FastAPI.

Ce module fait l'interface entre l'API REST et le modèle de machine learning,
gérant le chargement du modèle, la cache, et les prédictions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

from ml_pipeline.models.recommendation_model import OlistRecommendationModel
from ml_pipeline.preprocessing.feature_engineering import CustomerFeatureEngineer

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config import MLConfig
from backend.app.schemas.recommendation import (
    Recommendation, RecommendationResponse, ModelMetrics,
    FeatureImportance, ModelInfoResponse
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Service de recommandation centralisé.

    Gère:
    - Le chargement et la cache du modèle ML
    - La génération de recommandations
    - La gestion des features clients
    - Les métriques de performance
    """

    def __init__(self):
        self.model: Optional[OlistRecommendationModel] = None
        self.customer_features: Optional[pd.DataFrame] = None
        self.product_features: Optional[pd.DataFrame] = None
        self.feature_engineer: Optional[CustomerFeatureEngineer] = None
        self.model_loaded_at: Optional[datetime] = None
        self._cache: Dict = {}
        self._cache_ttl = timedelta(hours=1)

    async def initialize(self) -> bool:
        """
        Initialise le service en chargeant le modèle et les données.

        Returns:
            True si l'initialisation réussit, False sinon
        """
        try:
            logger.info("🔄 Initialisation du service de recommandation...")

            # Charger le modèle pré-entraîné
            await self._load_model()

            # Charger les features clients
            await self._load_customer_features()

            # Créer des features produits simulées
            await self._load_product_features()

            self.model_loaded_at = datetime.now()
            logger.info("Service de recommandation initialisé avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            return False

    async def _load_model(self):
        """Charge le modèle pré-entraîné."""

        self.model = OlistRecommendationModel.load_model()
        self.feature_engineer = CustomerFeatureEngineer()
        logger.info("Modèle chargé avec succès")


    async def _load_customer_features(self):
        """Charge les features clients pré-calculées."""
        customer_features_path = MLConfig.CUSTOMER_FEATURES_FILE
        self.customer_features = pd.read_csv(customer_features_path, index_col=0)
        logger.info(f"Features clients chargées: {len(self.customer_features)} clients")

    async def _load_product_features(self):
        """Charge les features produits réelles."""
        product_features_path = MLConfig.PRODUCT_FEATURES_FILE

        df = pd.read_csv(product_features_path)

        # Encodage simple de la catégorie
        df['category_encoded'] = df['product_category_name'].astype('category').cat.codes

        # Création d'une feature volume
        df['volume_cm3'] = (
                df['product_length_cm'] *
                df['product_height_cm'] *
                df['product_width_cm']
        )

        self.product_features = df[
            [
                'category_encoded',
                'product_name_lenght',
                'product_description_lenght',
                'product_photos_qty',
                'product_weight_g',
                'volume_cm3'
            ]
        ].set_index(df['product_id'])

        logger.info(f"Features produits chargées: {len(self.product_features)} produits")

    def _get_cache_key(self, customer_id: str, n_recommendations: int) -> str:
        """Génère une clé de cache."""
        return f"{customer_id}_{n_recommendations}"

    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Vérifie si le cache est encore valide."""
        return datetime.now() - cached_at < self._cache_ttl

    async def get_recommendations(self, customer_id: str, n_recommendations: int = 10) -> RecommendationResponse:
        """
        Génère des recommandations personnalisées pour un client.

        Args:
            customer_id: ID du client
            n_recommendations: Nombre de recommandations

        Returns:
            Réponse avec les recommandations
        """
        # Vérifier le cache
        cache_key = self._get_cache_key(customer_id, n_recommendations)
        if cache_key in self._cache:
            cached_result, cached_at = self._cache[cache_key]
            if self._is_cache_valid(cached_at):
                logger.info(f"📋 Recommandations servies depuis le cache pour {customer_id}")
                return cached_result

        # Obtenir les features du client
        customer_features = await self._get_customer_features(customer_id)

        # Générer les recommandations

        recommendations = await self._generate_ml_recommendations(
            customer_id, customer_features, n_recommendations
        )

        # Créer la réponse
        response = RecommendationResponse(
            customer_id=customer_id,
            recommendations=recommendations,
            total_recommendations=len(recommendations)
        )

        # Mettre en cache
        self._cache[cache_key] = (response, datetime.now())

        logger.info(f"{len(recommendations)} recommandations générées pour {customer_id}")
        return response

    async def _get_customer_features(self, customer_id: str) -> Dict:
        """Récupère les features d'un client."""
        if customer_id in self.customer_features.index:
            return self.customer_features.loc[customer_id].to_dict()
        else:
            logger.warning(f"⚠️ Client {customer_id} non trouvé, utilisation de features par défaut")
            return {
                'total_orders': None,
                'total_spent': None,
                'avg_order_value': None,
                'days_since_last_order': None,
                'avg_review_score': None,
                'unique_products_bought': None
            }

    async def _generate_ml_recommendations(self, customer_id: str, customer_features: Dict, n_recommendations: int) -> \
    List[Recommendation]:
        """Génère des recommandations avec le modèle ML."""
        # Utiliser le modèle pour obtenir les prédictions
        predictions = self.model.predict_proba(customer_features, self.product_features)

        # Limiter au nombre demandé
        top_predictions = predictions.head(n_recommendations)

        # Convertir en objets Recommendation
        recommendations = []
        for rank, (_, row) in enumerate(top_predictions.iterrows(), 1):
            product_id = row['product_id']
            probability = row['purchase_probability']

            recommendation = Recommendation(
                customer_id=customer_id,
                product_id=product_id,
                purchase_probability=float(probability),
                confidence=self._calculate_confidence(probability),
                rank=rank
            )
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

    async def get_model_info(self) -> ModelInfoResponse:
        """
        Retourne les informations sur le modèle.

        Returns:
            Informations détaillées sur le modèle
        """
        if not self.model or not self.model.is_trained:
            # Modèle non chargé, retourner des infos par défaut
            metrics = ModelMetrics(
                train_accuracy=0.0,
                test_accuracy=0.0,
                auc_score=0.0,
                cv_mean=0.0,
                cv_std=0.0
            )
            return ModelInfoResponse(
                metrics=metrics,
                feature_importance=[],
                model_status="not_loaded"
            )

        # Métriques du modèle
        perf = self.model.get_model_performance()
        metrics = ModelMetrics(
            train_accuracy=perf['train_accuracy'],
            test_accuracy=perf['test_accuracy'],
            auc_score=perf['auc_score'],
            cv_mean=perf['cv_mean'],
            cv_std=perf['cv_std'],
            last_trained=self.model_loaded_at
        )

        # Importance des features
        feature_importance = []
        if hasattr(self.model, 'get_feature_importance'):
            importance_df = self.model.get_feature_importance()
            for _, row in importance_df.iterrows():
                feature_importance.append(FeatureImportance(
                    feature=row['feature'],
                    importance=row['importance']
                ))

        return ModelInfoResponse(
            metrics=metrics,
            feature_importance=feature_importance,
            model_status="ready"
        )

    async def get_customer_list(self) -> List[str]:
        """Retourne la liste des clients disponibles."""
        if self.customer_features is not None:
            return self.customer_features.index.tolist()
        else:
            return [f"customer_{i:03d}" for i in range(10)]

    def is_healthy(self) -> bool:
        """Vérifie si le service est en bonne santé."""
        return (
                self.model is not None and
                self.customer_features is not None and
                self.product_features is not None
        )


# Instance singleton du service
recommendation_service = RecommendationService()
