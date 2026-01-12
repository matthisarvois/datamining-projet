# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - SCHEMAS
# Master 2 - SEP
# ===============================================

"""
Schémas Pydantic pour l'API de recommandation.

Ces schémas définissent la structure des données d'entrée et de sortie
de l'API, assurant la validation automatique et la documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class CustomerRequest(BaseModel):
    """Requête de recommandation pour un client."""

    customer_id: str = Field(
        ...,
        description="ID unique du client",
        example="customer_123"
    )

    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Nombre de recommandations souhaitées"
    )

    @validator('customer_id')
    def validate_customer_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('customer_id ne peut pas être vide')
        return v.strip()

class ProductInfo(BaseModel):
    """Informations détaillées d'un produit."""

    product_id: str = Field(..., description="ID unique du produit")
    category: Optional[str] = Field(None, description="Catégorie du produit")
    weight_g: Optional[float] = Field(None, description="Poids en grammes")
    length_cm: Optional[float] = Field(None, description="Longueur en cm")
    width_cm: Optional[float] = Field(None, description="Largeur en cm")
    height_cm: Optional[float] = Field(None, description="Hauteur en cm")

class Recommendation(BaseModel):
    """Une recommandation individuelle."""

    customer_id: str = Field(..., description="ID du client")
    product_id: str = Field(..., description="ID du produit recommandé")
    purchase_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilité d'achat (0-1)"
    )
    confidence: str = Field(
        ...,
        description="Niveau de confiance",
        example="High"
    )
    product_info: Optional[ProductInfo] = Field(
        None,
        description="Informations détaillées du produit"
    )
    rank: int = Field(..., description="Rang de la recommandation")

class RecommendationResponse(BaseModel):
    """Réponse complète de recommandation."""

    customer_id: str = Field(..., description="ID du client")
    recommendations: List[Recommendation] = Field(
        ...,
        description="Liste des recommandations"
    )
    total_recommendations: int = Field(
        ...,
        description="Nombre total de recommandations"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de génération"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Version du modèle utilisé"
    )

class ModelMetrics(BaseModel):
    """Métriques de performance du modèle."""

    model_name: str = Field(default="RandomForest", description="Nom du modèle")
    train_accuracy: float = Field(..., description="Précision d'entraînement")
    test_accuracy: float = Field(..., description="Précision de test")
    auc_score: float = Field(..., description="Score AUC-ROC")
    cv_mean: float = Field(..., description="Moyenne cross-validation")
    cv_std: float = Field(..., description="Écart-type cross-validation")
    last_trained: Optional[datetime] = Field(
        None,
        description="Date du dernier entraînement"
    )

class FeatureImportance(BaseModel):
    """Importance d'une feature."""

    feature: str = Field(..., description="Nom de la feature")
    importance: float = Field(
        ...,
        ge=0.0,
        description="Score d'importance"
    )

class ModelInfoResponse(BaseModel):
    """Informations détaillées sur le modèle."""

    metrics: ModelMetrics = Field(..., description="Métriques de performance")
    feature_importance: List[FeatureImportance] = Field(
        ...,
        description="Importance des features"
    )
    model_status: str = Field(
        default="ready",
        description="Statut du modèle"
    )

class HealthResponse(BaseModel):
    """Réponse de vérification de santé."""

    status: str = Field(default="healthy", description="Statut de l'API")
    model_loaded: bool = Field(..., description="Modèle chargé avec succès")
    version: str = Field(default="1.0.0", description="Version de l'API")
    uptime_seconds: float = Field(..., description="Temps de fonctionnement en secondes")

class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""

    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur détaillé")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Détails supplémentaires"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de l'erreur"
    )

class BatchRecommendationRequest(BaseModel):
    """Requête de recommandation pour plusieurs clients."""

    customer_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Liste des IDs clients"
    )
    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Nombre de recommandations par client"
    )

    @validator('customer_ids')
    def validate_customer_ids(cls, v):
        if not v:
            raise ValueError('La liste des customer_ids ne peut pas être vide')
        # Éliminer les doublons et valeurs vides
        cleaned_ids = [id.strip() for id in set(v) if id and id.strip()]
        if not cleaned_ids:
            raise ValueError('Aucun customer_id valide fourni')
        return cleaned_ids

class BatchRecommendationResponse(BaseModel):
    """Réponse de recommandation pour plusieurs clients."""

    results: List[RecommendationResponse] = Field(
        ...,
        description="Recommandations pour chaque client"
    )
    total_customers: int = Field(..., description="Nombre total de clients traités")
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de génération"
    )
    processing_time_seconds: float = Field(
        ...,
        description="Temps de traitement en secondes"
    )