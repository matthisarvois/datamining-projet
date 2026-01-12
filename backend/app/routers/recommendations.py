# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - ROUTERS
# Master 2 - SEP
# ===============================================

"""
Routes FastAPI pour l'API de recommandation.

Ce module définit tous les endpoints REST pour:
- Génération de recommandations
- Informations sur le modèle
- Métriques de performance
- Santé de l'API
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List
import time
import logging
from datetime import datetime

from ..schemas.recommendation import (
    CustomerRequest, RecommendationResponse, ModelInfoResponse,
    HealthResponse, BatchRecommendationRequest,
    BatchRecommendationResponse
)
from ..services.recommendation_service import recommendation_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Variable pour tracking du uptime
start_time = time.time()

@router.get("/health", response_model=HealthResponse, summary="🏥 Vérification de santé")
async def health_check():
    """
    Vérifie l'état de santé de l'API.

    Retourne:
    - Statut de l'API
    - État du modèle ML
    - Version et uptime
    """
    uptime = time.time() - start_time
    is_healthy = recommendation_service.is_healthy()

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=recommendation_service.model is not None,
        uptime_seconds=uptime
    )

@router.post("/recommendations", response_model=RecommendationResponse, summary="🎯 Recommandations personnalisées")
async def get_recommendations(request: CustomerRequest):
    """
    Génère des recommandations personnalisées pour un client.

    **Paramètres:**
    - **customer_id**: ID unique du client
    - **n_recommendations**: Nombre de recommandations (1-50, défaut: 10)

    **Réponse:**
    - Liste de recommandations triées par probabilité d'achat
    - Niveau de confiance pour chaque recommandation
    - Timestamp de génération

    **Exemple d'utilisation:**
    ```bash
    curl -X POST "http://localhost:8000/recommendations" \\
         -H "Content-Type: application/json" \\
         -d '{"customer_id": "customer_001", "n_recommendations": 5}'
    ```
    """
    try:
        logger.info(f"🎯 Génération de recommandations pour {request.customer_id}")

        response = await recommendation_service.get_recommendations(
            customer_id=request.customer_id,
            n_recommendations=request.n_recommendations
        )

        logger.info(f"✅ {len(response.recommendations)} recommandations générées")
        return response

    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération de recommandations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de recommandations: {str(e)}"
        )

@router.get("/model/info", response_model=ModelInfoResponse, summary="🤖 Informations sur le modèle")
async def get_model_info():
    """
    Retourne les informations détaillées sur le modèle ML.

    **Informations incluses:**
    - Métriques de performance (précision, AUC, etc.)
    - Importance des features
    - Date du dernier entraînement
    - Statut du modèle

    **Usage pédagogique:**
    - Analyse de la performance du modèle
    - Compréhension des features importantes
    - Debugging et amélioration
    """
    try:
        logger.info("📊 Récupération des informations du modèle")
        model_info = await recommendation_service.get_model_info()
        return model_info

    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des infos du modèle: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des informations: {str(e)}"
        )

@router.get("/customers", response_model=List[str], summary="👥 Liste des clients")
async def get_customers(
    limit: int = Query(default=1000, ge=1, le=1000, description="Nombre maximum de clients à retourner")
):
    """
    Retourne la liste des clients disponibles pour les recommandations.

    **Paramètres:**
    - **limit**: Nombre maximum de clients (1-500, défaut: 50)

    **Usage:**
    - Interface utilisateur pour sélection de clients
    - Tests et validation des recommandations
    - Exploration des données
    """
    try:
        logger.info(f"👥 Récupération de la liste des clients (limit: {limit})")
        customers = await recommendation_service.get_customer_list()
        return customers[:limit]

    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des clients: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des clients: {str(e)}"
        )

@router.get("/recommendations/{customer_id}", response_model=RecommendationResponse, summary="🎯 Recommandations par URL")
async def get_recommendations_by_path(
    customer_id: str,
    n_recommendations: int = Query(default=10, ge=1, le=50, description="Nombre de recommandations")
):
    """
    Génère des recommandations via paramètres d'URL (alternative GET).

    **Paramètres d'URL:**
    - **customer_id**: ID du client dans l'URL
    - **n_recommendations**: Nombre de recommandations (query parameter)

    **Exemple:**
    ```
    GET /recommendations/customer_001?n_recommendations=5
    ```

    **Usage:**
    - Tests rapides via navigateur
    - Intégration simple dans des liens
    - Debugging et validation
    """
    try:
        logger.info(f"🎯 Recommandations GET pour {customer_id}")

        request = CustomerRequest(
            customer_id=customer_id,
            n_recommendations=n_recommendations
        )

        return await recommendation_service.get_recommendations(
            customer_id=request.customer_id,
            n_recommendations=request.n_recommendations
        )

    except Exception as e:
        logger.error(f"❌ Erreur GET recommandations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de recommandations: {str(e)}"
        )

# Routes de debugging et monitoring

@router.get("/debug/cache/stats", summary="🔍 Statistiques du cache")
async def get_cache_stats():
    """
    **⚠️ Route de debugging - À utiliser uniquement en développement**

    Retourne les statistiques du cache interne du service.
    """
    try:
        cache_size = len(recommendation_service._cache)
        return {
            "cache_entries": cache_size,
            "cache_ttl_hours": recommendation_service._cache_ttl.total_seconds() / 3600,
            "service_uptime_seconds": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug/cache/clear", summary="🧹 Vider le cache")
async def clear_cache():
    """
    **⚠️ Route de debugging - À utiliser uniquement en développement**

    Vide le cache de recommandations pour forcer le recalcul.
    """
    try:
        recommendation_service._cache.clear()
        logger.info("🧹 Cache vidé")
        return {"message": "Cache vidé avec succès", "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route d'exemple pour les étudiants
@router.get("/example", summary="📚 Exemple d'utilisation")
async def get_usage_example():
    """
    **📚 Route pédagogique - Exemples d'utilisation de l'API**

    Retourne des exemples de requêtes et réponses pour aider les étudiants.
    """
    return {
        "title": "🛒 Olist Recommendation API - Exemples",
        "examples": [
            {
                "name": "Recommandations simples",
                "method": "POST",
                "url": "/recommendations",
                "body": {
                    "customer_id": "customer_001",
                    "n_recommendations": 5
                }
            },
            {
                "name": "Recommandations par URL",
                "method": "GET",
                "url": "/recommendations/customer_001?n_recommendations=5"
            },
            {
                "name": "Traitement en lot",
                "method": "POST",
                "url": "/recommendations/batch",
                "body": {
                    "customer_ids": ["customer_001", "customer_002"],
                    "n_recommendations": 10
                }
            },
            {
                "name": "Info modèle",
                "method": "GET",
                "url": "/model/info"
            }
        ],
        "tips": [
            "💡 Utilisez /health pour vérifier si l'API fonctionne",
            "💡 Consultez /customers pour voir la liste des clients",
            "💡 L'API met en cache les recommandations pendant 1h",
            "💡 Utilisez les routes /debug/ pour le développement",
            "💡 Les probabilités près de 1.0 indiquent une forte confiance"
        ]
    }