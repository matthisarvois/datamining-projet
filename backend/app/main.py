# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - MAIN APP
# Master 2 - SEP
# ===============================================

"""
Application FastAPI principale pour le système de recommandation Olist.

Cette application expose une API REST complète pour:
- Générer des recommandations personnalisées
- Consulter les performances du modèle
- Monitoring et debugging

Architecture:
- FastAPI pour l'API REST
- Pydantic pour la validation des données
- Service pattern pour la logique métier
- Logging structuré
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import APIConfig
from backend.app.routers.recommendations import router as recommendations_router
from backend.app.services.recommendation_service import recommendation_service

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="[{asctime}] {levelname:<8} {name}: {message}",
    style="{",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/olist_api.log", mode="a", encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.

    Startup:
    - Initialise le service de recommandation
    - Charge le modèle ML
    - Prépare les données

    Shutdown:
    - Nettoie les ressources
    - Ferme les connexions
    """
    # Startup
    logger.info("🚀 Démarrage de l'application Olist Recommendation API")

    # Initialiser le service de recommandation
    success = await recommendation_service.initialize()
    if not success:
        logger.error("❌ Échec de l'initialisation du service")
        # En production, vous pourriez vouloir arrêter l'application ici
        # sys.exit(1)
    else:
        logger.info("✅ Service de recommandation initialisé")

    yield

    # Shutdown
    logger.info("🔽 Arrêt de l'application")

# Créer l'application FastAPI
app = FastAPI(
    title=APIConfig.TITLE,
    version=APIConfig.VERSION,
    description=APIConfig.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware CORS pour permettre les requêtes cross-origin (nécessaire pour Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit par défaut
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Middleware de sécurité pour les hosts autorisés
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
)

# Gestionnaire d'exceptions global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions pour un retour d'erreur cohérent.
    """
    logger.error(f"❌ Erreur non gérée: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Une erreur inattendue s'est produite",
            "details": str(exc) if app.debug else None,
            "timestamp": str(Path(__file__).parent.parent.parent)
        }
    )

# Route racine avec informations de base
@app.get("/", summary="🏠 Page d'accueil de l'API")
async def root():
    """
    Page d'accueil de l'API avec liens utiles.

    **Informations fournies:**
    - Version de l'API
    - Liens vers la documentation
    - État du système
    - Guide de démarrage rapide
    """
    return {
        "message": "🛒 Bienvenue sur l'API Olist Recommendation System",
        "version": APIConfig.VERSION,
        "title": APIConfig.TITLE,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "quick_start": {
            "health_check": "/health",
            "get_customers": "/customers",
            "get_recommendations": "POST /recommendations",
            "model_info": "/model/info",
            "examples": "/example"
        },
        "tips": [
            "💡 Commencez par /health pour vérifier que l'API fonctionne",
            "💡 Consultez /customers pour voir les clients disponibles",
            "💡 Testez avec /recommendations pour obtenir des recommandations",
            "💡 Utilisez /docs pour une documentation interactive",
            "💡 Explorez /model/info pour comprendre le modèle ML"
        ]
    }

# Inclure les routers
app.include_router(
    recommendations_router,
    prefix="/api/v1",
    tags=["Recommandations"]
)

# Routes utiles pour les étudiants
@app.get("/api/v1/info", summary="ℹ️ Informations sur l'API")
async def get_api_info():
    """
    **📚 Route pédagogique - Informations techniques sur l'API**

    Retourne des informations détaillées sur l'architecture et les technologies utilisées.
    """
    return {
        "api": {
            "name": APIConfig.TITLE,
            "version": APIConfig.VERSION,
            "framework": "FastAPI",
            "python_version": sys.version.split()[0],
            "documentation": "OpenAPI 3.0"
        },
        "ml_stack": {
            "model": "RandomForest (scikit-learn)",
            "features": "RFM Analysis + Product Features",
            "approach": "Hybrid Recommendation (Collaborative + Content-based)"
        },
        "architecture": {
            "pattern": "Service Layer Architecture",
            "database": "CSV Files (demo) / PostgreSQL (production)",
            "caching": "In-memory with TTL",
            "logging": "Structured logging"
        },
        "endpoints": {
            "total": len(app.routes),
            "recommendation_endpoints": 4,
            "utility_endpoints": 3,
            "debug_endpoints": 2
        }
    }

# Route pour les métriques (monitoring)
@app.get("/metrics", summary="📊 Métriques de monitoring")
async def get_metrics():
    """
    **📊 Métriques pour monitoring et alerting**

    Format compatible avec Prometheus/Grafana.
    """
    import time
    import psutil
    import os

    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "process": {
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
            },
            "application": {
                "model_loaded": recommendation_service.model is not None,
                "cache_entries": len(recommendation_service._cache),
                "service_healthy": recommendation_service.is_healthy()
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": f"Could not collect metrics: {e}"}

def main():
    """
    Point d'entrée principal pour lancer l'API.

    Usage:
        python -m backend.app.main
        ou
        uvicorn backend.app.main:app --reload
    """
    logger.info("🚀 Lancement du serveur FastAPI")

    uvicorn.run(
        "backend.app.main:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=True,  # Rechargement automatique en développement
        log_level="info",
        reload_dirs=["backend"],  # Surveiller seulement le dossier backend
    )

if __name__ == "__main__":
    main()