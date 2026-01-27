# ===============================================
# 🚀 OLIST RECOMMENDATION SYSTEM - STREAMLIT APP
# Master 2 - SEP
# ===============================================

"""
Interface utilisateur Streamlit pour le système de recommandation Olist.

Cette application web permet de:
- Tester les recommandations personnalisées
- Visualiser les performances du modèle
- Explorer les données et résultats
- Démontrer le système complet aux étudiants

Architecture:
- Streamlit pour l'interface utilisateur
- Appels API REST vers le backend FastAPI
- Visualisations interactives avec Plotly
- Cache pour optimiser les performances
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="🛒 Olist Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ajouter la racine du projet au PYTHONPATH (pour src.config, src.ml_pipeline)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Configuration de l'API (env pour Docker : API_BASE_URL=http://backend:8000/api/v1)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")


# Cache Streamlit pour optimiser les performances
@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def get_customers():
    """Récupère la liste des clients depuis l'API."""
    try:
        response = requests.get(f"{API_BASE_URL}/customers")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error(
            "🔌 Impossible de se connecter à l'API. Assurez-vous que le serveur FastAPI est démarré."
        )
        return []
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return []


@st.cache_data(ttl=300)
def get_model_info():
    """Récupère les informations du modèle depuis l'API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération des infos du modèle: {e}")
        return None


def get_recommendations(customer_id, n_recommendations=10):
    """Obtient les recommandations pour un client."""
    try:
        payload = {"customer_id": customer_id, "n_recommendations": n_recommendations}
        response = requests.post(f"{API_BASE_URL}/recommendations", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération de recommandations: {e}")
        return None


def check_api_health():
    """Vérifie la santé de l'API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except Exception:
        return False


def main():
    """Interface principale de l'application Streamlit."""

    # Header principal
    st.markdown("""
    # 🛒 Olist Recommendation System
    ## Master 2 - SEP

    **Interface de démonstration du système de recommandation e-commerce**
    """)

    # Vérification de la connexion API
    if not check_api_health():
        st.error("""
        🔌 **Connexion à l'API impossible**

        **Comment résoudre:**
        1. Démarrez le serveur FastAPI: `uvicorn backend.app.main:app --reload`
        2. Vérifiez que le serveur tourne sur http://localhost:8000
        3. Consultez les logs pour les erreurs éventuelles
        """)
        st.stop()

    # Sidebar pour la navigation
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        page = st.selectbox(
            "Choisir une page",
            [
                "🎯 Recommandations",
                "📊 Performance du Modèle",
                "😊 Satisfaction Client",
                "🔍 Analyse des Données",
                "😎 CV de Enzo Potier",
            ],
        )

        st.markdown("---")
        st.markdown("### 🔧 Configuration")

        # Status API
        if check_api_health():
            st.success("✅ API connectée")
        else:
            st.error("❌ API déconnectée")

    # Routage vers les différentes pages
    if page == "🎯 Recommandations":
        show_recommendations_page()
    elif page == "📊 Performance du Modèle":
        show_model_performance_page()
    elif page == "😊 Satisfaction Client":
        show_satisfaction_page()
    elif page == "🔍 Analyse des Données":
        show_data_analysis_page()
    elif page == "😎 CV de Enzo Potier":
        show_enzo_cv_page()


def recommend(customer_id: str, k: int):
    MODEL_PATH = "src/ml_pipeline/pkl_docs/svd_recommender.pkl"

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    user_enc = model["user_encoder"]
    item_enc = model["item_encoder"]
    U = model["U"]  # embeddings utilisateurs
    V = model["V"]  # embeddings produits
    train_ui = model["train_ui"]  # matrice user-item (pour filtrer achats passés)
    # Vérification que le client existe
    if customer_id not in user_enc.classes_:
        raise ValueError("❌ Client inconnu (non vu au training)")

    # Encodage du client
    user_id = user_enc.transform([customer_id])[0]

    # Score = similarité utilisateur-produit
    scores = V @ U[user_id]

    # Exclure les produits déjà achetés
    already_bought = train_ui[user_id].indices
    scores[already_bought] = -np.inf

    # Top-K produits
    top_k_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

    # Retour aux product_id d'origine
    recommended_products = item_enc.inverse_transform(top_k_idx)

    return recommended_products


def show_recommendations_page():
    """Page principale de génération de recommandations."""

    st.markdown("## 🎯 Recommandations Personnalisées")

    models = st.selectbox("Selction du modèle", ["Model svd", "Model original"])
    if models == "Model original":
        # Configuration des recommandations
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Sélection du client")

            # Charger la liste des clients
            customers = get_customers()
            if not customers:
                st.warning("Aucun client disponible")
                return

            customer_id = st.selectbox(
                "Client à analyser",
                customers,
                help="Sélectionnez un client pour générer ses recommandations personnalisées",
            )

        with col2:
            st.markdown("### Paramètres")
            n_recommendations = st.slider(
                "Nombre de recommandations",
                min_value=1,
                max_value=20,
                value=10,
                help="Nombre de produits à recommander",
            )

        # Bouton de génération
        if st.button("🚀 Générer les recommandations", type="primary"):
            with st.spinner("Génération des recommandations..."):
                recommendations_data = get_recommendations(customer_id, n_recommendations)

            if recommendations_data:
                display_recommendations(recommendations_data)
            else:
                st.error("Impossible de générer les recommandations")
    elif models == "Model svd":
        # Configuration des recommandations
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Sélection du client (SVD)")

            # Charger la liste des clients
            customers = get_customers()
            if not customers:
                st.warning("Aucun client disponible")
                st.stop()

            customer_id = st.selectbox(
                "Client à analyser (SVD)",
                customers,
                help="Sélectionnez un client pour générer ses recommandations SVD",
                key="svd_customer_id",
            )

        with col2:
            st.markdown("### Paramètres (SVD)")
            n_recommendations = st.slider(
                "Nombre de recommandations (SVD)",
                min_value=1,
                max_value=20,
                value=10,
                help="Nombre de produits à recommander",
                key="svd_n_reco",
            )

        # Bouton de génération
        if st.button("🚀 Générer les recommandations (SVD)", type="primary", key="svd_button"):
            with st.spinner("Génération des recommandations SVD..."):
                recommendations_data = recommend(customer_id, n_recommendations)
                st.write(recommendations_data)
            c1, c2, c3, c4 = st.columns(4)
            # c1.metric()


def display_recommendations(data):
    """Affiche les recommandations de manière interactive."""

    st.markdown("---")
    st.markdown("## 🎁 Recommandations Générées")

    # Informations générales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Client ID", data["customer_id"])
    with col2:
        st.metric("Recommandations", data["total_recommendations"])
    with col3:
        st.metric("Généré le", data["generated_at"][:10])

    # Graphique des probabilités
    recommendations = data["recommendations"]
    df_recs = pd.DataFrame(recommendations)

    fig = px.bar(
        df_recs,
        x="rank",
        y="purchase_probability",
        color="confidence",
        title="📈 Probabilités d'achat par produit",
        labels={
            "rank": "Rang de la recommandation",
            "purchase_probability": "Probabilité d'achat",
            "confidence": "Niveau de confiance",
        },
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Tableau détaillé
    st.markdown("### 📋 Détail des recommandations")

    for _i, rec in enumerate(recommendations):
        with st.expander(
            f"#{rec['rank']} - {rec['product_id']} (Probabilité: {rec['purchase_probability']:.3f})"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Product ID:** {rec['product_id']}")
                st.write(f"**Probabilité:** {rec['purchase_probability']:.3f}")
                st.write(f"**Confiance:** {rec['confidence']}")


def _load_satisfaction_model():
    """Charge le modèle de satisfaction (depuis data/models/satisfaction_model.joblib). Retourne None si absent."""
    try:
        from src.config import MLConfig
        from src.ml_pipeline.models.satisfaction_model import SatisfactionModel

        if not MLConfig.SATISFACTION_MODEL_FILE.exists():
            return None
        return SatisfactionModel.load_model()
    except Exception:
        return None


def precision_at_k(recommended, relevant, k: int) -> float:
    """Precision@K = (# items pertinents dans top-K) / K"""
    if k <= 0:
        return 0.0
    if len(recommended) == 0:
        return 0.0
    rel = set(relevant)
    return len(set(recommended[:k]) & rel) / float(k)


def recall_at_k(recommended, relevant, k: int) -> float:
    """Recall@K = (# items pertinents dans top-K) / (# items pertinents)"""
    if len(relevant) == 0:
        return 0.0
    rel = set(relevant)
    return len(set(recommended[:k]) & rel) / float(len(rel))


def average_precision_at_k(recommended, relevant, k: int) -> float:
    """AP@K = moyenne des precision@i sur les positions i où il y a un hit."""
    if len(relevant) == 0:
        return 0.0

    rel = set(relevant)
    score = 0.0
    hits = 0

    for i, item in enumerate(recommended[:k], start=1):
        if item in rel:
            hits += 1
            score += hits / float(i)

    return score / float(min(len(rel), k))


def hit_rate_at_k(recommended, relevant, k: int) -> float:
    """HitRate@K = 1 si au moins un item pertinent est dans top-K, sinon 0."""
    if len(relevant) == 0:
        return 0.0
    rel = set(relevant)
    return 1.0 if len(set(recommended[:k]) & rel) > 0 else 0.0


def show_model_performance_page():
    """Page d'analyse des performances des modèles (recommandation + satisfaction)."""

    st.markdown("## 📊 Performance des Modèles ML")

    # ---------- Modèle de recommandation (via API) ----------
    st.markdown("### 🎯 Modèle de Recommandation")
    model_info = get_model_info()
    if not model_info:
        st.warning("Impossible de récupérer les infos du modèle de recommandation (API).")
    else:
        metrics = model_info.get("metrics", {})
        feature_importance = model_info.get("feature_importance", [])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Précision Train", f"{metrics.get('train_accuracy', 0):.3f}")
        with col2:
            st.metric("Précision Test", f"{metrics.get('test_accuracy', 0):.3f}")
        with col3:
            st.metric("Score AUC", f"{metrics.get('auc_score', 0):.3f}")
        with col4:
            st.metric("CV Score", f"{metrics.get('cv_mean', 0):.3f}")

        auc_score = metrics.get("auc_score", 0)
        if auc_score >= 0.9:
            st.success("🏆 Performance excellente pour les recommandations.")
        elif auc_score >= 0.8:
            st.success("👍 Très bonne performance.")
        elif auc_score >= 0.7:
            st.info("✅ Performance correcte.")
        else:
            st.warning("⚠️ Performance à améliorer.")

        if feature_importance:
            df_imp = pd.DataFrame(feature_importance)
            fig = px.bar(
                df_imp.head(10),
                x="importance",
                y="feature",
                orientation="h",
                title="Importance des features (recommandation)",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # ---------- Modèle de satisfaction client ----------
    st.markdown("### 😊 Modèle de Satisfaction Client (review_score 1–5)")
    satisfaction_model = _load_satisfaction_model()
    if satisfaction_model is None:
        st.info(
            "Le modèle de satisfaction n’est pas encore entraîné. "
            "Lancez : `uv run python src/scripts/train_satisfaction.py` puis rechargez cette page."
        )
    else:
        m = satisfaction_model.get_metrics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² test", f"{m.get('test_r2', m.get('train_r2', 0)):.3f}")
        with col2:
            st.metric("MAE test", f"{m.get('test_mae', m.get('train_mae', 0)):.3f}")
        with col3:
            st.metric("RMSE test", f"{m.get('test_rmse', m.get('train_rmse', 0)):.3f}")
        with col4:
            st.metric("CV R²", f"{m.get('cv_r2_mean', 0):.3f}")

        df_sat_imp = satisfaction_model.get_feature_importance(top_n=11)
        if not df_sat_imp.empty:
            fig_sat = px.bar(
                df_sat_imp,
                x="importance",
                y="feature",
                orientation="h",
                title="Importance des features (satisfaction)",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig_sat.update_layout(height=400)
            st.plotly_chart(fig_sat, use_container_width=True)
            st.caption(
                "Features : délai livraison, retard vs estimé, nb articles, montant, fret, "
                "historique client, première commande, mois, jour de la semaine."
            )


def show_satisfaction_page():
    """Page dédiée au modèle de satisfaction : prédiction et infos."""

    st.markdown("## 😊 Satisfaction Client")
    st.markdown(
        "Prédiction du **score d’avis** (1–5) à partir des caractéristiques de la commande et du client."
    )

    satisfaction_model = _load_satisfaction_model()
    if satisfaction_model is None:
        st.info(
            "Le modèle de satisfaction n’est pas encore entraîné. "
            "Lancez : `uv run python src/scripts/train_satisfaction.py` puis rechargez cette page."
        )
        return

    # Simulateur de prédiction
    st.markdown("### 🎚️ Simulateur : prédire un score d’avis")
    st.caption(
        "Renseignez les champs pour estimer le score de satisfaction (1–5) qu’un client pourrait donner."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        days_until_delivery = st.number_input(
            "Délai livraison (jours)",
            min_value=0,
            max_value=120,
            value=15,
            help="Jours entre achat et livraison",
        )
        delay_vs_estimated = st.number_input(
            "Retard vs date estimée (jours)",
            min_value=-30,
            max_value=60,
            value=0,
            help="> 0 = livré en retard",
        )
        nb_items = st.number_input(
            "Nb articles dans la commande", min_value=1, max_value=50, value=3
        )
        total_price = st.number_input(
            "Montant total (€)",
            min_value=0.0,
            max_value=5000.0,
            value=150.0,
            step=10.0,
        )
    with col2:
        total_freight = st.number_input(
            "Fret total (€)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
            step=5.0,
        )
        total_orders = st.number_input(
            "Nb total de commandes du client",
            min_value=1,
            max_value=100,
            value=5,
        )
        total_spent = st.number_input(
            "Dépense totale client (€)",
            min_value=0.0,
            max_value=50000.0,
            value=800.0,
            step=100.0,
        )
        is_first_order = st.selectbox(
            "Première commande ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non"
        )
    with col3:
        month = st.slider("Mois de livraison", 1, 12, 6)
        day_of_week = st.slider("Jour de la semaine (0=lun, 6=dim)", 0, 6, 2)
        avg_price_per_item = total_price / nb_items if nb_items else 0.0
        st.metric("Panier moyen par article (€)", f"{avg_price_per_item:.1f}")

    if st.button("🔮 Prédire le score de satisfaction", type="primary"):
        row = pd.DataFrame(
            [
                {
                    "days_until_delivery": float(days_until_delivery),
                    "delay_vs_estimated": float(delay_vs_estimated),
                    "nb_items": int(nb_items),
                    "total_price": float(total_price),
                    "total_freight": float(total_freight),
                    "avg_price_per_item": float(avg_price_per_item),
                    "total_orders": int(total_orders),
                    "total_spent": float(total_spent),
                    "is_first_order": int(is_first_order),
                    "month": int(month),
                    "day_of_week": int(day_of_week),
                }
            ]
        )
        # Aligner l’ordre des colonnes avec le modèle
        if hasattr(satisfaction_model, "feature_columns") and satisfaction_model.feature_columns:
            row = row.reindex(columns=satisfaction_model.feature_columns, fill_value=0)
        pred = satisfaction_model.predict(row)
        score = float(pred[0])
        st.success(f"**Score de satisfaction prédit : {score:.2f} / 5**")
        if score >= 4:
            st.info("Interprétation : client probablement satisfait.")
        elif score >= 3:
            st.info("Interprétation : satisfaction moyenne.")
        else:
            st.warning("Interprétation : risque d’insatisfaction (délai, retard, etc.).")

    # Métriques du modèle (repliable)
    with st.expander("📊 Métriques du modèle de satisfaction"):
        m = satisfaction_model.get_metrics()
        for k, v in m.items():
            st.write(f"- **{k}** : {v:.4f}" if isinstance(v, int | float) else f"- **{k}** : {v}")

    st.divider()
    st.markdown("## 🧠 Performance du modèle SVD (ranking implicite)")

    st.caption(
        "Métriques adaptées à la recommandation implicite : Precision@K, Recall@K, MAP@K, HitRate@K. "
        "Elles évaluent la qualité du TOP-K recommandé (et non une accuracy de classification)."
    )

    # Choix des K
    default_ks = [1, 5, 10, 20]
    ks = st.multiselect(
        "Choisir les valeurs de K à évaluer",
        options=list(range(1, 51)),
        default=default_ks,
        help="K = taille de la liste recommandée (Top-K)",
        key="svd_k_values",
    )

    if not ks:
        st.info("Sélectionne au moins une valeur de K.")
        return

    # Calcul
    try:
        df_svd = compute_svd_metrics_for_ks(SVD_PKL_PATH, ks)
    except FileNotFoundError:
        st.error(f"Fichier PKL introuvable : {SVD_PKL_PATH}")
        return
    except Exception as e:
        st.error(f"Erreur lors du calcul des métriques SVD : {e}")
        return

    # Affichage rapide (pour un K choisi)
    st.markdown("### 🎯 Résumé (K sélectionné)")
    k_focus = st.selectbox(
        "K affiché en métriques", options=df_svd["K"].tolist(), index=0, key="svd_k_focus"
    )
    row = df_svd[df_svd["K"] == k_focus].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Precision@{k_focus}", f"{row['precision']:.4f}")
    c2.metric(f"Recall@{k_focus}", f"{row['recall']:.4f}")
    c3.metric(f"MAP@{k_focus}", f"{row['map']:.4f}")
    c4.metric(f"HitRate@{k_focus}", f"{row['hit_rate']:.4f}")

    st.caption(f"Users évalués : {int(row['users_evaluated'])}")

    # Tableau complet
    st.markdown("### 📋 Détail des métriques par K")
    st.dataframe(
        df_svd.rename(
            columns={
                "K": "K",
                "users_evaluated": "Users évalués",
                "precision": "Precision@K",
                "recall": "Recall@K",
                "map": "MAP@K",
                "hit_rate": "HitRate@K",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Courbes
    st.markdown("### 📈 Évolution des métriques en fonction de K")

    df_long = df_svd.melt(
        id_vars=["K", "users_evaluated"],
        value_vars=["precision", "recall", "map", "hit_rate"],
        var_name="metric",
        value_name="value",
    )

    metric_labels = {
        "precision": "Precision@K",
        "recall": "Recall@K",
        "map": "MAP@K",
        "hit_rate": "HitRate@K",
    }
    df_long["metric"] = df_long["metric"].map(metric_labels)

    fig = px.line(
        df_long,
        x="K",
        y="value",
        color="metric",
        markers=True,
        title="Métriques de ranking (SVD) vs K",
        labels={"value": "Score", "K": "K", "metric": "Métrique"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interprétation courte
    st.markdown("#### 💡 Interprétation (SVD)")
    st.write(
        "- **HitRate@K** : proportion d’utilisateurs pour lesquels au moins 1 item pertinent apparaît dans le Top-K.\n"
        "- **Recall@K** augmente souvent avec K (on retrouve plus d’achats), mais **Precision@K** peut baisser (dilution).\n"
        "- **MAP@K** mesure la qualité de l’ordre des recommandations (plus c’est haut, mieux c’est classé)."
    )


SVD_PKL_PATH = "src/ml_pipeline/pkl_docs/svd_recommender.pkl"


@st.cache_resource
def load_svd_pack(pkl_path: str = SVD_PKL_PATH):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def compute_svd_metrics_for_ks(pkl_path: str, ks: list[int]) -> pd.DataFrame:
    pack = load_svd_pack(pkl_path)

    U = pack["U"]
    V = pack["V"]
    train_ui = pack["train_ui"]
    test_ui = pack["test_ui"]

    n_users = U.shape[0]
    n_items = V.shape[0]

    results = []

    for K in ks:
        precisions, recalls, maps, hits = [], [], [], []
        users_eval = 0

        for u in range(n_users):
            relevant_items = test_ui[u].indices
            if len(relevant_items) == 0:
                continue

            users_eval += 1

            scores = V @ U[u]
            scores[train_ui[u].indices] = -np.inf

            k_eff = min(int(K), n_items)
            top_k = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
            top_k = top_k[np.argsort(-scores[top_k])]

            precisions.append(precision_at_k(top_k, relevant_items, K))
            recalls.append(recall_at_k(top_k, relevant_items, K))
            maps.append(average_precision_at_k(top_k, relevant_items, K))
            hits.append(hit_rate_at_k(top_k, relevant_items, K))

        if users_eval == 0:
            results.append(
                {
                    "K": K,
                    "users_evaluated": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "map": 0.0,
                    "hit_rate": 0.0,
                }
            )
        else:
            results.append(
                {
                    "K": K,
                    "users_evaluated": users_eval,
                    "precision": float(np.mean(precisions)),
                    "recall": float(np.mean(recalls)),
                    "map": float(np.mean(maps)),
                    "hit_rate": float(np.mean(hits)),
                }
            )

    return pd.DataFrame(results).sort_values("K")


def show_data_analysis_page():
    """Page d'analyse exploratoire des données Olist avec choix de thème et visualisations."""

    import pandas as pd
    import plotly.express as px
    import streamlit as st

    # -----------------------
    # Choix du thème
    # -----------------------
    theme = st.sidebar.selectbox(
        "🎨 Choisir le thème des graphiques",
        ["plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
    )

    # -----------------------
    # Chargement des données Olist
    # -----------------------
    BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"
    files_urls = {
        "customers": BASE_URL + "olist_customers_dataset.csv",
        "orders": BASE_URL + "olist_orders_dataset.csv",
        "order_items": BASE_URL + "olist_order_items_dataset.csv",
        "products": BASE_URL + "olist_products_dataset.csv",
        "reviews": BASE_URL + "olist_order_reviews_dataset.csv",
    }

    df_customers = pd.read_csv(files_urls["customers"])
    df_orders = pd.read_csv(files_urls["orders"])
    df_order_items = pd.read_csv(files_urls["order_items"])
    df_products = pd.read_csv(files_urls["products"])
    df_reviews = pd.read_csv(files_urls["reviews"])

    # Connexion DuckDB
    con = duckdb.connect(database=":memory:")
    con.register("customers", df_customers)
    con.register("orders", df_orders)
    con.register("order_items", df_order_items)
    con.register("products", df_products)
    con.register("reviews", df_reviews)

    # -----------------------
    # Onglets
    # -----------------------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "ℹ️ Présentation",
            "📌 Vue générale",
            "👥 Comportement client",
            "🔗 Corrélations",
            "🧩 Segmentation RFM",
            "🚚 Livraison & commandes",
        ]
    )

    # -----------------------
    # TAB 0 — Introduction
    # -----------------------
    with tab0:
        st.markdown("## Bienvenue dans l'Analyse Exploratoire des Données Olist")
        st.write(
            """
            Explorez les habitudes d'achat des clients via :
            - Comportements individuels et globaux
            - Corrélations entre indicateurs
            - Segmentation RFM
            """
        )

    # -----------------------
    # Requête SQL principale
    # -----------------------
    query = """
    WITH delivered_orders AS (
        SELECT o.order_id, o.customer_id, o.order_purchase_timestamp
        FROM orders o
        WHERE o.order_status = 'delivered'
    ),
    orders_with_customer AS (
        SELECT c.customer_unique_id, d.order_id, d.order_purchase_timestamp
        FROM delivered_orders d
        JOIN customers c
            ON d.customer_id = c.customer_id
    ),
    monetary AS (
        SELECT owc.customer_unique_id,
               SUM(oi.price + oi.freight_value) AS total_spent
        FROM orders_with_customer owc
        JOIN order_items oi
            ON owc.order_id = oi.order_id
        GROUP BY owc.customer_unique_id
    ),
    frequency_recency AS (
        SELECT customer_unique_id,
               COUNT(DISTINCT order_id) AS total_orders,
               DATE_DIFF('day', MAX(order_purchase_timestamp)::DATE, CURRENT_DATE) AS days_since_last_order
        FROM orders_with_customer
        GROUP BY customer_unique_id
    ),
    reviews AS (
        SELECT owc.customer_unique_id,
               AVG(r.review_score) AS avg_review_score
        FROM orders_with_customer owc
        JOIN reviews r
            ON owc.order_id = r.order_id
        GROUP BY owc.customer_unique_id
    )
    SELECT fr.customer_unique_id,
           fr.total_orders,
           fr.days_since_last_order,
           r.avg_review_score,
           m.total_spent
    FROM frequency_recency fr
    LEFT JOIN monetary m ON fr.customer_unique_id = m.customer_unique_id
    LEFT JOIN reviews r ON fr.customer_unique_id = r.customer_unique_id
    """

    df_clients = con.execute(query).df()

    # -----------------------
    # TAB 1 — Vue générale
    # -----------------------
    with tab1:
        st.markdown("### 📌 Indicateurs clés")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clients", len(df_clients))
        col2.metric("Panier moyen (€)", f"{df_clients['total_spent'].mean():.0f}")
        col3.metric("Commandes moyennes", f"{df_clients['total_orders'].mean():.1f}")
        col4.metric("Note moyenne", f"{df_clients['avg_review_score'].mean():.2f}")

        st.markdown("### 📊 Distributions")
        fig_orders = px.histogram(
            df_clients,
            x="total_orders",
            nbins=10,
            title="Distribution du nombre de commandes",
            template=theme,
        )
        fig_spent = px.box(
            df_clients,
            y="total_spent",
            title="Distribution des montants dépensés (€)",
            template=theme,
        )
        st.plotly_chart(fig_orders, width="stretch")
        st.plotly_chart(fig_spent, width="stretch")

    # -----------------------
    # TAB 2 — Comportement client
    # -----------------------
    with tab2:
        st.markdown("### 👥 Analyse du comportement d’achat")
        fig_behavior = px.scatter(
            df_clients,
            x="total_orders",
            y="total_spent",
            color="avg_review_score",
            size="days_since_last_order",
            title="Fréquence vs Dépense vs Satisfaction",
            labels={
                "total_orders": "Nombre de commandes",
                "total_spent": "Montant dépensé (€)",
                "avg_review_score": "Note moyenne",
                "days_since_last_order": "Récence (jours)",
            },
            template=theme,
        )
        st.plotly_chart(fig_behavior, width="stretch")

    # -----------------------
    # TAB 2 — Tableau clients
    # -----------------------
    with tab2:
        st.markdown("### 🔢 Tableau client individuel")
        # Tronquer ID pour lisibilité
        df_display = df_clients.copy()
        df_display["customer_unique_id"] = df_display["customer_unique_id"].str[:6]
        st.dataframe(df_display)

    # -----------------------
    # TAB 3 — Corrélation
    # -----------------------
    with tab3:
        st.markdown("### 🔗 Matrice de corrélation")
        numeric_cols = df_clients.select_dtypes(include="number")
        corr = numeric_cols.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            title="Corrélation entre indicateurs clients",
            template=theme,
        )
        st.plotly_chart(fig_corr, width="stretch")

    # -----------------------
    # TAB 4 — Segmentation RFM
    # -----------------------
    with tab4:  # si tu veux un onglet RFM séparé, remplacer tab4 par tab5 et ajuster
        st.markdown("### 📊 Segmentation RFM")
        df_clients["R"] = (
            pd.qcut(df_clients["days_since_last_order"], q=4, labels=False, duplicates="drop") + 1
        )
        df_clients["F"] = (
            pd.qcut(df_clients["total_orders"], q=4, labels=False, duplicates="drop") + 1
        )
        df_clients["M"] = (
            pd.qcut(df_clients["total_spent"], q=4, labels=False, duplicates="drop") + 1
        )
        rfm = df_clients.groupby(["F", "M"]).size().reset_index(name="Clients")
        fig_rfm = px.scatter(
            rfm,
            x="F",
            y="M",
            size="Clients",
            title="Segmentation Fréquence vs Monétaire",
            labels={"F": "Fréquence", "M": "Monétaire"},
            template=theme,
        )
        st.plotly_chart(fig_rfm, width="stretch")

    # -----------------------
    # TAB 5 — Livraison & évolution commandes
    # -----------------------
    with tab5:
        st.markdown("### 🚚 Temps de livraison et évolution des commandes")
        df_orders["order_purchase_timestamp"] = pd.to_datetime(
            df_orders["order_purchase_timestamp"]
        )
        df_orders["order_delivered_customer_date"] = pd.to_datetime(
            df_orders["order_delivered_customer_date"]
        )
        df_orders["delivery_time_days"] = (
            df_orders["order_delivered_customer_date"] - df_orders["order_purchase_timestamp"]
        ).dt.days

        # Livraison moyenne par mois
        df_orders["month"] = df_orders["order_purchase_timestamp"].dt.to_period("M")
        delivery_by_month = df_orders.groupby("month")["delivery_time_days"].mean().reset_index()
        delivery_by_month["month"] = delivery_by_month["month"].dt.to_timestamp()
        fig_delivery = px.bar(
            delivery_by_month,
            x="month",
            y="delivery_time_days",
            title="Temps de livraison moyen par mois",
            labels={"month": "Mois", "delivery_time_days": "Temps de livraison moyen (jours)"},
            template=theme,
        )
        st.plotly_chart(fig_delivery, width="stretch")

        # Évolution du nombre de commandes
        orders_over_time = (
            df_orders.groupby(df_orders["order_purchase_timestamp"].dt.to_period("M"))
            .size()
            .reset_index(name="nb_orders")
        )
        orders_over_time["order_purchase_timestamp"] = orders_over_time[
            "order_purchase_timestamp"
        ].dt.to_timestamp()
        fig_orders_time = px.line(
            orders_over_time,
            x="order_purchase_timestamp",
            y="nb_orders",
            title="Évolution du nombre de commandes",
            labels={"order_purchase_timestamp": "Date", "nb_orders": "Nombre de commandes"},
            template=theme,
        )
        st.plotly_chart(fig_orders_time, width="stretch")


def show_enzo_cv_page():
    """Affiche le CV d'Enzo Potier dans un onglet Streamlit et propose le téléchargement."""
    st.markdown("## 😎 CV de Enzo Potier")
    st.write("Voici le CV de mon camarade Enzo Potier.")

    cv_path = "images/cv_enzo_potier.pdf"

    if os.path.exists(cv_path):
        # Lecture du PDF pour le téléchargement
        with open(cv_path, "rb") as f:
            pdf_data = f.read()
        st.download_button(
            label="📄 Télécharger le CV",
            data=pdf_data,
            file_name="cv_enzo_potier.pdf",
            mime="application/pdf",
        )

    cv_image_path = "images/cv_enzo_potier.png"  # chemin vers ton PNG

    try:
        st.image(cv_image_path, width=700)
    except FileNotFoundError:
        st.warning(
            "Image du CV introuvable. Place le fichier `cv_enzo_potier.png` dans le dossier `images/`."
        )


if __name__ == "__main__":
    main()
