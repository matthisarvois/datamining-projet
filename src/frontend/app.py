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


def show_recommendations_page():
    """Page principale de génération de recommandations."""

    st.markdown("## 🎯 Recommandations Personnalisées")

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


def show_data_analysis_page():
    """Page d'analyse exploratoire des données avec choix de thème."""

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
    # Données simulées
    # -----------------------
    np.random.seed(42)
    n_customers = 100

    df = pd.DataFrame(
        {
            "Total Orders": np.random.poisson(3, n_customers) + 1,
            "Total Spent": np.random.exponential(200, n_customers) + 50,
            "Avg Review Score": np.random.normal(4.0, 0.8, n_customers).clip(1, 5),
            "Days Since Last Order": np.random.exponential(30, n_customers) + 1,
        }
    )

    # 2. CHARGEMENT DES DONNÉES OLIST

    # URL de base du dépôt officiel Olist (miroir Kaggle)
    BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"

    # Dictionnaire des fichiers principaux
    files_urls = {
        "customers": BASE_URL + "olist_customers_dataset.csv",
        "orders": BASE_URL + "olist_orders_dataset.csv",
        "order_items": BASE_URL + "olist_order_items_dataset.csv",
        "products": BASE_URL + "olist_products_dataset.csv",
        "reviews": BASE_URL + "olist_order_reviews_dataset.csv",
        "sellers": BASE_URL + "olist_sellers_dataset.csv",
    }

    # Tentative de chargement des données réelles

    df_customers = pd.read_csv(files_urls["customers"])
    df_orders = pd.read_csv(files_urls["orders"])
    df_order_items = pd.read_csv(files_urls["order_items"])
    df_products = pd.read_csv(files_urls["products"])
    df_reviews = pd.read_csv(files_urls["reviews"])

    # Création d'une connexion DuckDB en mémoire
    con = duckdb.connect(database=":memory:")

    # Enregistrement des DataFrames pandas comme tables SQL
    con.register("customers", df_customers)
    con.register("orders", df_orders)
    con.register("order_items", df_order_items)
    con.register("products", df_products)
    con.register("reviews", df_reviews)

    # -----------------------
    # Onglets
    # -----------------------
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        [
            "ℹ️ Présentation",
            "📌 Vue générale",
            "👥 Comportement client",
            "🔗 Corrélations",
            "🧩 Segmentation RFM",
        ]
    )

    # -----------------------
    # TAB 0 — Introduction
    # -----------------------
    with tab0:
        st.markdown("## Bienvenue dans l'Analyse Exploratoire des Données Olist")
        st.write("""
        Cette section permet de découvrir les habitudes d'achat des clients
        à travers des visualisations interactives et des métriques clés.

        Vous pourrez explorer :
        - Les comportements d'achat individuels et globaux
        - Les corrélations entre les différentes variables
        - La segmentation RFM pour identifier les profils clients
        """)

        st.image("images/analyse_dashboard.jpg", width=900)
        st.markdown(
            "💡 Explorez les onglets pour analyser les indicateurs, visualisations et segments clients."
        )

    # -----------------------
    # TAB 1 — Vue générale
    # -----------------------
    with tab1:
        st.markdown("### 📌 Indicateurs clés")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clients", len(df))
        col2.metric("Panier moyen (€)", f"{df['Total Spent'].mean():.0f}")
        col3.metric("Commandes moyennes", f"{df['Total Orders'].mean():.1f}")
        col4.metric("Note moyenne", f"{df['Avg Review Score'].mean():.2f}")

        st.markdown("### 📊 Distributions")

        # Compute total orders per customer via DuckDB
        query = """
        SELECT
            c.customer_unique_id,
            COUNT(o.order_id) AS total_orders
        FROM orders o
        JOIN customers c
            ON o.customer_id = c.customer_id
        GROUP BY c.customer_unique_id
        """
        df_customers = con.execute(query).df()
        col1, col2 = st.columns(2)

        with col1:
            fig_orders = px.histogram(
                df_customers,
                x="total_orders",
                nbins=10,
                title="Distribution du nombre de commandes",
                template=theme,
            )
            st.plotly_chart(fig_orders, use_container_width=True)

        with col2:
            fig_spent = px.box(
                df, y="Total Spent", title="Distribution des montants dépensés (€)", template=theme
            )
            st.plotly_chart(fig_spent, use_container_width=True)

    # -----------------------
    # TAB 2 — Comportement client
    # -----------------------
    with tab2:
        st.markdown("### 👥 Analyse du comportement d’achat")

        fig_behavior = px.scatter(
            df,
            x="Total Orders",
            y="Total Spent",
            color="Avg Review Score",
            size="Days Since Last Order",
            title="Fréquence d’achat vs Dépense vs Satisfaction",
            labels={
                "Total Orders": "Nombre de commandes",
                "Total Spent": "Montant dépensé (€)",
                "Avg Review Score": "Note moyenne",
                "Days Since Last Order": "Récence (jours)",
            },
            template=theme,
        )
        st.plotly_chart(fig_behavior, use_container_width=True)

    # -----------------------
    # TAB 3 — Corrélations
    # -----------------------
    with tab3:
        st.markdown("### 🔗 Corrélations entre variables")

        corr = df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            title="Matrice de corrélation",
            template=theme,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # -----------------------
    # TAB 4 — Segmentation RFM
    # -----------------------
    with tab4:
        st.markdown("### 📊 Segmentation RFM simplifiée")

        df["R"] = pd.qcut(df["Days Since Last Order"], 4, labels=[4, 3, 2, 1])
        df["F"] = pd.qcut(df["Total Orders"], 4, labels=[1, 2, 3, 4])
        df["M"] = pd.qcut(df["Total Spent"], 4, labels=[1, 2, 3, 4])

        rfm = df.groupby(["F", "M"]).size().reset_index(name="Clients")

        fig_rfm = px.scatter(
            rfm,
            x="F",
            y="M",
            size="Clients",
            title="Segmentation Fréquence vs Monétaire",
            labels={"F": "Fréquence", "M": "Monétaire"},
            template=theme,
        )
        st.plotly_chart(fig_rfm, use_container_width=True)


if __name__ == "__main__":
    main()
