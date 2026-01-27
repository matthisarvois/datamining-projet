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

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

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
    elif page == "🔍 Analyse des Données":
        show_data_analysis_page()
    elif page == "😎 CV de Enzo Potier":
        show_enzo_cv_page()


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


def show_model_performance_page():
    """Page d'analyse des performances du modèle."""

    st.markdown("## 📊 Performance du Modèle ML")

    model_info = get_model_info()
    if not model_info:
        st.error("Impossible de récupérer les informations du modèle")
        return

    metrics = model_info.get("metrics", {})
    feature_importance = model_info.get("feature_importance", [])

    # Métriques principales
    st.markdown("### 🎯 Métriques de Performance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Précision Train",
            f"{metrics.get('train_accuracy', 0):.3f}",
            help="Précision sur les données d'entraînement",
        )
    with col2:
        st.metric(
            "Précision Test",
            f"{metrics.get('test_accuracy', 0):.3f}",
            help="Précision sur les données de test",
        )
    with col3:
        st.metric(
            "Score AUC", f"{metrics.get('auc_score', 0):.3f}", help="Area Under the ROC Curve"
        )
    with col4:
        st.metric(
            "CV Score", f"{metrics.get('cv_mean', 0):.3f}", help="Score de validation croisée"
        )

    # Interprétation des résultats
    auc_score = metrics.get("auc_score", 0)
    if auc_score >= 0.9:
        st.success(
            "🏆 Performance excellente! Le modèle distingue très bien les clients qui vont acheter."
        )
    elif auc_score >= 0.8:
        st.success("👍 Très bonne performance! Le modèle est fiable pour les recommandations.")
    elif auc_score >= 0.7:
        st.info("✅ Performance correcte. Le modèle peut être amélioré.")
    else:
        st.warning("⚠️ Performance faible. Considérez l'amélioration du modèle.")

    # Importance des features
    if feature_importance:
        st.markdown("### 🔍 Importance des Features")

        df_importance = pd.DataFrame(feature_importance)
        fig_importance = px.bar(
            df_importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 des features les plus importantes",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)

        # Explication pédagogique
        st.markdown("#### 💡 Interprétation")
        st.write("""
        **L'importance des features nous indique:**
        - Quelles informations client sont les plus prédictives
        - Comment améliorer le modèle en collectant de meilleures données
        - Quels aspects du comportement client privilégier

        **Features typiques importantes:**
        - `total_spent`: Montant total dépensé par le client
        - `total_orders`: Nombre de commandes passées
        - `avg_review_score`: Satisfaction moyenne du client
        - `days_since_last_order`: Récence de la dernière commande
        """)


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
