# ===============================================
# OLIST - MODÈLE DE SATISFACTION CLIENT
# Master 2 - SEP
# ===============================================

"""
Modèle de prédiction de la satisfaction client (score d'avis 1–5).

Features utilisées:
- Commande: délai livraison, retard vs estimé, nb articles, montant, fret, panier moyen
- Client: nb commandes, dépense totale, première commande ou non
- Temporal: mois, jour de la semaine (saisonnalité)

Algorithme: HistGradientBoostingRegressor.
Prédictions bornées entre 1 et 5.
"""

from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MLConfig
from src.ml_pipeline.preprocessing.feature_engineering import load_and_prepare_data


def build_satisfaction_features(
    reviews: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construit les features et la cible pour le modèle de satisfaction.

    Une ligne par avis. Features:
    - days_until_delivery: délai réel achat → livraison
    - delay_vs_estimated: (livré - estimé) en jours (>0 = en retard)
    - nb_items, total_price, total_freight, avg_price_per_item
    - total_orders, total_spent (historique client)
    - is_first_order: 1 si première commande du client
    - month, day_of_week: saisonnalité
    Cible: review_score (1–5).
    """
    order_cols = [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "order_status",
    ]
    order_cols = [c for c in order_cols if c in orders.columns]
    rev_ord = reviews.merge(orders[order_cols], on="order_id", how="inner")

    rev_ord["order_purchase_timestamp"] = pd.to_datetime(rev_ord["order_purchase_timestamp"])
    rev_ord["order_delivered_customer_date"] = pd.to_datetime(
        rev_ord["order_delivered_customer_date"]
    )
    if "order_estimated_delivery_date" in rev_ord.columns:
        rev_ord["order_estimated_delivery_date"] = pd.to_datetime(
            rev_ord["order_estimated_delivery_date"], errors="coerce"
        )

    rev_ord["days_until_delivery"] = (
        rev_ord["order_delivered_customer_date"] - rev_ord["order_purchase_timestamp"]
    ).dt.days

    if "order_estimated_delivery_date" in rev_ord.columns:
        rev_ord["delay_vs_estimated"] = (
            rev_ord["order_delivered_customer_date"] - rev_ord["order_estimated_delivery_date"]
        ).dt.days
        rev_ord["delay_vs_estimated"] = rev_ord["delay_vs_estimated"].fillna(0)
    else:
        rev_ord["delay_vs_estimated"] = 0

    agg_dict = {
        "nb_items": ("order_item_id", "count"),
        "total_price": ("price", "sum"),
    }
    if "freight_value" in order_items.columns:
        agg_dict["total_freight"] = ("freight_value", "sum")
    order_agg = order_items.groupby("order_id").agg(**agg_dict).reset_index()
    if "total_freight" not in order_agg.columns:
        order_agg["total_freight"] = 0.0
    rev_ord = rev_ord.merge(order_agg, on="order_id", how="left")
    rev_ord["nb_items"] = rev_ord["nb_items"].fillna(0).astype(int)
    rev_ord["total_price"] = rev_ord["total_price"].fillna(0)
    rev_ord["total_freight"] = rev_ord["total_freight"].fillna(0)
    rev_ord["avg_price_per_item"] = np.where(
        rev_ord["nb_items"] > 0,
        rev_ord["total_price"] / rev_ord["nb_items"],
        0,
    )

    cust_orders = (
        orders.groupby("customer_id").agg(total_orders=("order_id", "count")).reset_index()
    )
    cust_spent = (
        order_items.merge(orders[["order_id", "customer_id"]], on="order_id")
        .groupby("customer_id")["price"]
        .sum()
        .to_frame("total_spent")
        .reset_index()
    )
    cust = cust_orders.merge(cust_spent, on="customer_id", how="left")
    cust["total_spent"] = cust["total_spent"].fillna(0)
    rev_ord = rev_ord.merge(cust, on="customer_id", how="left")
    rev_ord["total_orders"] = rev_ord["total_orders"].fillna(1).astype(int)
    rev_ord["total_spent"] = rev_ord["total_spent"].fillna(0)
    rev_ord["is_first_order"] = (rev_ord["total_orders"] <= 1).astype(int)

    rev_ord["month"] = rev_ord["order_delivered_customer_date"].dt.month
    rev_ord["day_of_week"] = rev_ord["order_delivered_customer_date"].dt.dayofweek

    feature_cols = [
        "days_until_delivery",
        "delay_vs_estimated",
        "nb_items",
        "total_price",
        "total_freight",
        "avg_price_per_item",
        "total_orders",
        "total_spent",
        "is_first_order",
        "month",
        "day_of_week",
    ]
    for c in feature_cols:
        if c not in rev_ord.columns:
            rev_ord[c] = 0
    X = rev_ord[feature_cols].copy()
    X["days_until_delivery"] = X["days_until_delivery"].clip(lower=0).fillna(0)
    X["delay_vs_estimated"] = X["delay_vs_estimated"].fillna(0)
    y = rev_ord["review_score"].astype(float)

    return X, y


class SatisfactionModel:
    """
    Modèle de prédiction du score de satisfaction (review_score 1–5).
    HistGradientBoostingRegressor + StandardScaler.
    Prédictions bornées entre 1 et 5.
    """

    def __init__(self, model_params: dict[str, Any] | None = None) -> None:
        params = model_params or getattr(MLConfig, "SATISFACTION_MODEL_PARAMS", {})
        if not params:
            params = {"max_iter": 200, "max_depth": 8, "learning_rate": 0.05, "random_state": 42}
        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", HistGradientBoostingRegressor(**params)),
            ]
        )
        self.feature_columns: list[str] = []
        self.is_trained: bool = False
        self.feature_importance_: pd.DataFrame | None = None
        self.training_metrics_: dict[str, float] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SatisfactionModel":
        """Entraîne le modèle."""
        self.feature_columns = X.columns.tolist()
        self.pipeline.fit(X, y)

        reg = cast(HistGradientBoostingRegressor, self.pipeline.named_steps["regressor"])
        imp = getattr(reg, "feature_importances_", None)
        if imp is not None:
            self.feature_importance_ = pd.DataFrame(
                {"feature": self.feature_columns, "importance": imp}
            ).sort_values("importance", ascending=False)
        else:
            n = min(5000, len(X))
            idx = X.sample(n=n, random_state=42).index if len(X) > n else X.index
            perm = permutation_importance(
                self.pipeline, X.loc[idx], y.loc[idx], n_repeats=3, random_state=42
            )
            self.feature_importance_ = pd.DataFrame(
                {"feature": self.feature_columns, "importance": perm.importances_mean}
            ).sort_values("importance", ascending=False)
        pred = self.pipeline.predict(X)
        self.training_metrics_ = {
            "train_mae": float(mean_absolute_error(y, pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "train_r2": float(r2_score(y, pred)),
        }
        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit le score de satisfaction (1–5), borné entre 1 et 5."""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        pred = self.pipeline.predict(X)
        return np.clip(pred, 1.0, 5.0)

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques d'entraînement."""
        if not self.is_trained or self.training_metrics_ is None:
            return {}
        return dict(self.training_metrics_)

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Retourne l'importance des features."""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        return self.feature_importance_.head(top_n)

    def save_model(self, filepath: Path | None = None) -> None:
        """Sauvegarde le modèle dans data/models/."""
        if not self.is_trained:
            raise ValueError("Impossible de sauvegarder un modèle non entraîné")
        path = filepath or MLConfig.SATISFACTION_MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"   Modèle satisfaction sauvegardé: {path}")

    @classmethod
    def load_model(cls, filepath: Path | None = None) -> "SatisfactionModel":
        """Charge un modèle pré-entraîné."""
        path = filepath or MLConfig.SATISFACTION_MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"Modèle satisfaction non trouvé: {path}")
        model = joblib.load(path)
        print(f"   Modèle satisfaction chargé: {path}")
        return cast(SatisfactionModel, model)


class SatisfactionPipeline:
    """Pipeline complet: chargement données → features → entraînement → sauvegarde."""

    def __init__(self) -> None:
        self.model = SatisfactionModel()

    def train_pipeline(self, raw_data_dir: Path) -> dict[str, Any]:
        """Entraîne le modèle et le sauvegarde dans data/models/satisfaction_model.joblib."""
        print("=" * 50)
        print("ENTRAÎNEMENT MODÈLE DE SATISFACTION CLIENT")
        print("=" * 50)

        customers, orders, order_items, products, reviews = load_and_prepare_data(raw_data_dir)
        if reviews.empty or "review_score" not in reviews.columns:
            raise ValueError("Données d'avis (reviews) manquantes ou sans review_score.")

        X, y = build_satisfaction_features(reviews, orders, order_items, customers)
        print(f"   {len(X)} avis, {X.shape[1]} features, cible review_score (1–5)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MLConfig.TEST_SIZE, random_state=MLConfig.RANDOM_STATE
        )
        self.model.fit(X_train, y_train)

        pred_test = self.model.predict(X_test)
        metrics = {
            **self.model.get_metrics(),
            "test_mae": float(mean_absolute_error(y_test, pred_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "test_r2": float(r2_score(y_test, pred_test)),
        }

        cv_scores = cross_val_score(
            self.model.pipeline, X, y, cv=min(MLConfig.CV_FOLDS, 5), scoring="r2"
        )
        metrics["cv_r2_mean"] = float(cv_scores.mean())
        metrics["cv_r2_std"] = float(cv_scores.std())

        self.model.training_metrics_ = metrics
        self.model.save_model()

        print("\nPipeline satisfaction entraîné avec succès.")
        return metrics
