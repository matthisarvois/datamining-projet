import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, normalize

# -----------------------------
# CONFIG
# -----------------------------
# Chemins vers les fichiers CSV nécessaires
PATH_ORDERS = "data/raw/olist_orders_dataset.csv"
PATH_ORDER_ITEMS = "data/raw/olist_order_items_dataset.csv"

# Graine aléatoire pour rendre les résultats reproductibles
RANDOM_SEED = 42

# Proportion d'interactions mises de côté pour le test
TEST_RATIO = 0.2

# Dimension latente k : taille des embeddings (users/items)
SVD_COMPONENTS = 64


# -----------------------------
# UTIL: split implicite (holdout)
# -----------------------------
def random_holdout(interaction_matrix, test_ratio: float = 0.2, seed: int = 42):
    """
    Split aléatoire d'une matrice d'interactions implicites (sparse).
    On ne split PAS sur les users ou items, mais sur les interactions non-nulles (nnz).

    - interaction_matrix : matrice user-item (sparse)
    - test_ratio : proportion d'interactions à mettre dans le test
    - seed : graine aléatoire

    Retourne :
    - train (CSR) : matrice d'entraînement
    - test  (CSR) : matrice de test (sans overlap avec train)
    """
    rng = np.random.default_rng(seed)

    # On passe en format COO pour accéder facilement aux coordonnées (row, col, data)
    coo = interaction_matrix.tocoo()

    # Nombre d'interactions (cases non nulles)
    nnz = coo.nnz

    # Indices [0..nnz-1] qu'on va mélanger pour faire un split aléatoire
    idx = np.arange(nnz)
    rng.shuffle(idx)

    # Point de coupe : une partie pour train, le reste pour test
    cut = int((1.0 - test_ratio) * nnz)
    train_idx = idx[:cut]
    test_idx = idx[cut:]

    # Construction de la matrice train avec les interactions sélectionnées
    train = coo_matrix(
        (coo.data[train_idx], (coo.row[train_idx], coo.col[train_idx])),
        shape=interaction_matrix.shape,
        dtype=np.float32,
    ).tocsr()  # conversion en CSR pour calculs rapides

    # Construction de la matrice test avec les interactions restantes
    test = coo_matrix(
        (coo.data[test_idx], (coo.row[test_idx], coo.col[test_idx])),
        shape=interaction_matrix.shape,
        dtype=np.float32,
    ).tocsr()

    # Sécurité : on enlève toute interaction du test qui serait aussi dans train
    # (en pratique, ça évite les overlaps si jamais il y en a)
    # test = test.multiply(train == 0)

    return train, test


# -----------------------------
# MAIN
# -----------------------------
def main_model():
    # 1) Lecture des fichiers CSV
    orders = pd.read_csv(PATH_ORDERS)
    order_items = pd.read_csv(PATH_ORDER_ITEMS)

    # 2) Fusion commandes + items : on obtient les lignes "client - produit" via order_id
    df = orders.merge(order_items, on="order_id", how="inner")

    # Filtre optionnel : on ne garde que les commandes livrées (signal d'achat "confirmé")
    if "order_status" in df.columns:
        df = df[df["order_status"] == "delivered"]

    # 3) Construction des interactions implicites : (customer_id, product_id)
    # drop_duplicates évite de compter plusieurs fois le même achat si répétitions
    interactions = df[["customer_id", "product_id"]].drop_duplicates()

    # 4) Encodage en entiers (obligatoire pour construire une matrice sparse)
    # - user_enc transforme customer_id -> [0..n_users-1]
    # - item_enc transforme product_id   -> [0..n_items-1]
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    interactions["user_id"] = user_enc.fit_transform(interactions["customer_id"])
    interactions["item_id"] = item_enc.fit_transform(interactions["product_id"])

    # Nombre d'utilisateurs et d'items distincts
    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()

    # 5) Création de la matrice user-item implicite (1 = achat)
    # On met des 1 partout où il y a une interaction (achat)
    user_item = coo_matrix(
        (
            np.ones(len(interactions), dtype=np.float32),  # valeurs = 1
            (interactions["user_id"].to_numpy(), interactions["item_id"].to_numpy()),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()  # CSR = pratique pour accès par user (filtrage déjà achetés)

    # 6) Split train/test sur les interactions
    train_ui, test_ui = random_holdout(user_item, test_ratio=TEST_RATIO, seed=RANDOM_SEED)

    # 7) Factorisation par TruncatedSVD (équivalent à une factorisation matricielle)
    # Objectif : approximer train_ui ≈ U @ V^T
    # - U : embeddings users (n_users, k)
    # - V : embeddings items (n_items, k)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)

    # fit_transform donne U
    U = svd.fit_transform(train_ui)  # shape (n_users, k)

    # components_ a shape (k, n_items), on transpose pour obtenir (n_items, k)
    V = svd.components_.T  # shape (n_items, k)

    # 8) Normalisation des embeddings
    # Après normalisation, le score (U[u]·V[i]) se rapproche d'un cosinus similarity
    # => souvent meilleur pour le ranking Top-K.
    U = normalize(U)
    V = normalize(V)

    # 9) Sauvegarde du "modèle" (en pratique : encodeurs + embeddings + matrices)
    pkl_path = Path("src/ml_pipeline/pkl_docs/svd_recommender.pkl")
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "user_encoder": user_enc,
                "item_encoder": item_enc,
                "U": U,
                "V": V,
                "train_ui": train_ui,  # utile pour filtrer les produits déjà achetés
                "test_ui": test_ui,  # utile si tu veux évaluer plus tard
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Logs
    print("✅ Modèle entraîné et sauvegardé dans svd_recommender.pkl")
    print(f"Users: {n_users:,} | Items: {n_items:,} | Interactions: {user_item.nnz:,}")


# Point d'entrée du script
if __name__ == "__main__":
    main_model()
