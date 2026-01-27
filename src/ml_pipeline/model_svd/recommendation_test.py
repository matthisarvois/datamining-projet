import pickle

import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "src/ml_pipeline/pkl_docs/svd_recommender.pkl"
TOP_K = 10


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


# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(customer_id: str, k: int):
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


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    customer_id = input("Entre un customer_id : ").strip()

    products = recommend(customer_id, k=10)

    print("\n🎯 Produits recommandés :")
    for p in products:
        print("-", p)
