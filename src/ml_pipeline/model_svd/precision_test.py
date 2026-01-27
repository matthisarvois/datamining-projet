import pickle

import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "src/ml_pipeline/pkl_docs/svd_recommender.pkl"
K = 20


# -----------------------------
# LOAD MODEL
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

U = model["U"]  # (n_users, k_latent)
V = model["V"]  # (n_items, k_latent)
train_ui = model["train_ui"]  # CSR user-item (train)
test_ui = model["test_ui"]  # CSR user-item (test)


# -----------------------------
# METRICS
# -----------------------------
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


# -----------------------------
# EVALUATION LOOP
# -----------------------------
precisions, recalls, maps, hits = [], [], [], []

n_users = U.shape[0]
n_items = V.shape[0]

for u in range(n_users):
    # Items pertinents = achats du user dans le TEST
    relevant_items = test_ui[u].indices
    if len(relevant_items) == 0:
        continue

    # Scores pour tous les items
    scores = V @ U[u]  # shape (n_items,)

    # Exclure items déjà vus/achetés dans le TRAIN
    seen_items = train_ui[u].indices
    scores[seen_items] = -np.inf

    # Top-K
    k_eff = min(K, n_items)
    top_k = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
    top_k = top_k[np.argsort(-scores[top_k])]

    precisions.append(precision_at_k(top_k, relevant_items, K))
    recalls.append(recall_at_k(top_k, relevant_items, K))
    maps.append(average_precision_at_k(top_k, relevant_items, K))
    hits.append(hit_rate_at_k(top_k, relevant_items, K))


# -----------------------------
# RESULTS
# -----------------------------
if len(precisions) == 0:
    print("⚠️ Aucun utilisateur n'a pu être évalué (test_ui vide pour tous).")
else:
    print("📊 ÉVALUATION DU MODÈLE (ranking implicite)")
    print(f"Users évalués : {len(precisions)}")
    print(f"Precision@{K} : {np.mean(precisions):.4f}")
    print(f"Recall@{K}    : {np.mean(recalls):.4f}")
    print(f"MAP@{K}       : {np.mean(maps):.4f}")
    print(f"HitRate@{K}   : {np.mean(hits):.4f}")
