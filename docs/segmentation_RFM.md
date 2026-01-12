# Segmentation RFM – Cas E-commerce

## 1. Contexte

Dans un site **e-commerce**, la segmentation RFM permet d’analyser le comportement d’achat des clients à partir des données transactionnelles afin de :
- identifier les clients à forte valeur,
- améliorer la fidélisation,
- optimiser les campagnes marketing (email, retargeting, promotions).

---

## 2. Définition de la méthode RFM

La segmentation **RFM** repose sur trois indicateurs clés :

### R – Récence (Recency)
- Nombre de jours depuis le **dernier achat** du client.
- Plus la valeur est faible, plus le client est actif.

### F – Fréquence (Frequency)
- Nombre total de **commandes** passées par le client sur une période donnée.
- Indique la fidélité.

### M – Montant (Monetary)
- **Chiffre d’affaires cumulé** généré par le client.
- Représente la valeur économique du client.

---

## 3. Données nécessaires (e-commerce)

Table de transactions typique :

| client_id | order_id | order_date | order_amount |
|----------|----------|------------|--------------|
| C001 | O1001 | 2025-01-10 | 120 € |
| C001 | O1023 | 2025-03-18 | 80 € |
| C002 | O1045 | 2024-11-02 | 45 € |

---

## 4. Calcul des indicateurs RFM

Pour chaque client :

- **Récence** = Date d’analyse – Date du dernier achat
- **Fréquence** = Nombre total de commandes
- **Montant** = Somme des montants dépensés

Exemple :

| client_id | Recency (jours) | Frequency | Monetary (€) |
|----------|-----------------|-----------|--------------|
| C001 | 15 | 8 | 950 |
| C002 | 120 | 2 | 90 |

---

## 5. Attribution des scores RFM

Chaque indicateur est transformé en **score de 1 à 5** (quintiles) :

- **5 = meilleur score**
- **1 = score le plus faible**

Exemple de règles :

### Récence
- 0–30 jours → 5
- 31–60 jours → 4
- 61–90 jours → 3
- 91–180 jours → 2
- >180 jours → 1

### Fréquence et Montant
- Découpage en quintiles selon la distribution des clients.

---

## 6. Score RFM final

Le score RFM est une **concaténation des trois scores** :

| client_id | R | F | M | Score RFM |
|----------|---|---|---|-----------|
| C001 | 5 | 4 | 5 | 545 |
| C002 | 2 | 2 | 1 | 221 |

---

## 7. Segments clients en e-commerce

| Segment | Score RFM typique | Description | Actions marketing |
|-------|------------------|------------|------------------|
| Clients VIP | 555, 554 | Très actifs et rentables | Offres exclusives, avant-premières |
| Clients fidèles | 454, 444 | Achètent régulièrement | Programme de fidélité |
| Nouveaux clients | 5x1 | Achat récent mais unique | Emails d’onboarding |
| Clients à risque | 2x3 | Inactivité croissante | Coupons, relance email |
| Clients perdus | 111 | Inactifs et peu rentables | Exclusion ou campagnes low-cost |

---

## 8. Cas d’usage marketing concrets

### Email marketing
- VIP → emails premium personnalisés
- À risque → campagnes de réactivation

### Publicité
- Exclure les clients perdus des campagnes payantes
- Lookalike audiences basées sur les VIP

### Promotions
- Réductions ciblées uniquement sur les segments sensibles au prix

---

## 9. Avantages pour le e-commerce

- Segmentation rapide et actionnable
- Basée sur des données transactionnelles fiables
- Améliore le ROI des campagnes marketing
- Facilement automatisable (CRM, marketing automation)

---

## 10. Limites

- Ne prend pas en compte :
  - le type de produits achetés,
  - le canal d’acquisition,
  - les préférences clients.

👉 À compléter avec :
- segmentation comportementale,
- analyse panier,
- scoring prédictif (CLV).

---

## 11. Conclusion

La segmentation RFM est une **base incontournable** pour piloter la relation client en e-commerce. Simple à mettre en place, elle permet de prendre rapidement des décisions marketing ciblées et efficaces.
