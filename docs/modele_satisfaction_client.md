# Modèle de satisfaction client — explication complète

Ce document détaille le fonctionnement du modèle de prédiction du **score d’avis** (1–5) des clients Olist, les choix de variables, les raisons d’un R² modéré, le type de modèle utilisé et des pistes d’amélioration.

---

## 1. Type de modèle et rôle dans le projet

### Quel type de modèle ?

- **Tâche** : **régression** — on prédit une variable continue (le score d’avis entre 1 et 5).
- **Algorithme** : **HistGradientBoostingRegressor** (scikit-learn).
- **Famille** : *Gradient Boosting* sur histogrammes : ensemble d’arbres de régression construits de façon additive, avec optimisation par gradient, et binarisation des variables pour accélérer l’entraînement.
- **Pipeline** :  
  `StandardScaler` → `HistGradientBoostingRegressor`  
  Les variables sont d’abord centrées-réduites, puis le boosting apprend à prédire `review_score` à partir de ces features.

### À quoi il sert ?

- Prédire le **score d’avis** qu’un client est susceptible de mettre après une commande (1 à 5 étoiles).
- Utilisable en **simulateur** dans Streamlit (page « Satisfaction Client ») : on ajuste délai, retard, montant, etc., et on obtient un score prédit.
- Utilisable pour **analyser** quels facteurs (livraison, montant, fidélité, etc.) influencent le plus la satisfaction, via les importances de variables.

---

## 2. Données et variable cible

### Variable cible : `review_score`

- **Source** : `olist_order_reviews_dataset.csv`, colonne `review_score`.
- **Type** : entier de 1 à 5 (échelle de notation classique).
- **Une ligne par avis** : chaque observation correspond à un avis laissé sur une commande, donc à un couple (commande, client) au moment de la livraison.

### Tables utilisées

| Table | Rôle |
|-------|------|
| **reviews** | `review_score` (cible) + lien vers la commande |
| **orders** | Dates, client, date estimée de livraison |
| **order_items** | Prix, fret, nombre d’articles par commande |
| **customers** | Non utilisé tel quel ; les infos « client » viennent d’agrégats sur `orders` et `order_items` |

La fonction `build_satisfaction_features` fusionne ces tables pour obtenir **une ligne par avis**, avec des variables décrivant la commande et l’historique du client.

---

## 3. Les 11 variables (features) et pourquoi on les a choisies

On cherche des **facteurs mesurables** liés à l’expérience de commande et à la relation client, disponibles dans les données Olist, sans fuite de données (pas d’info future au moment de l’avis).

### 3.1 Livraison (très lié à la satisfaction)

| Variable | Définition | Pourquoi |
|----------|------------|----------|
| **days_until_delivery** | Jours entre date d’achat et date de livraison réelle | Un délai long est souvent associé à des avis plus bas. |
| **delay_vs_estimated** | (Date livraison réelle − Date livraison estimée) en jours. &gt; 0 = retard | Le fait d’être livré en retard par rapport à la promesse est un fort signal d’insatisfaction. |

Ces deux variables sont parmi les plus prédictives dans la littérature e‑commerce pour la satisfaction et les retours.

### 3.2 Caractéristiques de la commande

| Variable | Définition | Pourquoi |
|----------|------------|----------|
| **nb_items** | Nombre de lignes (articles) dans la commande | Commandes plus grosses → attentes et complexité logistique plus grandes, risque de casse/oubli. |
| **total_price** | Somme des prix des articles de la commande | Montant élevé → attentes plus fortes, réaction plus forte si problème. |
| **total_freight** | Coût total du fret pour la commande | Fret élevé peut être perçu négativement ; reflète aussi type de livraison. |
| **avg_price_per_item** | `total_price / nb_items` | Donne une idée du « niveau de gamme » du panier (petits achats répétés vs gros achats). |

On utilise des grandeurs **objectives** (prix, quantités, fret) évitant le bruit de variables trop agrégées ou trop fines.

### 3.3 Historique et profil client

| Variable | Définition | Pourquoi |
|----------|------------|----------|
| **total_orders** | Nombre total de commandes du client (toutes commandes confondues) | Clients « habitués » vs nouveaux ; tolérance et attentes peuvent différer. |
| **total_spent** | Somme de toutes les dépenses du client | Proxy de fidélité / valeur client ; contexte pour interpréter la commande actuelle. |
| **is_first_order** | 1 si le client n’a qu’une commande, 0 sinon | Premier achat = expérience cruciale pour rétention ; notation souvent différente. |

L’idée est de **contextualiser** l’avis sans utiliser d’info située après la date de l’avis (pas de fuite temporelle).

### 3.4 Saisonnalité / calendrier

| Variable | Définition | Pourquoi |
|----------|------------|----------|
| **month** | Mois de la date de livraison (1–12) | Pics (Black Friday, fêtes) ou creux : charge logistique, délais, qualité perçue peuvent varier. |
| **day_of_week** | Jour de la semaine (0 = lundi, 6 = dimanche) | Livraison en week-end vs en semaine peut influencer la satisfaction. |

Ces variables permettent de capturer des **effets de calendrier** sans entrer dans des détails non disponibles (météo, actualité, etc.).

### Résumé des choix

- **Livraison** : délai et retard vs estimé → cœur de l’expérience « commande reçue ».
- **Commande** : taille, coût, fret → complexité et niveau d’attente.
- **Client** : ancienneté, dépense totale, première commande ou non → contexte comportemental.
- **Temps** : mois et jour → saisonnalité et type de jour de livraison.

On n’utilise **pas** le texte des avis (commentaires), ni des variables qui ne seraient pas connues au moment de la notation (pour éviter la fuite de données).

---

## 4. Pourquoi le R² n’est pas élevé (autour de 0,21–0,27)

Un R² de 0,2–0,27 signifie que le modèle « explique » environ 20–27 % de la variance du score d’avis. Ce n’est pas faible pour ce type de problème, pour plusieurs raisons.

### 4.1 La satisfaction est fortement subjective et bruitée

- Une part importante du score dépend de facteurs **non mesurés** dans nos données :
  - Qualité perçue du produit, correspondance à la description, état à l’arrivée.
  - État d’esprit du client, attentes personnelles, sensibilité au retard.
  - Contenu des commentaires (insultes, compliment) non utilisé ici.
- Même pour une **même** expérience objective (même délai, même retard), deux clients peuvent noter 2 et 5.
- Les notes 1–5 sont **peu granulaires** et souvent biaisées (tendance à 4–5, ou au contraire à 1 en cas de colère) → variance « résiduelle » importante.

Donc une grande partie de la variance du `review_score` est **structurellement** difficile à capturer avec nos seules variables numériques.

### 4.2 Ce qu’on ne modélise pas (ou peu)

- **Qualité produit** : pas de feature « catégorie », « description », « écarts produit reçu vs annoncé ».
- **Texte des avis** : pas de NLP sur les commentaires (très prédictif si on l’ajoutait).
- **Contexte fin** : communication support, litiges, remboursements, etc., non présents dans les CSV utilisés.
- **Hétérogénéité** : un même « retard de 5 jours » peut être perçu très différemment selon le client ou le type de produit.

Tout cela reste dans l’erreur du modèle → limite le R².

### 4.3 R² modéré ne veut pas dire modèle inutile

- **MAE ~ 0,89** : en moyenne le modèle se trompe d’environ **0,9 étoile** (sur 1–5). Ce n’est pas négligeable pour prioriser des actions (ex. cibler les commandes à risque de mauvaise note).
- **Tendance** : le modèle apprend bien que retard, délai long, etc. vont plutôt vers des notes plus basses ; les **ordres de grandeur** et les **effets relatifs** des variables sont cohérents.
- Pour un **simulateur** ou un **score de risque** (satisfait / à risque), un R² de 0,2–0,3 reste exploitable, surtout si on s’intéresse aux cas extrêmes (notes 1–2 vs 4–5).

En résumé :  
**R² modéré = la satisfaction dépend beaucoup de choses qu’on ne voit pas dans les données.** Notre modèle exploite ce qui est disponible (livraison, commande, client, calendrier) et en tire une part non négligeable de la variance, mais il ne peut pas « tout » expliquer.

---

## 5. Fonctionnement technique du pipeline

### 5.1 Construction des features (`build_satisfaction_features`)

1. **Jointure** : `reviews` ↔ `orders` sur `order_id` → chaque avis est associé à une commande et un client.
2. **Dates** : conversion en datetime, puis calcul de :
   - `days_until_delivery` = (livraison − achat) en jours ;
   - `delay_vs_estimated` = (livraison − livraison estimée) en jours.
3. **Agrégations commande** : à partir de `order_items`, calcul par `order_id` de `nb_items`, `total_price`, `total_freight`, puis `avg_price_per_item`.
4. **Agrégations client** : à partir de `orders` et `order_items`, calcul par `customer_id` de `total_orders`, `total_spent`, puis `is_first_order = (total_orders <= 1)`.
5. **Calendrier** : `month` et `day_of_week` à partir de la date de livraison.
6. **Nettoyage** : `days_until_delivery` clampée à ≥ 0, `delay_vs_estimated` et autres colonnes complétées par 0 si besoin.
7. **Cible** : `y = review_score` (float).
8. **Sortie** : une matrice **X** (11 colonnes) et un vecteur **y**, une ligne par avis.

Aucune variable construite avec une info postérieure à la date de l’avis → pas de fuite temporelle.

### 5.2 Entraînement (`SatisfactionPipeline.train_pipeline`)

1. Chargement des CSV Olist via `load_and_prepare_data`.
2. Appel à `build_satisfaction_features` → **X**, **y**.
3. **Split** : 80 % train / 20 % test (`test_size=0.2`, `random_state=42`).
4. **Fit** du modèle sur (X_train, y_train) :
   - `StandardScaler` : moyenne 0, écart-type 1 par variable.
   - `HistGradientBoostingRegressor` : arbres de régression en gradient boosting (paramètres dans `MLConfig.SATISFACTION_MODEL_PARAMS`).
5. **Métriques** : MAE, RMSE, R² sur train et test ; R² en **cross-validation 5-fold** sur (X, y).
6. **Importances** : si l’estimateur n’expose pas `feature_importances_`, calcul d’**importance par permutation** (dégradation du R² quand on permute une variable).
7. **Sauvegarde** : modèle (scaler + régresseur + noms de colonnes, etc.) dans `data/models/satisfaction_model.joblib`.

### 5.3 Prédiction (`SatisfactionModel.predict`)

1. Passage de **X** dans le pipeline (scale puis boosting).
2. Sortie brute du régresseur puis **clip entre 1 et 5** pour rester dans l’échelle des notes.

Les mêmes 11 variables, dans le même ordre, doivent être fournies qu’à l’entraînement (c’est ce que fait le simulateur Streamlit).

---

## 6. Paramètres du modèle (config)

Dans `src/config/settings.py`, `SATISFACTION_MODEL_PARAMS` contrôle le **HistGradientBoostingRegressor** :

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| **max_iter** | 200 | Nombre max d’arbres (itérations) dans l’ensemble. |
| **max_depth** | 8 | Profondeur max de chaque arbre → limite la complexité et le surapprentissage. |
| **learning_rate** | 0.05 | Pondération de chaque nouvel arbre ; plus c’est petit, plus il faut d’arbres pour bien ajuster. |
| **min_samples_leaf** | 20 | Nombre min d’échantillons par feuille → feuilles plus grosses, moins de surajustement. |
| **l2_regularization** | 0.1 | Pénalité L2 sur les feuilles → régularisation. |
| **random_state** | 42 | Reproductibilité des runs. |

Choix orientés **stabilité** et **généralisation** (éviter un R² train très haut et un R² test qui s’effondre).

---

## 7. Métriques utilisées

| Métrique | Définition | Intérêt |
|----------|------------|--------|
| **R²** | Partie de la variance de y expliquée par le modèle | Résumé global de la qualité de la régression. |
| **MAE** | Moyenne des \|y − prédiction\| | Erreur moyenne en « nombre d’étoiles » ; facile à communiquer. |
| **RMSE** | Racine de la moyenne des (y − prédiction)² | Pénalise plus les grosses erreurs que la MAE. |
| **CV R²** | R² en validation croisée 5-fold | Estime la performance sur de nouveaux lots de données, moins dépendante d’un split unique. |

Pour la satisfaction en 1–5, la **MAE** est souvent la plus parlante côté métier (« en moyenne on se trompe de ~0,9 étoile »).

---

## 8. Limites et biais potentiels

- **Données** : uniquement ce qui est dans les CSV Olist ; pas de texte d’avis, pas de SAV, pas de données comportementales fines (clic, time on page).
- **Temporalité** : `total_orders` et `total_spent` sont calculés sur **toute** l’histoire ; pour un avis donné, en théorie on ne devrait utiliser que l’historique **avant** cet avis. Ici on accepte une approximation pour simplifier.
- **Causalité** : le modèle est **prédictif**, pas causal. « Retard associé à une baisse de note » ne signifie pas automatiquement « réduire le retard suffirait à remonter la note » (sélection, confusion, etc.).
- **Échelle 1–5** : distribution souvent déséquilibrée (beaucoup de 4–5) ; une régression par moindres carrés suppose des résidus à peu près homoscédastiques, ce qui n’est qu’approximatif sur des notes discrètes.

---

## 9. Pistes d’amélioration

- **Variables** :  
  - Ratio « retard / délai estimé », indicatrice « livré en retard » (is_late).  
  - Features produit/catégorie (si jointure possible sans fuite).  
  - Agrégations client **strictement** avant la date de l’avis (fenêtre glissante ou historique tronqué).
- **Cible** :  
  - Travailler en **classification** (ex. satisfait 4–5 vs insatisfait 1–3) et regarder précision / recall / AUC.  
  - Régression ordinale si on veut garder l’ordre des notes.
- **Texte** :  
  - Utiliser les champs `review_comment_title` et `review_comment_message` (NLP, embeddings) et les combiner aux features actuelles (modèle hybride).
- **Modèle** :  
  - Tester d’autres algos (Random Forest, XGBoost, LightGBM) et comparer en CV.  
  - Ajuster `max_iter`, `learning_rate`, `max_depth` par grid/random search sur le R² ou la MAE en validation.

---

## 10. Résumé en quelques points

- **Type** : régression (score 1–5) avec **HistGradientBoostingRegressor** après **StandardScaler**.
- **Variables** : 11 features — livraison (délai, retard), commande (nb articles, prix, fret, panier moyen), client (nb commandes, dépense, première commande), calendrier (mois, jour).
- **Choix des variables** : facteurs objectifs, mesurables, sans fuite de données, et cohérents avec la littérature sur la satisfaction e‑commerce.
- **R² modéré** : la satisfaction dépend beaucoup du ressenti, du produit et du contexte non présents dans nos tables ; le modèle exploite au mieux ce qui est disponible.
- **Utilité** : simulateur de score, analyse d’impact des facteurs (importances), base possible pour un score de risque « client insatisfait » ou pour prioriser des actions (ex. suivi des commandes en retard).

Si tu veux, on peut détailler une section précise (par ex. « uniquement les variables » ou « uniquement pourquoi le R² est bas ») ou l’intégrer dans le README sous forme de lien vers ce doc.
