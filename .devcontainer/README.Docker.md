# Docker – Olist Recommendation System

Setup Docker dans **`.devcontainer`** : UV, pre-commit, pytest, setup (données Olist) et entraînement.

## Lancer l’appli

Depuis la **racine du projet** :

```bash
docker compose -f .devcontainer/compose.yaml up --build
```

- **Backend** : http://localhost:8000 | /docs | /api/v1/health  
- **Frontend** : http://localhost:8501  

## Commandes utiles

```bash
# En arrière-plan
docker compose -f .devcontainer/compose.yaml up --build -d

# Tests
docker compose -f .devcontainer/compose.yaml run --rm backend uv run pytest tests/ -v

# Pre-commit
docker compose -f .devcontainer/compose.yaml run --rm backend uv run pre-commit run --all-files
```

Voir le **README.md** (section *Avec Docker*) pour plus de détails.
