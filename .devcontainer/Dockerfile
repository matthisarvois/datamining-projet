# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Dépendances système (curl pour healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer UV (gestionnaire de paquets rapide)
RUN pip install --no-cache-dir uv

# Copier les fichiers de dépendances (context = racine projet)
COPY pyproject.toml uv.lock .python-version ./
COPY .pre-commit-config.yaml ./
# README.md requis par hatchling pour le build editable (readme = "README.md" dans pyproject.toml)
COPY README.md ./

# Installer les dépendances avec UV (prod + dev + test : pre-commit, ruff, pytest, etc.)
RUN uv sync --frozen --extra dev --extra test

# Copier le code source
COPY . .

# Créer l'utilisateur non-privilégié
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Setup + entraînement (données Olist + modèles ML : recommandation + satisfaction)
RUN uv run python src/scripts/setup.py \
    && uv run python src/scripts/train.py \
    && uv run python src/scripts/train_satisfaction.py

# Donner à appuser la propriété de tout /app (dont .venv) pour que uv run puisse modifier l’install editable au démarrage
RUN mkdir -p /app/.cache \
    && chown -R appuser:appuser /app

# uv et autres outils utilisent $HOME/.cache : définir HOME=/app pour l’utilisateur appuser
ENV HOME=/app

USER appuser

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "src.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
