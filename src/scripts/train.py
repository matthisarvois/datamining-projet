"""
Point d'entrée pour l'entraînement du modèle.
Usage: uv run python src/scripts/train.py [--data-dir PATH]  ou  uv run olist-train
"""

import sys

from src.ml_pipeline.train_model import main as _train_main


def main():
    """Délègue à ml_pipeline.train_model.main (pour olist-train)."""
    return _train_main() or 0


if __name__ == "__main__":
    sys.exit(main())
