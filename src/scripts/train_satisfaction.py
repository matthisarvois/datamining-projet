"""
Point d'entrée pour l'entraînement du modèle de satisfaction client.
Usage: uv run python src/scripts/train_satisfaction.py [--data-dir PATH]
"""

import sys

from src.ml_pipeline.train_satisfaction import main as _train_main


def main():
    """Délègue à ml_pipeline.train_satisfaction.main."""
    return _train_main()


if __name__ == "__main__":
    sys.exit(main())
