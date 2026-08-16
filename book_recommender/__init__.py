"""TF-IDF + cosine similarity book recommendation engine."""

from .data_loader import load_books
from .recommender import BookRecommender

__all__ = ["load_books", "BookRecommender"]
