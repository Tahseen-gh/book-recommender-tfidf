"""Semantic-search backend: sentence embeddings + cosine similarity.

Mirrors recommender.py's two-step search (find the closest book to the
query, then rank every book against that book) but swaps TF-IDF vectors
for dense embeddings from a SentenceTransformer model. Unlike the TF-IDF
backend, which fits a separate title-only vectorizer to anchor the query,
this backend encodes only `combined_text` and uses that same embedding
space for both steps - see README for why.

Embeddings are cached to disk after the first run, keyed off the CSV's
size and modification time plus the model name, so a changed dataset or
model doesn't silently serve a stale cache.
"""

import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import clean_text

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "embeddings"


def _cache_path(csv_path):
    """Build a cache file path keyed off the CSV's identity and the model name.

    Using size + mtime (not a full content hash) means the cache doesn't
    need to read the ~11,000-row CSV a second time just to invalidate itself.
    """
    csv_path = Path(csv_path)
    stat = csv_path.stat()
    key = f"{csv_path.resolve()}|{stat.st_size}|{stat.st_mtime}|{MODEL_NAME}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.npy"


class EmbeddingRecommender:
    """Fits a SentenceTransformer embedding space over a books DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of data_loader.load_books - must have 'combined_text',
        'title', 'authors', and 'average_rating' columns.
    csv_path : str or Path
        Path to the source CSV, used only to key the on-disk embedding cache.
    """

    def __init__(self, df, csv_path):
        self.df = df
        self.model = SentenceTransformer(MODEL_NAME)

        cache_file = _cache_path(csv_path)
        if cache_file.exists():
            self.embeddings = np.load(cache_file)
        else:
            self.embeddings = self.model.encode(
                df['combined_text'].tolist(), show_progress_bar=False
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self.embeddings)

        self.cosine_sim_combined = cosine_similarity(self.embeddings, self.embeddings)

    def get_recommendations(self, query, cosine_sim=None):
        """Return the top 10 books most similar to `query`'s closest match.

        Same two-step shape as BookRecommender.get_recommendations: find
        the single book whose embedding is closest to the query's
        embedding, then rank every book against that book using
        `cosine_sim` - the full combined_text similarity matrix. Defaults
        to `self.cosine_sim_combined`.

        Returns a DataFrame with columns: title, authors, average_rating.
        """
        if cosine_sim is None:
            cosine_sim = self.cosine_sim_combined

        query_cleaned = clean_text(query)
        query_embedding = self.model.encode([query_cleaned], show_progress_bar=False)
        query_similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        best_match_index = query_similarities.argmax()

        combined_similarities = cosine_sim[best_match_index]
        similar_indices = combined_similarities.argsort()[-10:][::-1]
        recommended_books = self.df.iloc[similar_indices][['title', 'authors', 'average_rating']]
        return recommended_books
