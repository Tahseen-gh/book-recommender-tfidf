"""TF-IDF vectorization and cosine-similarity scoring.

Vectorization is a direct port of the original notebook: two TF-IDF
spaces are built at fit time.

- tfidf_combined: title + authors + publisher + language_code, capped at
  5000 features. cosine_sim_combined is its full book-to-book similarity
  matrix.
- tfidf_title: title text only, unbounded vocabulary. Used to match a
  free-text query against the corpus, since a typed query can't be looked
  up in a book-to-book matrix directly.

get_recommendations uses both: title similarity finds the single closest
book to the typed query (partial title matching), then cosine_sim_combined
ranks every book against that match. That's what makes recommendations
reflect shared author/publisher/language, not just title text - see the
project README for the reasoning. In the original notebook, scoring was
title-only and cosine_sim_combined was computed but never read; this is
the fixed version.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import clean_text


class BookRecommender:
    """Fits TF-IDF vectorizers over a books DataFrame and serves recommendations.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of data_loader.load_books - must have 'combined_text' and
        'clean_title' columns, plus 'title', 'authors', 'average_rating'.
    """

    def __init__(self, df):
        self.df = df

        self.tfidf_combined = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_title = TfidfVectorizer(stop_words='english')

        self.tfidf_combined_matrix = self.tfidf_combined.fit_transform(df['combined_text'])
        self.tfidf_title_matrix = self.tfidf_title.fit_transform(df['clean_title'])

        self.cosine_sim_combined = cosine_similarity(
            self.tfidf_combined_matrix, self.tfidf_combined_matrix
        )

    def get_recommendations(self, query, cosine_sim=None):
        """Return the top 10 books most similar to `query`'s closest title match.

        Two steps: find the single book whose title is closest to `query`
        (partial title matching via the title-only vectorizer), then rank
        every book against that book using `cosine_sim` - the combined
        title+author+publisher+language_code similarity matrix. Defaults to
        `self.cosine_sim_combined`; accepting it as a parameter mirrors the
        original notebook's function signature.

        A query with no meaningful title overlap still anchors on whatever
        book scores highest (there's no similarity floor), so it still
        returns 10 books - just not meaningful ones. Same for a query that
        exactly matches a book already in the corpus: that book will
        anchor on itself and typically rank first in its own results,
        since a book is maximally similar to itself.

        Returns a DataFrame with columns: title, authors, average_rating.
        """
        if cosine_sim is None:
            cosine_sim = self.cosine_sim_combined

        query_cleaned = clean_text(query)
        query_vector = self.tfidf_title.transform([query_cleaned])
        title_similarities = cosine_similarity(query_vector, self.tfidf_title_matrix).flatten()
        best_match_index = title_similarities.argmax()

        combined_similarities = cosine_sim[best_match_index]
        similar_indices = combined_similarities.argsort()[-10:][::-1]
        recommended_books = self.df.iloc[similar_indices][['title', 'authors', 'average_rating']]
        return recommended_books
