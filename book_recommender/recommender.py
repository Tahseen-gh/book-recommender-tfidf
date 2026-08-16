"""TF-IDF vectorization and cosine-similarity scoring.

Direct port of the vectorization/recommendation cell from the original
notebook. Two separate TF-IDF spaces are built:

- tfidf_combined: title + authors + publisher + language_code, capped at
  5000 features. Used to compute cosine_sim_combined at fit time.
- tfidf_title: title text only, unbounded vocabulary. Used to score
  incoming queries in get_recommendations.

The combined-text space is fit and its full pairwise similarity matrix is
computed, but get_recommendations does not read from it - see the
'cosine_sim' parameter below and the project README for details. That
behavior is unchanged from the original notebook.
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
        """Return the top 10 books whose titles are closest to `query`.

        `cosine_sim` mirrors the original notebook's function signature,
        which defaulted to the precomputed combined-text similarity matrix.
        It's accepted here for the same interface parity but - as in the
        source notebook - is never read inside this method; scoring is
        done entirely via a fresh query-vs-title-corpus similarity below.

        Returns a DataFrame with columns: title, authors, average_rating.
        """
        if cosine_sim is None:
            cosine_sim = self.cosine_sim_combined

        query_cleaned = clean_text(query)
        query_vector = self.tfidf_title.transform([query_cleaned])
        title_similarities = cosine_similarity(query_vector, self.tfidf_title_matrix).flatten()

        similar_indices = title_similarities.argsort()[-10:][::-1]
        recommended_books = self.df.iloc[similar_indices][['title', 'authors', 'average_rating']]
        return recommended_books
