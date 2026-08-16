"""Load the books CSV and clean its text fields for vectorization.

This is a direct port of the preprocessing cell from the original notebook
(IDriven_Book_Shopping_Recommendation_Engine.ipynb) - same regex, same
stopword filtering, same column combination. Only the NLTK download has
been made conditional so it doesn't hit the network on every run.
"""

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords


def ensure_nltk_stopwords():
    """Download the NLTK stopwords corpus if it isn't already cached locally.

    The original notebook called nltk.download('stopwords') unconditionally
    on every run. This checks first so a warm cache (e.g. after the first
    run, or in a Docker image that pre-seeds NLTK data) skips the network
    call entirely.
    """
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


ensure_nltk_stopwords()


def clean_text(text):
    """Strip non-letters, lowercase, and drop English stopwords.

    Note: this re-fetches stopwords.words('english') on every word during
    filtering rather than loading it once - a pre-existing inefficiency
    carried over unchanged from the original notebook.

    Non-string input (e.g. NaN from a missing CSV field) returns ''.
    """
    if isinstance(text, str):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        words = [w for w in words if not w in stopwords.words('english')]
        return ' '.join(words)
    else:
        return ''


def load_books(csv_path):
    """Load books.csv and add the cleaned/combined text columns.

    Reproduces the original notebook's preprocessing step: cleans title,
    authors, publisher, and language_code independently, then concatenates
    them (in that order, space-separated) into 'combined_text'.

    Expects a CSV with at least: title, authors, publisher, language_code,
    average_rating columns (see data/README.md for the expected dataset).
    """
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_authors'] = df['authors'].apply(clean_text)
    df['clean_publisher'] = df['publisher'].apply(clean_text)
    df['clean_language'] = df['language_code'].apply(clean_text)

    df['combined_text'] = (
        df['clean_title'] + ' ' + df['clean_authors'] + ' '
        + df['clean_publisher'] + ' ' + df['clean_language']
    )

    return df
