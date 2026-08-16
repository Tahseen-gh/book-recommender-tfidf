"""Gradio entry point.

Wires data_loader and recommender together and reproduces the original
notebook's gr.Interface (same inputs, outputs, title, and description).
Run with `python run.py` from the repo root.
"""

from pathlib import Path

import gradio as gr

from .data_loader import load_books
from .recommender import BookRecommender

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "books.csv"


def build_interface(csv_path=DATA_PATH, backend='tfidf'):
    """Load the dataset, fit the recommender for `backend`, and return a gr.Interface."""
    df = load_books(csv_path)

    if backend == 'embeddings':
        from .embeddings import EmbeddingRecommender
        recommender = EmbeddingRecommender(df, csv_path)
    else:
        recommender = BookRecommender(df)

    def recommend_books(query):
        recommendations = recommender.get_recommendations(query)
        if recommendations.empty:
            return "No similar books found. Please try a different title."
        else:
            return recommendations.to_html(index=False)

    return gr.Interface(
        fn=recommend_books,
        inputs=gr.Textbox(lines=1, placeholder='Enter a book title'),
        outputs='html',
        title='Book Recommendation Search Engine',
        description=f'Enter a book title to get similar book recommendations. Backend: {backend}.',
    )


def main(backend='tfidf'):
    try:
        iface = build_interface(backend=backend)
    except FileNotFoundError:
        raise SystemExit(
            f"Couldn't find {DATA_PATH}.\n"
            "See data/README.md for which dataset to download and where to put it."
        )
    iface.launch()


if __name__ == "__main__":
    main()
